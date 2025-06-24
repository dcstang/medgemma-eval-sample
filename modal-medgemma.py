"""
modal-medgemma.py
-----------------

This module defines a Modal app and FastAPI endpoint for running the MedGemma agent, a multimodal LLM that can process text and images. It provides a streaming API for inference, including Wikipedia tool-calling capabilities, and handles model download, loading, and inference with GPU support.

Nb. needs to be deployed with the following command:
`modal deploy modal-medgemma.py`

Key components:
- Modal app and volume setup for model weights
- MedGemmaAgent class for model loading and inference
- FastAPI endpoint for streaming responses
- Utility for processing base64-encoded images
"""
import modal
from typing import Optional, Generator, Dict, Any, List
import os
from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import base64
from PIL import Image
import io

app = modal.App("example-medgemma-agent")
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = "/models"
# MODEL_ID = "google/medgemma-4b-it"
MODEL_ID = "unsloth/medgemma-4b-it-unsloth-bnb-4bit"
MINUTES = 60

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validates the provided API key against the environment variable.

    Args:
        api_key_header (str): The API key provided in the request header.

    Raises:
        HTTPException: If the API key is invalid.

    Returns:
        str: The validated API key.
    """
    if api_key_header != os.environ["FASTAPI_KEY"]:
        raise HTTPException(
            status_code=403, detail="Invalid API Key"
        )
    return api_key_header

image = (
    modal.Image.debian_slim()
    .pip_install(
        "smolagents[vllm]",
        "fastapi[standard]",
        "wikipedia-api",
        "accelerate",
        "bitsandbytes",
        "huggingface-hub[hf_transfer]",
        "Pillow")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1", 
        "HF_HUB_CACHE": MODEL_DIR
    })
)

with image.imports():
    from smolagents import VLLMModel, ToolCallingAgent, tool
    from pydantic import BaseModel
    import wikipediaapi

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("access_medgemma_hf")],
    volumes={MODEL_DIR: volume}
)
def download_model():
    """
    Downloads the model weights from Hugging Face Hub using the provided token.

    Returns:
        dict: Status message indicating success.
    """
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        token=os.environ["HF_TOKEN"]
    )
    return {"status": "Model downloaded successfully"}

@app.cls(
    image=image,
    gpu="L4:1",
    volumes={MODEL_DIR: volume},
    min_containers=1,
    max_containers=1,
    timeout=15 * MINUTES,
    secrets=[modal.Secret.from_name("access_medgemma_hf")],
)
class MedGemmaAgent:
    """
    Modal class for managing the MedGemma model and running inference with optional tool-calling.
    """
    @modal.enter()
    def load_models(self):
        """
        Loads the MedGemma model into memory and prepares it for inference.
        Downloads the model weights if not already present.
        """
        download_model.remote()
        model_kwargs = {
            "max_model_len": 8192,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.95,
            "tensor_parallel_size": 1,
            "trust_remote_code": True
        }
        self.model = VLLMModel(
            model_id=MODEL_ID,
            model_kwargs=model_kwargs
        )
        print(f"Model: {MODEL_ID} loaded successfully")

    @modal.method()
    def run(self, prompt: str, images: Optional[List[Image.Image]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Runs the MedGemma agent on the provided prompt and optional images, yielding streaming responses.

        Args:
            prompt (str): The user prompt to process.
            images (Optional[List[Image.Image]]): List of PIL Images to provide as context (optional).

        Yields:
            Dict[str, Any]: Streaming response chunks, including 'thinking' and 'final' messages.
        """
        @tool
        def wiki(query: str) -> str:
            """
            Fetches a summary of a Wikipedia page based on a given search query (only one word or group of words).

            Args:
                query: The search term for the Wikipedia page (only one word or group of words).
            """
            wiki = wikipediaapi.Wikipedia(language="en", user_agent="MinimalAgent/1.0")
            page = wiki.page(query)
            if not page.exists():
                return "No Wikipedia page found."
            return page.summary[:1000]

        self.agent = ToolCallingAgent(
            tools=[wiki],
            model=self.model,
            max_steps=3
        )

        # Yield thinking step
        yield {
            "type": "thinking",
            "content": {"message": "Starting to process your request..."}
        }

        # Run the agent and capture the result
        result = self.agent.run(
            task=prompt,
            stream=False,
            reset=True,
            images=images if images else None,
            additional_args={"flatten_messages_as_text": False},
            max_steps=3
        )
        
        # Yield the final response
        yield {
            "type": "final",
            "content": {"response": result}
        }

class PromptRequest(BaseModel):
    """
    Request model for the /run_medgemma endpoint.

    Attributes:
        prompt (str): The user prompt to process.
        image (Optional[str]): Base64-encoded image string (optional).
        history (Optional[list]): Conversation history (optional).
    """
    prompt: str
    image: Optional[str] = None  # Base64 encoded image
    history: Optional[list] = None

class GenerationResponse(BaseModel):
    """
    Response model for non-streaming generation (not used in this file).

    Attributes:
        response (str): The generated response.
    """
    response: str

class StreamResponse(BaseModel):
    """
    Response model for streaming responses.

    Attributes:
        type (str): The type of message ('thinking', 'tool_call', 'tool_result', 'final').
        content (Dict[str, Any]): The content of the message.
    """
    type: str  # 'thinking', 'tool_call', 'tool_result', 'final'
    content: Dict[str, Any]

def process_image(image_base64: Optional[str]) -> Optional[Image.Image]:
    """
    Decodes a base64-encoded image string into a PIL Image.

    Args:
        image_base64 (Optional[str]): Base64-encoded image string.

    Returns:
        Optional[Image.Image]: The decoded PIL Image, or None if decoding fails or input is None.
    """
    if not image_base64:
        return None
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("access_medgemma_hf"),
        modal.Secret.from_name("FASTAPI_KEY")
    ]
)
@modal.fastapi_endpoint(method="POST")
async def run_medgemma(request: PromptRequest, api_key: str = Depends(get_api_key)):
    """
    FastAPI endpoint for running the MedGemma agent with streaming responses.

    Args:
        request (PromptRequest): The request payload containing prompt and optional image.
        api_key (str): The validated API key (injected by Depends).

    Returns:
        StreamingResponse: An event-stream response yielding model output chunks.
    """
    model_handler = MedGemmaAgent()
    
    # Process image if provided
    image = process_image(request.image)
    images = [image] if image else None
    
    async def generate():
        stream = model_handler.run.remote_gen.aio(request.prompt, images=images)
        async for chunk in stream:
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )