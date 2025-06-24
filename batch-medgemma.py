"""
batch-medgemma.py
-----------------

This module defines a Modal app for running the MedGemma agent with vLLM, 
providing web server capabilities.

Key components:
- Modal app and volume setup for model weights
- vLLM-based model serving via web server
- Web server endpoint for inference
"""
import modal
from typing import List, Dict, Any
import os

app = modal.App("medgemma-vllm-web-server")
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = "/models"
MODEL_ID = "unsloth/medgemma-4b-it-unsloth-bnb-4bit"
MINUTES = 60
VLLM_PORT = 8000

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install(
        "vllm==0.9.1",
        "bitsandbytes",
        "flashinfer-python==0.2.6.post1",
        "huggingface-hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1", 
        "HF_HUB_CACHE": MODEL_DIR
    })
)

with image.imports():
    from vllm import LLM, SamplingParams

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("access_medgemma_hf")],
    volumes={MODEL_DIR: volume}
)
def download_model():
    """Downloads the model weights from Hugging Face Hub."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        token=os.environ["HF_TOKEN"]
    )
    return {"status": "Model downloaded successfully"}

@app.function(
    image=image,
    gpu="L4:1",
    min_containers=1,
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=15 * MINUTES,  # how long should we wait for container start?
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("access_medgemma_hf"), modal.Secret.from_name("MODAL_API_KEY")]
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Serve the MedGemma model using vLLM web server."""
    import subprocess
    
    # Download model first
    download_model.remote()
    
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_ID,
        "--served-model-name",
        "medgemma",
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
        "--api-key",
        os.environ["MODAL_API_KEY"]
    ]

    # Add tensor parallel size for single GPU
    cmd += ["--tensor-parallel-size", "1"]

    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)