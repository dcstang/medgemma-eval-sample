import gradio as gr
import httpx
import os
import json
import base64
from PIL import Image
import io
import docs  

API_KEY = os.getenv("API_KEY")  
MODAL_API_ENDPOINT = os.getenv("MODAL_API_ENDPOINT")

def encode_image_to_base64(image):
    if image is None:
        return None
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

async def call_my_api(message, history, image=None):
    # Support multimodal: message can be dict with 'text' and 'files'
    user_text = message["text"] if isinstance(message, dict) else message
    if user_text.strip().lower().startswith("(example)"):
        user_text = user_text.strip()[9:].lstrip()
    image_obj = None
    
    if image is None and isinstance(message, dict) and message.get("files"):
        # message["files"] is a list of file paths or file objects
        # For Gradio, it may be a list of PIL Images
        image_obj = message["files"][0] if message["files"] else None
    else:
        image_obj = image

    if isinstance(image_obj, str):
        image_obj = Image.open(image_obj)

    image_base64 = encode_image_to_base64(image_obj) if image_obj else None

    payload = {
        "prompt": "You are a helpful and positive health coach. Explain in simple terms to a patient or non-medical person on the following question or statement.\n\n" + user_text,
        "image": image_base64
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', MODAL_API_ENDPOINT, json=payload, headers=headers, timeout=120.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            if data['type'] == 'final':
                                yield data['content']['response']
                            elif data['type'] == 'thinking':
                                yield data['content'].get('message', '')
                            elif data['type'] == 'tool_call':
                                yield f"Using tool: {data['content'].get('name', '')}"
                            elif data['type'] == 'tool_result':
                                yield f"Tool result: {data['content'].get('result', '')}"
                        except json.JSONDecodeError:
                            continue
    except httpx.RequestError as e:
        print(f"Error calling API: {e}")
        yield f"Error: Could not connect to the API. {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        yield "An unexpected error occurred."


def vote(chatbot, history, vote):
    print(f"Vote: {vote}")
    print(f"Chatbot: {chatbot}")
    print(f"History: {history}")
    return chatbot, history

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "sans-serif"],
    ),
    css_paths=["assets/custom.css"],
    title="Agent medGemma"
) as demo:
    chatbot = gr.Chatbot(
        placeholder="Ask me anything about a medical condition.",
        type="messages",
        height=600
    )
    chatbot.like(vote, inputs=[chatbot, gr.State([]), gr.State("")], outputs=[chatbot, gr.State([])])

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("# 🏥 Agent MedGemma", elem_id="main-title")
        with gr.Column(scale=3):
            gr.Markdown(
                "<div class='tagline'>Simple and accessible medical facts</div>",
                elem_id="main-tagline"
            )

    with gr.Row():
        with gr.Column(scale=1, elem_id="chat-col"):
            chat_interface = gr.ChatInterface(
                multimodal=False,
                fn=call_my_api,
                chatbot=chatbot,
                theme="soft",
                examples=[
                    "(example) Tell me about the causes of a heart attack.",
                    "(example) What should I do with serious vomiting?",
                    "(example) Should I take double my insulin now that I forgot to take it?",
                    "(example) What are the most common symptoms of a stroke?"
                ],
                example_icons=["assets/heartbreak.svg",
                                "assets/sickface.svg",
                                "assets/injection.svg",
                                "assets/brain.svg"]
            )

    gr.Markdown(
        """
        <div class='disclaimer-box'>
            <p> Modal backend is turned off since completion of hackathon. Host your own Modal LLM endpoint by referring to the .py files. </p>
            <p>
                <strong>Medical Disclaimer:</strong> This AI assistant is designed for educational and informational purposes only. 
                It does not constitute medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals 
                for medical decisions. This tool aims to promote health literacy and empower individuals to better understand their 
                health, but should not replace professional medical consultation. See below link for more on the technical details.
            </p>
        </div>
        """,
        elem_id="disclaimer"
    )

    gr.Markdown(
        """
        <div style='text-align: center; margin-top: 20px; padding: 10px; border-top: 1px solid #e0e0e0;'>
            <a href='https://agents-mcp-hackathon-agentic-coach-advisor-medgemma.hf.space/docs' target="_blank" rel="noopener" style='text-decoration: none; color: #666;'>📚 View Technical Documentation</a>
        </div>
        """,
        elem_id="footer"
    )

with demo.route("Technical Documentation", "/docs"):
    docs.docs_demo.render()

if __name__ == "__main__":
    demo.launch(favicon_path="assets/medical_cross_icon_144218.ico")