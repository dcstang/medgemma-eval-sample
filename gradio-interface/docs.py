import gradio as gr
import os


with gr.Blocks(title="Technical Documentation", css="footer {visibility: hidden}") as docs_demo:
    
    with gr.Column():
        gr.Markdown("""
        # Technical Documentation

        ## Overview
        This page provides details about the architecture, API, and usage of the MedGemma Agent application.

        ## Features
        - Multimodal (text + image)
        - Wikipedia tool integration
        - Real-time streaming
        - Medical knowledge base

        ---

        ## Architecture
        - **Frontend:** Gradio Blocks, custom CSS
        - **Backend:** Modal, FastAPI, VLLM, MedGemma-4B
        - **Security:** API key authentication

        ### 🏗️ Technical Stack
        - Streaming responses for real-time interaction
        - Secure API key authentication
        - Base64 image processing for multimodal inputs 

        ### Frontend Interface
        - Built with Gradio for seamless user interaction
        - Custom CSS theming for professional appearance
        - Example queries for common medical scenarios            

        ```mermaid
        graph TD
            A[MedGemma Agent] --> B[Backend]
            A --> C[Frontend]
            A --> D[Model]
            
            B --> B1[Modal]
            B --> B2[FastAPI]
            B --> B3[VLLM]
            
            C --> C1[Gradio]
            C --> C2[Custom CSS]
            
            D --> D1[MedGemma-4B]
            D --> D2[4-bit Quantization]
        ```
        """)
        
        gr.Markdown("""
        ## Backend Architecture
        
        ### 🎯 Performance Features
        
        - Optimized for low latency responses
        - GPU-accelerated inference
        - Efficient memory utilization with 4-bit quantization
        - Maximum context length of 8192 tokens
        
        ### 🔒 Security Measures
        
        - API key authentication for all requests
        - Secure image processing
        - Protected model endpoints
                    
        ```mermaid
        flowchart LR
            A[Client] --> B[FastAPI]
            B --> C[Modal Container]
            C --> D[VLLM]
            D --> E[MedGemma-4B]
            B --> F[Wikipedia API]
        ```
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ## 💾 Model Deployment
                
                ### Model
                - **Model:** unsloth/medgemma-4b-it-unsloth-bnb-4bit
                - **Context Length:** 8192 tokens
                - **Quantization:** 4-bit, bfloat16
                - Utilizes Modal's GPU-accelerated containers
                - Implements efficient model loading with VLLM
                - Supports bfloat16 precision for optimal performance
                """)
            with gr.Column():
                gr.Markdown("""
                ```mermaid
                graph TD
                    A[Model Loading] --> B[GPU Acceleration]
                    B --> C[4-bit Quantization]
                    C --> D[8192 Token Context]
                    D --> E[Streaming Response]
                ```
                """)
    with gr.Column():
        gr.Markdown("""
        ## 📊 System Architecture
        
        ```mermaid
        flowchart TD
            A[User Interface] --> B[API Gateway]
            B --> C[Authentication]
            C --> D[Model Service]
            D --> E[Wikipedia Service]
            D --> F[Image Processing]
            F --> G[Model Inference]
            E --> H[Response Generation]
            G --> H
            H --> I[Stream Response]
            I --> A
        ```
        """)
        
        gr.Markdown("""
        [Back to Main Application](https://huggingface.co/spaces/Agents-MCP-Hackathon/agentic-coach-advisor-medgemma)
        """)

if __name__ == "__main__":
    docs_demo.launch()