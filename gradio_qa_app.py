"""
CareHandoff AI — Gradio Q&A App
Standalone Clinical Q&A interface mounted at /qa by FastAPI.
Uses a lazy chat wrapper so the Gradio blocks can be created at module
import time (before AppState is initialized by the FastAPI lifespan).
"""
import gradio as gr


def create_qa_blocks() -> gr.Blocks:
    def _chat(message: str, history: list) -> str:
        # Resolve state lazily at request time, not at block-creation time.
        from api.state import get_state
        return get_state().guarded_rag.chat(message, history)

    with gr.Blocks(title="Clinical Q&A — CareHandoff AI") as blocks:
        gr.ChatInterface(
            fn=_chat,
            chatbot=gr.Chatbot(label="Clinical Documentation Assistant", height=520),
            textbox=gr.Textbox(
                placeholder="e.g. What are the documentation requirements for heart failure discharge?",
                container=False,
                scale=7,
            ),
            examples=[
                "What is the recommended documentation for heart failure discharge?",
                "What are the guidelines for hypertension management in discharge notes?",
                "Is it required to document medication reconciliation at discharge?",
                "Give me a comprehensive overview of all discharge documentation requirements.",
                "What lab values should be documented for patients with AKI?",
                "How should COPD exacerbation be documented at discharge?",
            ],
            cache_examples=False,
        )

    return blocks
