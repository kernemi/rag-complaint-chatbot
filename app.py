import sys
sys.path.append("./src")

import gradio as gr
from transformers import pipeline

from ragUtilities import (
    load_embedding_model,
    load_chroma_collection,
    rag_pipeline
)

# -----------------------------
# Initialize Once (Important)
# -----------------------------
embedder = load_embedding_model()

collection = load_chroma_collection(
    persist_dir="./vector_store",
    collection_name="complaints_full"
)

llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)


# -----------------------------
# Gradio Function
# -----------------------------
def ask_question(question):
    answer, docs, metas = rag_pipeline(
        question,
        collection,
        embedder,
        llm,
        top_k=5
    )

    sources = "\n\n".join(
        [f"Source {i+1}:\n{doc[:500]}" for i, doc in enumerate(docs[:2])]
    )

    return answer, sources


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("# 💬 CrediTrust Complaint Analysis Assistant")

    question = gr.Textbox(
        label="Ask a question about customer complaints",
        placeholder="e.g. What problems do customers report about credit cards?"
    )

    ask_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear")

    answer = gr.Textbox(label="AI Answer", lines=8)
    sources = gr.Textbox(label="Sources Used", lines=10)

    ask_btn.click(
        ask_question,
        inputs=question,
        outputs=[answer, sources]
    )

    clear_btn.click(
        lambda: ("", ""),
        outputs=[answer, sources]
    )

demo.launch()
