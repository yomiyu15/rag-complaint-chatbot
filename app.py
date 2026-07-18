"""
Task 4 - Interactive chat interface for the CrediTrust complaint RAG assistant.

A Gradio app where an internal user (Product / Support / Compliance) asks a
plain-English question, gets a synthesized answer, and sees the source complaint
excerpts the answer is grounded in (for trust and verification).

Run:
    python app.py
Then open the printed local URL in a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from rag_pipeline import RAGPipeline  # noqa: E402

# Lazy singleton so the UI loads instantly; the model loads on first question.
_rag: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline(k=5)
    return _rag


def format_sources(sources) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = ["### 📎 Sources used\n"]
    for i, s in enumerate(sources, 1):
        lines.append(
            f"**[{i}] {s.product_category}** · {s.company} · _{s.issue}_ "
            f"· cosine `{s.score:.3f}`\n\n"
            f"> {s.text[:400]}{'…' if len(s.text) > 400 else ''}\n"
        )
    return "\n".join(lines)


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    res = get_pipeline().answer(question)
    return res["answer"], format_sources(res["sources"])


def clear():
    return "", "", ""


EXAMPLES = [
    "Why are people unhappy with credit cards?",
    "What are the most common issues with money transfers?",
    "Are customers being charged unexpected fees?",
    "Do customers report unauthorized transactions?",
]

with gr.Blocks(title="CrediTrust Complaint Analyst") as demo:
    gr.Markdown(
        "# 💳 CrediTrust Complaint Analyst\n"
        "Ask plain-English questions about customer complaints across **Credit Cards, "
        "Personal Loans, Savings Accounts, and Money Transfers**. Answers are grounded "
        "in real complaint narratives, with the source excerpts shown below."
    )
    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. Why are people unhappy with credit cards?",
            lines=2, scale=5,
        )
    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")
    gr.Examples(examples=EXAMPLES, inputs=question)
    answer = gr.Markdown(label="Answer")
    sources = gr.Markdown()

    ask_btn.click(ask, inputs=question, outputs=[answer, sources])
    question.submit(ask, inputs=question, outputs=[answer, sources])
    clear_btn.click(clear, outputs=[question, answer, sources])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False,
                theme=gr.themes.Soft())
