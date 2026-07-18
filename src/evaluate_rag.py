"""
Task 3 - Qualitative evaluation of the RAG pipeline.

Runs a set of representative questions through the pipeline and records the
generated answer plus the top retrieved sources, so a scored evaluation table can
be written into the final report.

Outputs:
    reports/rag_eval_raw.json   - machine-readable answers + sources
    reports/rag_evaluation.md   - human-readable table (scores added in report)

Usage:
    python src/evaluate_rag.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "reports" / "rag_eval_raw.json"
MD_PATH = ROOT / "reports" / "rag_evaluation.md"

QUESTIONS = [
    "Why are people unhappy with credit cards?",
    "What are the most common issues customers report with money transfers?",
    "What problems do customers report with savings accounts?",
    "What are the main complaints about personal loans?",
    "Are customers being charged unexpected fees on their credit cards?",
    "Do customers report unauthorized or fraudulent transactions?",
    "What issues do customers face when trying to close an account or get a refund?",
]


def main() -> None:
    rag = RAGPipeline(k=5)
    results = []
    for q in QUESTIONS:
        print(f"\n>>> {q}")
        res = rag.answer(q)
        print("ANSWER:", res["answer"])
        top = []
        for s in res["sources"][:2]:
            top.append({
                "score": round(s.score, 3),
                "product_category": s.product_category,
                "company": s.company,
                "issue": s.issue,
                "excerpt": s.text[:200],
            })
        results.append({"question": q, "answer": res["answer"], "top_sources": top})

    RAW_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved raw eval -> {RAW_PATH}")

    # draft markdown table (Quality Score / Comments filled in the final report)
    lines = [
        "# RAG Qualitative Evaluation\n",
        "| # | Question | Generated Answer | Top Retrieved Source (1) | Quality (1-5) | Comments |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        ans = r["answer"].replace("\n", " ").replace("|", "\\|")
        src = ""
        if r["top_sources"]:
            s0 = r["top_sources"][0]
            src = f"({s0['product_category']}, cos={s0['score']}) {s0['excerpt'][:120]}"
            src = src.replace("\n", " ").replace("|", "\\|")
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {i} | {q} | {ans} | {src} | | |")
    MD_PATH.write_text("\n".join(lines))
    print(f"Saved eval table -> {MD_PATH}")


if __name__ == "__main__":
    main()
