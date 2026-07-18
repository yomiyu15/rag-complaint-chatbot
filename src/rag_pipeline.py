"""
Task 3 - RAG core logic: retriever + prompt + LLM generator.

Loads the FAISS vector store built in Task 2, retrieves the top-k most relevant
complaint chunks for a question, injects them into a grounded prompt, and asks a
local LLM (default: google/flan-t5-base) to answer using only that context.

The retrieved sources are returned alongside the answer so the UI (Task 4) can
show evidence and let users verify.

Usage (quick manual test):
    python src/rag_pipeline.py "Why are people unhappy with credit cards?"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
VECTOR_STORE = ROOT / "vector_store"
INDEX_PATH = VECTOR_STORE / "index.faiss"
CHUNKS_PATH = VECTOR_STORE / "chunks.parquet"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Small, CPU-friendly instruction model that follows "answer from context" well.
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "google/flan-t5-base")

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Using only \
the customer complaint excerpts below, write a concise, well-formed summary that \
answers the question. Identify the main recurring problems. If the excerpts do not \
contain the answer, say you don't have enough information.

Customer complaint excerpts:
{context}

Question: {question}

Detailed answer:"""

# flan-t5 accepts ~512 input tokens; keep the joined context within a char budget
# (~4 chars/token) so the question + template still fit.
CONTEXT_CHAR_BUDGET = 1800


@dataclass
class Source:
    text: str
    score: float
    complaint_id: str
    product_category: str
    issue: str
    company: str


class RAGPipeline:
    def __init__(self, k: int = 5, llm_model: str = LLM_MODEL):
        if not INDEX_PATH.exists():
            raise SystemExit(
                f"{INDEX_PATH} not found. Run src/chunk_embed_index.py first (Task 2)."
            )
        self.k = k
        print(f"Loading FAISS index + chunks from {VECTOR_STORE} ...")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = pd.read_parquet(CHUNKS_PATH)
        print(f"Loading embedding model {EMBED_MODEL} ...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print(f"Loading LLM {llm_model} ...")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        self.model.eval()

    # ---- retrieval --------------------------------------------------------- #
    def retrieve(self, question: str, k: int | None = None) -> list[Source]:
        k = k or self.k
        q = self.embedder.encode([question], normalize_embeddings=True).astype(np.float32)
        scores, idx = self.index.search(q, k)
        sources: list[Source] = []
        for pos, score in zip(idx[0], scores[0]):
            if pos < 0:
                continue
            row = self.chunks.iloc[int(pos)]
            sources.append(Source(
                text=str(row["document"]),
                score=float(score),
                complaint_id=str(row.get("complaint_id", "")),
                product_category=str(row.get("product_category", "")),
                issue=str(row.get("issue", "")),
                company=str(row.get("company", "")),
            ))
        return sources

    # ---- prompt ------------------------------------------------------------ #
    @staticmethod
    def build_context(sources: list[Source]) -> str:
        parts, used = [], 0
        for s in sources:
            snippet = s.text.strip()
            block = f"- {snippet}"
            if used + len(block) > CONTEXT_CHAR_BUDGET and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    # ---- generation -------------------------------------------------------- #
    def answer(self, question: str, k: int | None = None) -> dict:
        sources = self.retrieve(question, k)
        context = self.build_context(sources)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        with self._torch.no_grad():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=256, num_beams=4, early_stopping=True,
            )
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        if not text:
            text = "I don't have enough information to answer that."
        return {"question": question, "answer": text, "sources": sources}


def _cli() -> None:
    question = " ".join(sys.argv[1:]) or "Why are people unhappy with credit cards?"
    rag = RAGPipeline()
    res = rag.answer(question)
    print("\n" + "=" * 80)
    print("Q:", res["question"])
    print("\nA:", res["answer"])
    print("\nSources:")
    for i, s in enumerate(res["sources"], 1):
        print(f"  [{i}] cos={s.score:.3f} | {s.product_category} | {s.company} | "
              f"{s.issue}\n      {s.text[:140]}...")


if __name__ == "__main__":
    _cli()
