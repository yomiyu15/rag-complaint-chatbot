"""
Task 2 - Chunking, embedding and vector-store indexing.

Pipeline:
  1. Load the cleaned corpus from data/filtered_complaints.csv (Task 1 output).
  2. Draw a STRATIFIED sample (~10k-15k complaints) proportional to product
     category, so the minority Personal Loan class stays represented.
  3. Split each narrative into overlapping chunks with LangChain's
     RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50).
  4. Embed every chunk with sentence-transformers/all-MiniLM-L6-v2 (384-dim).
  5. Index the chunk vectors into a persisted FAISS index under vector_store/,
     with a parquet sidecar holding the chunk text + per-chunk metadata (row i
     of the sidecar == vector i in the index) so retrieved chunks trace back to
     their source complaint.
  6. Write reports/task2_summary.json and run a quick retrieval sanity check.

FAISS stores only vectors + positions, so chunk text and metadata live in
vector_store/chunks.parquet alongside vector_store/index.faiss.

Usage:
    python src/chunk_embed_index.py --sample-size 12000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
FILTERED_CSV = ROOT / "data" / "filtered_complaints.csv"
VECTOR_STORE = ROOT / "vector_store"
INDEX_PATH = VECTOR_STORE / "index.faiss"
CHUNKS_PATH = VECTOR_STORE / "chunks.parquet"
SUMMARY_PATH = ROOT / "reports" / "task2_summary.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

META_COLS = {
    "Complaint ID": "complaint_id",
    "product_category": "product_category",
    "Product": "product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "State": "state",
    "Date received": "date_received",
}


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows, allocated across product_category in proportion to size."""
    frac = n / len(df)
    parts = []
    for cat, grp in df.groupby("product_category"):
        take = max(1, round(len(grp) * frac))
        take = min(take, len(grp))
        parts.append(grp.sample(n=take, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main(sample_size: int, seed: int, chunk_size: int, chunk_overlap: int,
         batch_size: int) -> None:
    if not FILTERED_CSV.exists():
        raise SystemExit(
            f"{FILTERED_CSV} not found. Run src/eda_preprocessing.py first (Task 1)."
        )

    print(f"Loading {FILTERED_CSV.name} ...")
    df = pd.read_csv(FILTERED_CSV, dtype=str)
    df = df[df["cleaned_narrative"].notna() & (df["cleaned_narrative"].str.strip() != "")]
    print(f"  corpus rows with narrative: {len(df):,}")

    # ---- 1. stratified sample ------------------------------------------------ #
    full_dist = df["product_category"].value_counts()
    sample = stratified_sample(df, sample_size, seed)
    sample_dist = sample["product_category"].value_counts()
    print(f"\nStratified sample: {len(sample):,} complaints (seed={seed})")
    for cat in full_dist.index:
        print(f"  {cat:<16} full {full_dist[cat]:>8,} "
              f"({100*full_dist[cat]/len(df):5.1f}%)  ->  "
              f"sample {int(sample_dist.get(cat, 0)):>6,} "
              f"({100*sample_dist.get(cat,0)/len(sample):5.1f}%)")

    # ---- 2. chunk ------------------------------------------------------------ #
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for _, row in sample.iterrows():
        chunks = splitter.split_text(row["cleaned_narrative"])
        cid = str(row["Complaint ID"])
        total = len(chunks)
        for ci, chunk in enumerate(chunks):
            meta = {dst: ("" if pd.isna(row[src]) else str(row[src]))
                    for src, dst in META_COLS.items()}
            meta["chunk_index"] = ci
            meta["total_chunks"] = total
            ids.append(f"{cid}-{ci}")
            documents.append(chunk)
            metadatas.append(meta)

    n_chunks = len(documents)
    chunk_lens = np.array([len(d) for d in documents])
    print(f"\nChunking: {n_chunks:,} chunks "
          f"({n_chunks/len(sample):.2f} per complaint), "
          f"mean {chunk_lens.mean():.0f} chars, max {chunk_lens.max()}")

    # ---- 3. embed ------------------------------------------------------------ #
    print(f"\nEmbedding with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        documents, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    dim = int(embeddings.shape[1])
    print(f"  embeddings: {embeddings.shape}")

    # ---- 4. index into FAISS ------------------------------------------------ #
    VECTOR_STORE.mkdir(parents=True, exist_ok=True)
    print(f"\nIndexing {n_chunks:,} chunks into FAISS at {VECTOR_STORE} ...")
    # embeddings are L2-normalised, so inner product == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))

    # sidecar: row i == vector i in the index (chunk text + metadata)
    sidecar = pd.DataFrame(metadatas)
    sidecar.insert(0, "id", ids)
    sidecar["document"] = documents
    sidecar.to_parquet(CHUNKS_PATH, index=False)
    print(f"  wrote {INDEX_PATH.name} ({index.ntotal:,} vectors) "
          f"+ {CHUNKS_PATH.name} ({len(sidecar):,} rows)")

    # ---- 5. summary + sanity check ------------------------------------------ #
    summary = {
        "source_corpus_rows": int(len(df)),
        "sample_size_complaints": int(len(sample)),
        "seed": seed,
        "full_category_distribution": {k: int(v) for k, v in full_dist.items()},
        "sample_category_distribution": {k: int(v) for k, v in sample_dist.items()},
        "chunking": {
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": n_chunks,
            "chunks_per_complaint": round(n_chunks / len(sample), 2),
            "chunk_char_len_mean": round(float(chunk_lens.mean()), 1),
            "chunk_char_len_max": int(chunk_lens.max()),
        },
        "embedding": {
            "model": EMBED_MODEL,
            "dim": dim,
            "normalized": True,
        },
        "vector_store": {
            "backend": "FAISS",
            "index_type": "IndexFlatIP",
            "path": "vector_store/",
            "index_file": INDEX_PATH.name,
            "sidecar_file": CHUNKS_PATH.name,
            "distance": "cosine (inner product on normalised vectors)",
            "count": int(index.ntotal),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {SUMMARY_PATH}")

    print("\nRetrieval sanity check: 'why was i charged a late fee on my credit card'")
    q = model.encode(["why was i charged a late fee on my credit card"],
                     normalize_embeddings=True).astype(np.float32)
    scores, idx = index.search(q, 3)
    for rank, (pos, score) in enumerate(zip(idx[0], scores[0]), 1):
        row = sidecar.iloc[pos]
        print(f"  [{rank}] ({row['product_category']}, cos={score:.3f}) "
              f"{row['document'][:110]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    main(args.sample_size, args.seed, args.chunk_size, args.chunk_overlap,
         args.batch_size)
