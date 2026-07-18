# Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) chatbot that turns CrediTrust's raw CFPB
consumer-complaint narratives into evidence-backed answers for Product, Support,
and Compliance teams across four product families: **Credit Card, Personal Loan,
Savings Account, Money Transfer**.

## Project structure
```
rag-complaint-chatbot/
├── data/
│   ├── raw/                     # source CFPB export (not versioned)
│   └── filtered_complaints.csv  # Task 1 output: cleaned + filtered corpus
├── vector_store/                # persisted FAISS index + parquet sidecar (Task 2)
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── eda_preprocessing.py     # Task 1 streaming EDA + preprocessing
│   ├── chunk_embed_index.py     # Task 2 chunking + embedding + FAISS index
│   ├── rag_pipeline.py          # Task 3 retriever + prompt + LLM generator
│   └── evaluate_rag.py          # Task 3 qualitative evaluation
├── reports/
│   ├── eda_summary.json         # machine-readable EDA stats
│   ├── task2_summary.json       # sampling/chunking/embedding metrics
│   ├── rag_evaluation.md        # scored evaluation table
│   ├── interim_report.(md|pdf)  # Task 1+2 interim report
│   ├── final_report.(md|pdf)    # full Medium-style report
│   └── figures/                 # EDA charts + UI screenshots
├── tests/
├── app.py                       # Gradio chat UI (Task 4)
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Task 1 — EDA & Preprocessing
The raw CFPB export is ~6 GB uncompressed (9.6M rows), so preprocessing streams the
CSV in chunks (no full in-memory load).

```bash
python src/eda_preprocessing.py --zip "C:/Users/coop/Downloads/complaints.csv.zip"
```
Outputs:
- `data/filtered_complaints.csv` — narratives for the four target families, cleaned
- `reports/eda_summary.json` — distributions, narrative-length stats
- `reports/figures/*.png` — product distribution, narrative availability, length histogram

See [`notebooks/01_eda_preprocessing.ipynb`](notebooks/01_eda_preprocessing.ipynb)
for the walkthrough and findings.

### Product mapping
CFPB uses many overlapping product labels; they are folded into four families in
`PRODUCT_MAP` (edit there to adjust):

| Target category | CFPB `Product` labels |
|---|---|
| Credit Card | Credit card; Credit card or prepaid card |
| Personal Loan | Payday loan, title loan, or personal loan; …personal loan, or advance loan; Consumer Loan; Payday loan |
| Savings Account | Checking or savings account; Bank account or service |
| Money Transfer | Money transfer, virtual currency, or money service; Money transfers; Virtual currency |

## Task 2 — Chunking, Embedding & Vector Store
Stratified 12k-complaint sample → `RecursiveCharacterTextSplitter` (500/50) →
`all-MiniLM-L6-v2` (384-dim) → FAISS `IndexFlatIP` + parquet metadata sidecar.
```bash
python src/chunk_embed_index.py --sample-size 12000 --seed 42
```
Outputs `vector_store/index.faiss`, `vector_store/chunks.parquet`, `reports/task2_summary.json`.

## Task 3 — RAG Pipeline & Evaluation
Retriever (`all-MiniLM-L6-v2` + FAISS top-k=5) → grounded prompt → `flan-t5-base`
generator; answers are returned with their source chunks.
```bash
python src/rag_pipeline.py "Why are people unhappy with credit cards?"   # single query
python src/evaluate_rag.py                                                # 7-question eval
```
Scored results: [`reports/rag_evaluation.md`](reports/rag_evaluation.md).

## Task 4 — Interactive UI
Gradio app: question box, Ask/Clear buttons, answer area, and **source excerpts** below
each answer for verification.
```bash
python app.py     # open http://127.0.0.1:7860
```

## Full report
See [`reports/final_report.md`](reports/final_report.md) (PDF alongside) for the
end-to-end write-up: technical choices, evaluation, and UI showcase.

## Environment note
This project runs in a **Python 3.12 virtualenv** (`.venv/`). Python 3.14 + a missing
Visual C++ runtime broke `torch`/`onnxruntime`; installing the MS VC++ Redistributable
and using 3.12 resolves it. The vector store uses **FAISS** (ChromaDB is a permitted
alternative) for install reliability.

## Roadmap
- [x] **Task 1** — EDA & preprocessing
- [x] **Task 2** — chunking, embedding (`all-MiniLM-L6-v2`), FAISS index
- [x] **Task 3** — RAG retriever + prompt + LLM generator, qualitative evaluation
- [x] **Task 4** — Gradio chat UI with source display
