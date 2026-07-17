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
├── vector_store/                # persisted FAISS / ChromaDB index (Task 2)
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   └── eda_preprocessing.py     # Task 1 streaming EDA + preprocessing
├── reports/
│   ├── eda_summary.json         # machine-readable EDA stats
│   └── figures/                 # EDA charts
├── tests/
├── app.py                       # Gradio/Streamlit UI (Task 4)
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

## Roadmap
- [x] **Task 1** — EDA & preprocessing
- [ ] **Task 2** — chunking, embedding (`all-MiniLM-L6-v2`), FAISS/ChromaDB index
- [ ] **Task 3** — RAG retriever + prompt + LLM generator, qualitative evaluation
- [ ] **Task 4** — Gradio/Streamlit chat UI with source display
