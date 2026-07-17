# Notebooks

| Notebook | Task | Purpose |
|---|---|---|
| `01_eda_preprocessing.ipynb` | Task 1 | EDA of the CFPB complaints and preparation of the cleaned, filtered corpus. |

The heavy full-dataset pass runs via [`../src/eda_preprocessing.py`](../src/eda_preprocessing.py);
the notebook loads the produced artifacts (`reports/eda_summary.json`, figures,
`data/filtered_complaints.csv`) and demonstrates the EDA/cleaning logic on a sample.
