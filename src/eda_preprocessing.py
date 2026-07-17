"""
Task 1 - Exploratory Data Analysis & Preprocessing for the CFPB complaint dataset.

The raw CFPB export is ~6 GB uncompressed (9.6M rows), so this script streams the
CSV in chunks and never holds the whole file in memory. In a single pass it:

  1. Builds the product distribution over the FULL dataset.
  2. Counts complaints with / without a consumer narrative.
  3. Collects narrative word-count statistics for the four target product families.
  4. Cleans the narratives and writes the filtered dataset to
     data/filtered_complaints.csv (append mode, chunk by chunk).

It then renders EDA figures to reports/figures/ and writes a machine-readable
summary to reports/eda_summary.json.

Usage:
    python src/eda_preprocessing.py --zip "C:/Users/coop/Downloads/complaints.csv.zip"
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Map the messy CFPB `Product` labels (current + legacy) onto the four target
# families. Edit this dict to change what gets folded into each category.
PRODUCT_MAP: dict[str, str] = {
    # Credit Card
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    # Personal Loan (CFPB bundles payday/title/personal into single categories)
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",
    "Payday loan": "Personal Loan",
    # Savings Account
    "Checking or savings account": "Savings Account",
    "Bank account or service": "Savings Account",
    # Money Transfer
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
    "Virtual currency": "Money Transfer",
}

# Columns we actually need (keeps per-chunk memory small).
USECOLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Complaint ID",
]

NARRATIVE_COL = "Consumer complaint narrative"
CHUNKSIZE = 500_000

# Boilerplate openers frequently found in CFPB narratives. Removed to reduce
# noise before embedding. Matched case-insensitively at/near the start of text.
BOILERPLATE_PATTERNS = [
    r"i am writing to (?:file|lodge|submit|make) a complaint(?: (?:about|regarding|against))?",
    r"i am writing to complain(?: about| regarding)?",
    r"i would like to (?:file|submit|lodge) a complaint(?: about| regarding)?",
    r"i wish to (?:file|lodge) a complaint",
    r"to whom it may concern[,:]?",
    r"dear (?:sir or madam|sir\/madam|sir|madam|cfpb)[,:]?",
    r"this (?:is a|is my) complaint (?:about|regarding|against)",
    r"i am filing (?:this|a) complaint(?: about| regarding| against)?",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)

# CFPB redacts PII with runs of X's (e.g. "XXXX", "XX/XX/XXXX"). Collapse them.
REDACTION_RE = re.compile(r"\b(?:x{2,}[\s/\-]*)+x*\b", flags=re.IGNORECASE)
# Keep letters, numbers and a little sentence punctuation; drop the rest.
NON_TEXT_RE = re.compile(r"[^a-z0-9\s.,!?$%'-]")
# Collapse repeated punctuation runs, e.g. "$$$" -> "$", "!!!" -> "!".
REPEAT_PUNCT_RE = re.compile(r"([.,!?$%'-])\1+")
MULTISPACE_RE = re.compile(r"\s+")


def clean_narrative(text: str) -> str:
    """Lowercase, strip boilerplate/redactions/special chars, normalise spaces."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = REDACTION_RE.sub(" ", t)          # drop XXXX redaction tokens
    t = BOILERPLATE_RE.sub(" ", t)        # drop boilerplate openers
    t = NON_TEXT_RE.sub(" ", t)           # drop leftover special characters
    t = REPEAT_PUNCT_RE.sub(r"\1", t)     # collapse "$$$" -> "$", "!!!" -> "!"
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def main(zip_path: Path, out_dir: Path) -> None:
    data_dir = out_dir / "data"
    fig_dir = out_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (data_dir).mkdir(parents=True, exist_ok=True)
    filtered_path = data_dir / "filtered_complaints.csv"
    if filtered_path.exists():
        filtered_path.unlink()  # fresh start (we append below)

    # --- accumulators ------------------------------------------------------ #
    product_counts: Counter[str] = Counter()
    product_with_narr: Counter[str] = Counter()
    total_rows = 0
    total_with_narr = 0
    # word counts for the RAW narratives of the four target families
    target_wordcounts: list[int] = []
    category_counts: Counter[str] = Counter()
    n_written = 0
    header_written = False

    zf = zipfile.ZipFile(zip_path)
    csv_name = zf.namelist()[0]
    print(f"Streaming {csv_name} from {zip_path.name} ...")

    with zf.open(csv_name) as fh:
        reader = pd.read_csv(fh, usecols=USECOLS, dtype=str, chunksize=CHUNKSIZE)
        for i, chunk in enumerate(reader, 1):
            total_rows += len(chunk)

            # narrative presence (non-null, non-blank)
            narr = chunk[NARRATIVE_COL].fillna("").str.strip()
            has_narr = narr != ""
            total_with_narr += int(has_narr.sum())

            # full-dataset product distribution
            product_counts.update(chunk["Product"].dropna().tolist())
            product_with_narr.update(chunk.loc[has_narr, "Product"].dropna().tolist())

            # ---- build the filtered / cleaned subset ---------------------- #
            chunk = chunk[has_narr].copy()
            chunk["product_category"] = chunk["Product"].map(PRODUCT_MAP)
            chunk = chunk[chunk["product_category"].notna()]
            if not chunk.empty:
                # raw word counts (for length EDA, before cleaning)
                wc = chunk[NARRATIVE_COL].str.split().str.len()
                target_wordcounts.extend(wc.tolist())
                category_counts.update(chunk["product_category"].tolist())

                # clean text
                chunk["cleaned_narrative"] = chunk[NARRATIVE_COL].map(clean_narrative)
                # drop rows that became empty after cleaning
                chunk = chunk[chunk["cleaned_narrative"].str.strip() != ""]

                out_cols = [
                    "Complaint ID", "product_category", "Product", "Sub-product",
                    "Issue", "Sub-issue", "Company", "State", "Date received",
                    "cleaned_narrative",
                ]
                chunk[out_cols].to_csv(
                    filtered_path, mode="a", index=False,
                    header=not header_written,
                )
                header_written = True
                n_written += len(chunk)

            print(f"  chunk {i:>3}: seen {total_rows:>10,} rows | "
                  f"filtered kept {n_written:>8,}", flush=True)

    zf.close()

    # ----------------------------------------------------------------------- #
    # Statistics
    # ----------------------------------------------------------------------- #
    wc_arr = np.array(target_wordcounts, dtype=np.int64)
    pct = np.percentile(wc_arr, [1, 25, 50, 75, 95, 99]).round(1).tolist()
    very_short = int((wc_arr < 5).sum())
    very_long = int((wc_arr > 300).sum())

    summary = {
        "total_rows": total_rows,
        "rows_with_narrative": total_with_narr,
        "rows_without_narrative": total_rows - total_with_narr,
        "pct_with_narrative": round(100 * total_with_narr / total_rows, 2),
        "filtered_rows_written": n_written,
        "category_counts": dict(category_counts.most_common()),
        "narrative_wordcount": {
            "count": int(wc_arr.size),
            "min": int(wc_arr.min()),
            "max": int(wc_arr.max()),
            "mean": round(float(wc_arr.mean()), 1),
            "median": float(np.median(wc_arr)),
            "p1_p25_p50_p75_p95_p99": pct,
            "very_short_lt5_words": very_short,
            "very_long_gt300_words": very_long,
        },
        "top_products_full_dataset": dict(product_counts.most_common(15)),
    }
    summary_path = out_dir / "reports" / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\nSaved summary ->", summary_path)
    print(json.dumps(summary, indent=2))

    # ----------------------------------------------------------------------- #
    # Figures
    # ----------------------------------------------------------------------- #
    # 1) Product distribution over the full dataset (top 12)
    top = product_counts.most_common(12)
    labels = [k[:38] for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals, color="#4C72B0")
    plt.title("Top products across the full CFPB dataset")
    plt.xlabel("Number of complaints")
    plt.tight_layout()
    plt.savefig(fig_dir / "product_distribution_full.png", dpi=120)
    plt.close()

    # 2) Complaint counts for the four target categories
    cats = list(category_counts.keys())
    cvals = [category_counts[c] for c in cats]
    plt.figure(figsize=(8, 5))
    plt.bar(cats, cvals, color="#55A868")
    plt.title("Filtered complaints by target category (with narrative)")
    plt.ylabel("Number of complaints")
    plt.xticks(rotation=15)
    for x, v in enumerate(cvals):
        plt.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "target_category_counts.png", dpi=120)
    plt.close()

    # 3) With vs without narrative
    plt.figure(figsize=(5, 5))
    plt.pie(
        [total_with_narr, total_rows - total_with_narr],
        labels=["Has narrative", "No narrative"],
        autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"], startangle=90,
    )
    plt.title("Narrative availability (full dataset)")
    plt.tight_layout()
    plt.savefig(fig_dir / "narrative_availability.png", dpi=120)
    plt.close()

    # 4) Narrative word-count distribution (clipped at p99 for readability)
    clip = int(pct[-1])  # p99
    plt.figure(figsize=(9, 5))
    plt.hist(wc_arr[wc_arr <= clip], bins=60, color="#8172B3", edgecolor="white")
    plt.axvline(np.median(wc_arr), color="black", linestyle="--",
                label=f"median = {np.median(wc_arr):.0f} words")
    plt.title("Consumer narrative length (target categories, clipped at p99)")
    plt.xlabel("Word count")
    plt.ylabel("Number of narratives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "narrative_wordcount_hist.png", dpi=120)
    plt.close()

    print("Saved 4 figures ->", fig_dir)
    print(f"Saved filtered dataset -> {filtered_path} ({n_written:,} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to complaints.csv.zip")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]),
                    help="Project root for outputs (default: repo root)")
    args = ap.parse_args()
    main(Path(args.zip), Path(args.out))
