# RAG Qualitative Evaluation

**System:** retriever = `all-MiniLM-L6-v2` + FAISS (top-k=5); generator = `google/flan-t5-base`.
**Method:** 7 representative business questions were run through the full pipeline
(`src/evaluate_rag.py`). Each answer and its retrieved sources were inspected and
scored 1–5 on relevance, grounding, and usefulness. Raw outputs:
[`reports/rag_eval_raw.json`](rag_eval_raw.json).

| # | Question | Generated Answer | Top Retrieved Source (cosine) | Quality (1-5) | Comments |
|---|---|---|---|:--:|---|
| 1 | Why are people unhappy with credit cards? | "These credit card companies abuse their financial power and just cut credit limits without any justification… pure abuse of power and financial leverage." | Credit Card · CITIBANK · *Trouble using your card* (0.61) | **4** | Relevant and grounded in a retrieved chunk; captures a real theme (arbitrary credit-limit cuts) but reflects one complaint rather than synthesizing several. |
| 2 | Most common issues with money transfers? | "The most common issues customers report with money transfers are:" *(incomplete)* | Money Transfer · Block, Inc. · *Other transaction problem* (0.61) | **2** | Retrieval is on-target (Money Transfer chunks), but the small LLM produced a degenerate, unfinished sentence. Clearest example of the generator being the weak link. |
| 3 | Problems with savings accounts? | "Customer complaints about poor service." | Savings Account · WELLS FARGO · *Managing an account* (0.60) | **3** | Correct product retrieved and answer is on-topic, but too generic/terse to be actionable. |
| 4 | Main complaints about personal loans? | "…they are trapping consumers in an endless debt trap from which there is no way out… high-risk loans to consumers with poor credit is their excuse…" | Personal Loan · Upstart Holdings · *Payoff process at end of loan* (0.62) | **3** | On-topic and vivid, but essentially extractive — it quotes one chunk rather than summarizing the range of personal-loan issues. |
| 5 | Are customers charged unexpected fees on credit cards? | "Credit card company charged cardholder's account twice for latest fees while available credit for the monthly fees were on the cardholder's account balance." | Credit Card · Bread Financial · *Fees or interest* (**0.76**) | **4** | Strongest retrieval in the set (0.76); answer is specific, grounded, and directly answers the question (yes, with a concrete pattern). |
| 6 | Do customers report unauthorized/fraudulent transactions? | "Unauthorized transactions were detected on an account… discovered during a quarterly audit." | Credit Card · GOLDMAN SACHS · *Problem with a purchase* (0.69) | **3** | Confirms the issue with a real example; extractive and narrow, but relevant. |
| 7 | Issues when closing an account or getting a refund? | "Their customer care department is playing with loyal customers if they decide to close their account." | Credit Card · HSBC · *Fees or interest* (0.68) | **3** | Relevant complaint about account-closure friction; again extractive rather than synthesized. |

## Summary

**Average quality: ~3.1 / 5.**

**What worked well**
- **Retrieval is strong and trustworthy.** For every question the top chunks came
  from the correct product family with sensible cosine scores (0.60–0.76), and the
  source panel makes each answer auditable. This is the core value of the system —
  a PM can immediately see the evidence.
- **Fee/fraud questions** (Q5, Q1) produced genuinely useful, specific answers.

**What could be improved**
- **Generation is the bottleneck, not retrieval.** `flan-t5-base` (250M params) is
  frequently *extractive* (quoting one chunk) or *terse*, and occasionally
  *degenerate* (Q2's unfinished sentence). It does not synthesize across the 5
  retrieved chunks the way the prompt asks.
- **Fixes / next steps:** (1) swap in a larger instruction model — `flan-t5-large`,
  `Mistral-7B-Instruct`, or an API model such as Claude — which would summarize
  across sources far better; (2) add light answer post-processing / length control;
  (3) try k=8 with a re-ranker to feed the generator richer context.

**Takeaway:** the retrieval + grounding foundation is solid and production-relevant;
the open-source CPU generator is the limiting factor and is the clearest lever for
improving answer quality.
