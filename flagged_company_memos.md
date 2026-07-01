# Earnings-Quality Screening — Flagged Names (Illustrative)

**IMPORTANT:** The companies below are from a **synthetic demonstration universe**,
not real listed companies. They exist to show the exact output format and reasoning
style this pipeline produces once real filings are plugged in. When you run this
against real NSE data, replace the company identity but keep this memo structure —
it mirrors how a due-diligence/QoE note is actually written.

All language below is deliberately hedged: a high risk score is a **screening
signal warranting further review**, never an accusation of fraud.

---

## Memo 1 — IT Services Co 37 (LIVE037)
**Risk score: 1.00 (highest in 60-company universe)**

**Snapshot:** FY2026 IT services company, most recent available filing.

**Primary red flags (SHAP-ranked):**
1. **Total accruals-to-assets (TATA) sharply elevated** — the single largest driver of the score. Reported net income is diverging materially from operating cash flow, meaning profit growth is not being backed by cash generation.
2. **CFO/Net Income ratio depressed** (~0.59) — consistent with #1; roughly 40% of reported profit is not converting to operating cash.
3. **Auditor change flagged** in the most recent fiscal year.
4. **SG&A-to-sales ratio moving atypically** relative to its own trend (SGAI elevated).

**What this would prompt in real diligence:**
- Request a cash-flow-statement walk-through reconciling net income to CFO, line by line.
- Ask management directly about the auditor change — voluntary rotation vs. resignation, and review any communication from the outgoing auditor.
- Pull the receivables ageing schedule and compare against sector peers.
- Cross-check related-party disclosures in the Notes to Accounts against the % of revenue flagged here.

**Caveat:** elevated accruals can also reflect legitimate one-off items (M&A accounting, revenue recognition timing under Ind AS 115, provisioning changes). This is a prioritization signal for where to dig first, not a conclusion.

---

## Memo 2 — Housing Finance NBFC Co 50 (LIVE050)
**Risk score: 1.00**

**Snapshot:** FY2026 housing-finance NBFC.

**Primary red flags:**
1. **TATA elevated** — same core signal as Memo 1: profit not converting to cash.
2. **Related-party transaction share elevated and the largest single contributor after TATA** — for an NBFC this is a meaningfully sharper flag than for most sectors, since related-party lending is a documented pattern in prior Indian NBFC stress cases.
3. **CFO/NI ~0.37** — among the weakest cash conversion ratios in the flagged set.
4. **Promoter pledge percentage elevated** relative to sector peers.

**What this would prompt in real diligence:**
- For an NBFC specifically: request the loan book breakdown by related-party vs. arm's-length borrowers, and compare disbursement growth to the broader housing-finance sector growth rate for the same period.
- Reconcile reported provisioning/NPA figures against RBI divergence-report methodology if available.
- Check promoter pledge trend over the last 8 quarters, not just the latest snapshot.

**Caveat:** NBFC accrual patterns are structurally different from operating companies (interest income accrual vs. cash collection timing); TATA should be read alongside NBFC-specific asset-quality metrics before drawing conclusions.

---

## Memo 3 — Financial Services Co 6 (LIVE006)
**Risk score: 1.00**

**Snapshot:** FY2026 financial services company.

**Primary red flags:**
1. **TATA elevated** (consistent driver across all three flagged names).
2. **Related-party transaction share is the second-largest contributor.**
3. **Results filing delay** flagged relative to statutory deadline — a governance-timing signal that is easy to verify independently via exchange filing records.
4. Unlike Memo 1, SG&A ratio here moved in the *opposite* direction (SGAI negative contribution) — a reminder that the same headline risk score can be driven by different underlying mechanics.

**What this would prompt in real diligence:**
- Confirm the exact filing delay against BSE/NSE corporate announcement records (public, easy to verify — do this first).
- Request explanation for the related-party transaction growth specifically, with counterparty names.

**Caveat:** filing delays are sometimes purely administrative (system issues, auditor scheduling) rather than substantive. Treat this as a "verify first" item, not a standalone conclusion.

---

## Cross-cutting observation

All three flagged names in this run share **TATA (accrual/cash-flow divergence)** as
the dominant SHAP driver — consistent with both the Beneish-methodology literature
and the leave-one-fraud-out backtest on the 12 real historical cases, where the same
feature dominated. This is a reasonable place to lead with when explaining the model
to a non-technical audience: *"the single strongest predictor across nearly every
historical fraud case and every flagged current name is that reported profit isn't
converting to cash the way it should."*
