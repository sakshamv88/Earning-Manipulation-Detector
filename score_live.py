"""
score_live.py
Scores a "current universe" of NSE-style companies with the trained LightGBM
model and produces a ranked risk table + SHAP-based explanation for the
top-flagged names.

NOTE: the "current universe" here is again SYNTHETIC (same reason as
generate_data.py -- no live NSE/Screener access from this sandbox). To use
real current filings: pull the same 17 raw line items (see SCHEMA in
generate_data.py) for real companies from your Screener scraper, run them
through features.compute_features() using two consecutive real fiscal years,
and feed the result into this script's scoring step instead of
build_live_universe().
"""

import numpy as np
import pandas as pd
import joblib
import shap
import sys

sys.path.insert(0, "/home/claude/fraud_detector/src")
from generate_data import generate_company_panel, RNG
from features import compute_features

SECTORS = ["IT Services", "Housing Finance NBFC", "FMCG Beverages", "Jewellery Retail",
           "Textiles/Plastics", "Financial Services", "Healthcare", "Retail/E-Governance",
           "Aviation", "Office Equipment/IT Services", "Infrastructure Finance", "Private Banking"]

OUT_DIR = "/home/claude/fraud_detector/outputs"
DATA_DIR = "/home/claude/fraud_detector/data"


def build_live_universe(n=60, anchor_year=2026):
    """
    Synthetic 'current' universe: mostly clean-pattern companies, with a
    handful of injected higher-risk companies (unlabeled -- the model has to
    find them). This mirrors how you'd actually screen: you don't know in
    advance which of your 60 names might be a problem.
    """
    panels = []
    for i in range(n):
        sector = SECTORS[i % len(SECTORS)]
        ticker = f"LIVE{i+1:03d}"
        company = f"{sector} Co {i+1}"

        # ~15% of the live universe gets an unlabeled, injected risk pattern
        inject_risk = RNG.random() < 0.15
        if inject_risk:
            panel = generate_company_panel(
                company=company, ticker=ticker, sector=sector,
                disclosure_year=anchor_year + 1,  # window ends right before "today"
                window_start=anchor_year - 1, window_end=anchor_year,
                is_fraud_case=True,
            )
        else:
            panel = generate_company_panel(
                company=company, ticker=ticker, sector=sector,
                disclosure_year=anchor_year + 1, is_fraud_case=False,
            )
        panels.append(panel)

    return pd.concat(panels, ignore_index=True)


def main():
    models = joblib.load(f"{OUT_DIR}/trained_models.joblib")
    gbm, feature_cols = models["gbm"], models["features"]

    universe_panel = build_live_universe()
    universe_panel.to_csv(f"{DATA_DIR}/live_universe_panel.csv", index=False)

    feats = compute_features(universe_panel)
    # keep only the most recent feature-year per company (latest available filing)
    latest = feats.sort_values("fiscal_year").groupby("company").tail(1).reset_index(drop=True)

    X = latest[feature_cols].values
    latest["risk_score"] = gbm.predict_proba(X)[:, 1]
    latest = latest.sort_values("risk_score", ascending=False).reset_index(drop=True)

    ranked_cols = ["company", "ticker", "sector", "fiscal_year", "risk_score",
                   "DSRI", "GMI", "TATA", "cfo_to_ni", "related_party_pct_revenue",
                   "auditor_changed", "promoter_pledge_pct"]
    ranked = latest[ranked_cols]
    ranked.to_csv(f"{OUT_DIR}/live_screening_ranked.csv", index=False)

    print(f"Screened {len(latest)} companies.\n")
    print("Top 10 by risk score:")
    print(ranked.head(10).to_string(index=False))

    # SHAP explanation for top 3
    explainer = shap.TreeExplainer(gbm)
    top3 = latest.head(3)
    print("\n\n=== Top 3 flagged: driving features ===")
    for _, row in top3.iterrows():
        x = row[feature_cols].values.astype(float).reshape(1, -1)
        sv = explainer.shap_values(pd.DataFrame(x, columns=feature_cols))
        sv = sv[1] if isinstance(sv, list) else sv
        contrib = pd.Series(sv[0], index=feature_cols).sort_values(key=abs, ascending=False)
        print(f"\n{row['company']} ({row['ticker']}) -- risk score {row['risk_score']:.3f}")
        print(contrib.head(5).round(3).to_string())

    print(f"\nSaved full ranked table to {OUT_DIR}/live_screening_ranked.csv")


if __name__ == "__main__":
    main()
