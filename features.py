"""
features.py
Computes Beneish M-Score components + cash-flow-divergence + governance
features for each company-year, using that year vs. the prior year.

Reads data/financial_panel.csv, writes data/features.csv
(one row per company-fiscal_year, from the 2nd year of each panel onward,
since every ratio needs a prior-year baseline).
"""

import numpy as np
import pandas as pd


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["company", "fiscal_year"]).reset_index(drop=True)
    rows = []

    for company, g in panel.groupby("company"):
        g = g.sort_values("fiscal_year").reset_index(drop=True)
        for i in range(1, len(g)):
            t, p = g.iloc[i], g.iloc[i - 1]  # current year t, prior year p

            gm_t = (t.revenue - t.cogs) / t.revenue
            gm_p = (p.revenue - p.cogs) / p.revenue

            dsri = (t.receivables / t.revenue) / (p.receivables / p.revenue)
            gmi = gm_p / gm_t
            aqi = (1 - (t.current_assets + t.ppe_net) / t.total_assets) / \
                  (1 - (p.current_assets + p.ppe_net) / p.total_assets)
            sgi = t.revenue / p.revenue
            depi = (p.depreciation / (p.depreciation + p.ppe_net)) / \
                   (t.depreciation / (t.depreciation + t.ppe_net))
            sgai = (t.sga / t.revenue) / (p.sga / p.revenue)
            tata = (t.net_income - t.cfo) / t.total_assets
            lvgi = ((t.current_liabilities + t.long_term_debt) / t.total_assets) / \
                   ((p.current_liabilities + p.long_term_debt) / p.total_assets)

            m_score = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                       + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)

            cfo_to_ni = t.cfo / t.net_income if t.net_income != 0 else np.nan
            cfo_divergence = (t.net_income - t.cfo) / t.total_assets  # same as TATA, kept explicit

            related_party_yoy = t.related_party_pct_revenue - p.related_party_pct_revenue

            rows.append(dict(
                company=company, ticker=t.ticker, sector=t.sector, fiscal_year=int(t.fiscal_year),
                is_fraud_case=int(t.is_fraud_case), label=int(t.label),
                DSRI=dsri, GMI=gmi, AQI=aqi, SGI=sgi, DEPI=depi, SGAI=sgai, TATA=tata, LVGI=lvgi,
                beneish_m_score=m_score,
                cfo_to_ni=cfo_to_ni, cfo_divergence=cfo_divergence,
                related_party_pct_revenue=t.related_party_pct_revenue,
                related_party_yoy_change=related_party_yoy,
                auditor_changed=int(t.auditor_changed),
                promoter_pledge_pct=t.promoter_pledge_pct,
                results_filing_delay_days=t.results_filing_delay_days,
            ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = pd.read_csv("/home/claude/fraud_detector/data/financial_panel.csv")
    feats = compute_features(panel)
    out_path = "/home/claude/fraud_detector/data/features.csv"
    feats.to_csv(out_path, index=False)
    print(f"Computed features for {len(feats)} company-year rows")
    print(f"Positive (label=1) rows: {feats['label'].sum()}  |  Negative: {(feats['label']==0).sum()}")
    print(f"Saved to {out_path}")
    print("\nBeneish M-Score sanity check (mean by label):")
    print(feats.groupby("label")["beneish_m_score"].mean())
