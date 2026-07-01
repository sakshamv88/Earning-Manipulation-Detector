"""
generate_data.py

Generates a company-year financial statement panel.

IMPORTANT: This produces SYNTHETIC financial statements. The company names,
sectors, and manipulation-window years for the 12 fraud cases are REAL and
documented (see data/fraud_cases.csv). The underlying rupee figures (revenue,
receivables, CFO, etc.) are synthetically generated because live scraping of
Screener/NSE/BSE is not reachable from this sandbox.

The synthetic figures are NOT random noise -- during each company's real
manipulation-window years, the generator deliberately injects the financial
distortions that forensic-accounting literature (Beneish 1999, and the actual
fact patterns of these cases) associates with earnings manipulation:
  - receivables growing faster than sales (channel stuffing / revenue inflation)
  - declining gross margin masked by aggressive revenue booking
  - rising total-accruals-to-assets (profit diverging from operating cash flow)
  - rising related-party transaction share
  - occasional auditor change / filing delay in the final manipulation year

TO REPLACE WITH REAL DATA: swap `generate_company_panel()` with a function
that pulls actual P&L/BS/CF line items (e.g. from your existing Screener.in
scraper) into the same column schema produced here. Everything downstream
(features.py, train_model.py, score_live.py) reads that schema and does not
care whether the numbers came from this generator or real filings.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SCHEMA = [
    "company", "ticker", "sector", "fiscal_year", "is_fraud_case", "label",
    "revenue", "cogs", "sga", "depreciation", "net_income",
    "receivables", "current_assets", "ppe_net", "total_assets",
    "current_liabilities", "long_term_debt", "total_liabilities", "cfo",
    "related_party_pct_revenue", "auditor_changed", "promoter_pledge_pct",
    "results_filing_delay_days",
]

N_YEARS = 5          # 5-year panel per company
N_CLEAN_PER_FRAUD = 3  # matched clean companies per fraud case, same sector


def _base_trajectory(start_revenue, years, growth_mu=0.11, growth_sigma=0.04):
    """Normal, non-manipulated revenue growth path."""
    revs = [start_revenue]
    for _ in range(years - 1):
        g = RNG.normal(growth_mu, growth_sigma)
        revs.append(revs[-1] * (1 + g))
    return np.array(revs)


def _clean_company_year(revenue, prev_state):
    """Generate one year of 'honest' financials given revenue and prior year's state."""
    gross_margin = np.clip(RNG.normal(0.30, 0.03), 0.10, 0.55)
    cogs = revenue * (1 - gross_margin)
    sga = revenue * np.clip(RNG.normal(0.12, 0.02), 0.03, 0.25)
    depreciation = revenue * np.clip(RNG.normal(0.03, 0.005), 0.005, 0.08)
    ebit = revenue - cogs - sga - depreciation
    net_income = ebit * np.clip(RNG.normal(0.68, 0.05), 0.4, 0.85)  # after tax/interest

    # occasional noisy year even for clean companies (a normal business hiccup,
    # not manipulation) -- keeps the classification problem realistically messy
    noise_boost = 1.0
    if RNG.random() < 0.12:
        noise_boost = RNG.uniform(1.1, 1.3)

    receivables = revenue * np.clip(RNG.normal(0.16, 0.02) * noise_boost, 0.05, 0.4)
    current_assets = receivables + revenue * np.clip(RNG.normal(0.10, 0.02), 0.02, 0.25)
    ppe_net = prev_state["ppe_net"] * 1.03 + revenue * 0.05 if prev_state else revenue * 0.6
    total_assets = current_assets + ppe_net + revenue * np.clip(RNG.normal(0.15, 0.03), 0.0, 0.4)

    current_liabilities = revenue * np.clip(RNG.normal(0.12, 0.02), 0.03, 0.3)
    long_term_debt = total_assets * np.clip(RNG.normal(0.20, 0.05), 0.0, 0.5)
    total_liabilities = current_liabilities + long_term_debt

    cfo = net_income * np.clip(RNG.normal(0.95, 0.08), 0.6, 1.25)  # CFO tracks NI closely = honest

    related_party_pct_revenue = np.clip(RNG.normal(0.02, 0.01), 0.0, 0.08)
    auditor_changed = 0
    promoter_pledge_pct = np.clip(RNG.normal(0.05, 0.05), 0.0, 0.3)
    results_filing_delay_days = max(0, int(RNG.normal(2, 3)))

    return dict(
        revenue=revenue, cogs=cogs, sga=sga, depreciation=depreciation, net_income=net_income,
        receivables=receivables, current_assets=current_assets, ppe_net=ppe_net,
        total_assets=total_assets, current_liabilities=current_liabilities,
        long_term_debt=long_term_debt, total_liabilities=total_liabilities, cfo=cfo,
        related_party_pct_revenue=related_party_pct_revenue, auditor_changed=auditor_changed,
        promoter_pledge_pct=promoter_pledge_pct,
        results_filing_delay_days=results_filing_delay_days,
    )


def _distort_for_manipulation(state, severity, final_year):
    """
    Apply manipulation-window distortions on top of an otherwise-clean year.
    Includes per-case jitter and a chance each distortion is muted, so fraud
    years overlap with clean-company noise rather than being perfectly
    separable (real fraud detection is never a clean split).
    """
    s = dict(state)
    intensity = np.clip(RNG.normal(1.0, 0.35), 0.35, 1.8)
    eff = severity * intensity

    def maybe(p=0.75):
        return 1.0 if RNG.random() < p else RNG.uniform(0.0, 0.3)

    s["revenue"] *= (1 + 0.06 * eff * maybe() + RNG.normal(0, 0.015))
    s["receivables"] *= (1 + 0.35 * eff * maybe() + RNG.normal(0, 0.05))
    s["current_assets"] += s["receivables"] * 0.15 * eff * maybe()

    margin_erosion = np.clip(0.05 * eff * maybe() + RNG.normal(0, 0.01), -0.02, 0.15)
    true_gross_profit = (s["revenue"] - s["cogs"]) * (1 - margin_erosion)
    s["cogs"] = s["revenue"] - true_gross_profit

    s["sga"] *= (1 - 0.10 * eff * maybe() + RNG.normal(0, 0.02))
    s["depreciation"] *= (1 - 0.12 * eff * maybe() + RNG.normal(0, 0.02))

    ebit = s["revenue"] - s["cogs"] - s["sga"] - s["depreciation"]
    s["net_income"] = ebit * np.clip(0.68 + 0.05 * eff + RNG.normal(0, 0.04), 0.35, 0.92)

    s["total_assets"] *= (1 + 0.08 * eff * maybe() + RNG.normal(0, 0.02))

    cfo_ratio = np.clip(0.85 - 0.55 * eff * maybe(0.85) + RNG.normal(0, 0.08), 0.05, 0.95)
    s["cfo"] = s["net_income"] * cfo_ratio

    s["long_term_debt"] *= (1 + 0.10 * eff * maybe() + RNG.normal(0, 0.03))
    s["total_liabilities"] = s["current_liabilities"] + s["long_term_debt"]

    s["related_party_pct_revenue"] = max(0.0, min(
        0.30, s["related_party_pct_revenue"] + 0.08 * eff * maybe(0.55) + RNG.normal(0, 0.01)))

    if final_year:
        s["auditor_changed"] = 1 if RNG.random() < 0.6 else 0
        s["results_filing_delay_days"] += int(max(0, RNG.normal(12 * eff, 8)))
        s["promoter_pledge_pct"] = min(0.9, s["promoter_pledge_pct"]
                                        + 0.25 * eff * maybe(0.6) + RNG.normal(0, 0.03))

    return s


def generate_company_panel(company, ticker, sector, disclosure_year=None,
                            window_start=None, window_end=None, is_fraud_case=False):
    """
    Generate a 5-fiscal-year panel for one company.
    fiscal_year runs from (disclosure_year - N_YEARS) to (disclosure_year - 1)
    for fraud cases, anchored so the manipulation window lines up with real years.
    For clean (matched) companies, fiscal_year is just a relative 1..5 index
    aligned to the same calendar years as their paired fraud company.
    """
    anchor_year = disclosure_year if disclosure_year else 2024
    start_year = anchor_year - N_YEARS
    years = list(range(start_year, anchor_year))

    start_revenue = RNG.uniform(300, 4000) * 1e6  # INR, ~30cr to 400cr starting revenue
    revenue_path = _base_trajectory(start_revenue, N_YEARS)

    rows = []
    prev_state = None
    for i, fy in enumerate(years):
        base = _clean_company_year(revenue_path[i], prev_state)
        in_window = is_fraud_case and window_start and window_start <= fy <= window_end
        label = 0
        if in_window:
            # severity escalates across the window, peaks in the final window year
            span = max(1, window_end - window_start)
            severity = 0.4 + 0.6 * ((fy - window_start) / span)
            final_year = (fy == window_end)
            base = _distort_for_manipulation(base, severity, final_year)
            label = 1
        row = dict(company=company, ticker=ticker, sector=sector, fiscal_year=fy,
                   is_fraud_case=int(is_fraud_case), label=label, **base)
        rows.append(row)
        prev_state = base

    return pd.DataFrame(rows)[SCHEMA]


def build_full_dataset():
    fraud_meta = pd.read_csv("/home/claude/fraud_detector/data/fraud_cases.csv")
    panels = []

    for _, fc in fraud_meta.iterrows():
        # fraud company panel
        panels.append(generate_company_panel(
            company=fc["company"], ticker=fc["ticker"], sector=fc["sector"],
            disclosure_year=int(fc["disclosure_year"]),
            window_start=int(fc["manipulation_window_start"]),
            window_end=int(fc["manipulation_window_end"]),
            is_fraud_case=True,
        ))
        # matched clean companies, same sector, same calendar window
        for j in range(N_CLEAN_PER_FRAUD):
            clean_ticker = f"{fc['ticker']}_PEER{j+1}"
            panels.append(generate_company_panel(
                company=f"{fc['sector']} Peer {j+1} (matched to {fc['company']})",
                ticker=clean_ticker, sector=fc["sector"],
                disclosure_year=int(fc["disclosure_year"]),
                is_fraud_case=False,
            ))

    full = pd.concat(panels, ignore_index=True)
    return full


if __name__ == "__main__":
    df = build_full_dataset()
    out_path = "/home/claude/fraud_detector/data/financial_panel.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} company-year rows across {df['company'].nunique()} companies")
    print(f"Fraud-year rows (label=1): {df['label'].sum()}  |  Clean-year rows: {(df['label']==0).sum()}")
    print(f"Saved to {out_path}")
