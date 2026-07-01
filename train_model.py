"""
train_model.py
- Splits features.csv into train/test (company-level split, not row-level,
  to avoid leaking the same company across train and test)
- Trains: (1) rule-based Beneish M-Score benchmark, (2) Logistic Regression,
  (3) LightGBM
- Runs a leave-one-fraud-company-out backtest: for each of the 12 real fraud
  cases, train on all other companies and test whether the model flags that
  held-out company's manipulation-window years using only its own features.
- Produces a SHAP summary plot for the LightGBM model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, roc_auc_score,
                              classification_report, confusion_matrix)
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

FEATURE_COLS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI",
                 "cfo_to_ni", "cfo_divergence", "related_party_pct_revenue",
                 "related_party_yoy_change", "auditor_changed",
                 "promoter_pledge_pct", "results_filing_delay_days"]

DATA_DIR = "/home/claude/fraud_detector/data"
OUT_DIR = "/home/claude/fraud_detector/outputs"


def company_level_split(feats, test_size=0.25, seed=42):
    companies = feats["company"].unique()
    fraud_companies = feats.loc[feats.is_fraud_case == 1, "company"].unique()
    clean_companies = feats.loc[feats.is_fraud_case == 0, "company"].unique()

    fraud_train, fraud_test = train_test_split(fraud_companies, test_size=test_size, random_state=seed)
    clean_train, clean_test = train_test_split(clean_companies, test_size=test_size, random_state=seed)

    train_companies = set(fraud_train) | set(clean_train)
    test_companies = set(fraud_test) | set(clean_test)

    train = feats[feats.company.isin(train_companies)].copy()
    test = feats[feats.company.isin(test_companies)].copy()
    return train, test


def fit_models(train):
    X_train = train[FEATURE_COLS].values
    y_train = train["label"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_s, y_train)

    gbm = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        class_weight="balanced", min_child_samples=5, verbose=-1,
    )
    gbm.fit(X_train, y_train)

    return scaler, logreg, gbm


def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    result = {
        "model": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 3) if len(set(y_true)) > 1 else None,
        "n_test": len(y_true), "n_positive": int(y_true.sum()),
    }
    print(f"\n--- {name} ---")
    print(json.dumps(result, indent=2))
    print(classification_report(y_true, y_pred, zero_division=0))
    return result


def leave_one_fraud_out_backtest(feats):
    """
    For each real fraud company: train LightGBM on everyone else, then score
    that company's own years. Check whether its manipulation-window years
    (label=1) score higher than its own pre-window years (label=0).
    This tests genuine early-warning capability, not just in-sample fit.
    """
    fraud_companies = feats.loc[feats.is_fraud_case == 1, "company"].unique()
    results = []

    for held_out in fraud_companies:
        train = feats[feats.company != held_out]
        test = feats[feats.company == held_out].sort_values("fiscal_year")

        X_train, y_train = train[FEATURE_COLS].values, train["label"].values
        gbm = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            class_weight="balanced", min_child_samples=5, verbose=-1,
        )
        gbm.fit(X_train, y_train)

        X_test = test[FEATURE_COLS].values
        probs = gbm.predict_proba(X_test)[:, 1]

        row = {"company": held_out}
        for fy, lab, prob in zip(test.fiscal_year, test.label, probs):
            row[f"FY{fy}_label{lab}"] = round(prob, 3)

        window_scores = probs[test.label.values == 1]
        baseline_scores = probs[test.label.values == 0]
        row["avg_window_year_score"] = round(window_scores.mean(), 3) if len(window_scores) else None
        row["avg_baseline_year_score"] = round(baseline_scores.mean(), 3) if len(baseline_scores) else None
        row["would_have_flagged"] = bool(len(window_scores) and window_scores.mean() > 0.5)
        results.append(row)

    return pd.DataFrame(results)


def shap_summary(gbm, train, path):
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(train[FEATURE_COLS])
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(sv, train[FEATURE_COLS], show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {path}")


def main():
    feats = pd.read_csv(f"{DATA_DIR}/features.csv")
    train, test = company_level_split(feats)

    print(f"Train: {len(train)} rows, {train.company.nunique()} companies, "
          f"{train.label.sum()} positive")
    print(f"Test:  {len(test)} rows, {test.company.nunique()} companies, "
          f"{test.label.sum()} positive")

    scaler, logreg, gbm = fit_models(train)

    # --- rule-based Beneish benchmark on test set ---
    beneish_pred_prob = 1 / (1 + np.exp(-test["beneish_m_score"]))  # sigmoid squash for comparability
    beneish_result = evaluate("Beneish M-Score (rule-based, threshold M > -1.78)",
                               test["label"].values,
                               (test["beneish_m_score"] > -1.78).astype(float).values,
                               threshold=0.5)

    # --- logistic regression ---
    X_test_s = scaler.transform(test[FEATURE_COLS].values)
    logreg_prob = logreg.predict_proba(X_test_s)[:, 1]
    logreg_result = evaluate("Logistic Regression", test["label"].values, logreg_prob)

    # --- LightGBM ---
    gbm_prob = gbm.predict_proba(test[FEATURE_COLS].values)[:, 1]
    gbm_result = evaluate("LightGBM", test["label"].values, gbm_prob)

    # --- leave-one-fraud-out backtest ---
    print("\n\n=== Leave-One-Fraud-Company-Out Backtest ===")
    backtest_df = leave_one_fraud_out_backtest(feats)
    print(backtest_df[["company", "avg_window_year_score", "avg_baseline_year_score",
                        "would_have_flagged"]].to_string(index=False))
    backtest_df.to_csv(f"{OUT_DIR}/leave_one_out_backtest.csv", index=False)

    n_flagged = backtest_df["would_have_flagged"].sum()
    print(f"\n{n_flagged} / {len(backtest_df)} fraud cases would have been flagged "
          f"(avg probability > 0.5 during their real manipulation-window years, "
          f"using a model that never saw that company during training).")

    # --- SHAP ---
    shap_summary(gbm, train, f"{OUT_DIR}/shap_summary.png")

    # --- save everything ---
    summary = {
        "train_rows": len(train), "test_rows": len(test),
        "beneish_benchmark": beneish_result,
        "logistic_regression": logreg_result,
        "lightgbm": gbm_result,
        "leave_one_out_flagged": f"{n_flagged}/{len(backtest_df)}",
    }
    with open(f"{OUT_DIR}/model_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    import joblib
    joblib.dump({"scaler": scaler, "logreg": logreg, "gbm": gbm, "features": FEATURE_COLS},
                f"{OUT_DIR}/trained_models.joblib")
    print(f"\nSaved trained models to {OUT_DIR}/trained_models.joblib")
    print(f"Saved results summary to {OUT_DIR}/model_results.json")


if __name__ == "__main__":
    main()
