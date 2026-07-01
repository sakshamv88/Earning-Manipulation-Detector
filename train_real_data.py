"""
train_real_data.py

Trains on the REAL Beneish M-Score dataset (data/real_beneish_dataset.csv):
1,239 Indian companies, 39 confirmed manipulators, 1,200 non-manipulators.
Source: user-provided CaseData.xlsx (academic Beneish-ratio dataset).

Given only 39 positive cases, a single train/test split is noisy -- so this
script reports BOTH:
  (a) Stratified 5-fold cross-validation (robust average performance)
  (b) One held-out stratified test set (for a concrete confusion matrix / SHAP)

Also fits the rule-based Beneish M-Score formula on this same data as the
benchmark to beat -- same treatment as the synthetic-data pipeline, so the
two are directly comparable.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (precision_score, recall_score, roc_auc_score,
                              average_precision_score, classification_report,
                              confusion_matrix)
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

DATA_DIR = "/home/claude/fraud_detector/data"
OUT_DIR = "/home/claude/fraud_detector/outputs"
FEATURE_COLS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]


def beneish_m_score(row):
    return (-4.84 + 0.92 * row.DSRI + 0.528 * row.GMI + 0.404 * row.AQI
            + 0.892 * row.SGI + 0.115 * row.DEPI - 0.172 * row.SGAI
            + 4.679 * row.TATA - 0.327 * row.LVGI)


def cross_validate(df, n_splits=5, seed=42):
    """Stratified k-fold CV -- robust performance estimate given small positive class."""
    X, y = df[FEATURE_COLS].values, df["label"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = {"logreg": [], "lightgbm": [], "beneish_rule": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
        logreg.fit(scaler.transform(X_tr), y_tr)
        prob = logreg.predict_proba(scaler.transform(X_te))[:, 1]
        results["logreg"].append(_fold_metrics(y_te, prob))

        gbm = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  class_weight="balanced", min_child_samples=5, verbose=-1)
        gbm.fit(X_tr, y_tr)
        prob = gbm.predict_proba(X_te)[:, 1]
        results["lightgbm"].append(_fold_metrics(y_te, prob))

        m_scores = df.iloc[test_idx].apply(beneish_m_score, axis=1)
        pred = (m_scores > -1.78).astype(int)
        results["beneish_rule"].append(_fold_metrics(y_te, pred.values.astype(float), is_prob=False))

    return results


def _fold_metrics(y_true, scores, is_prob=True):
    pred = (scores >= 0.5).astype(int) if is_prob else scores.astype(int)
    m = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, scores)
        m["pr_auc"] = average_precision_score(y_true, scores)
    else:
        m["roc_auc"] = None
        m["pr_auc"] = None
    return m


def summarize_cv(results):
    summary = {}
    for model, folds in results.items():
        keys = folds[0].keys()
        summary[model] = {
            k: round(float(np.nanmean([f[k] for f in folds if f[k] is not None])), 3)
            for k in keys
        }
    return summary


def held_out_test(df, seed=42, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)

    X_train, y_train = train[FEATURE_COLS].values, train["label"].values
    X_test, y_test = test[FEATURE_COLS].values, test["label"].values

    scaler = StandardScaler().fit(X_train)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(scaler.transform(X_train), y_train)

    gbm = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              class_weight="balanced", min_child_samples=5, verbose=-1)
    gbm.fit(X_train, y_train)

    logreg_prob = logreg.predict_proba(scaler.transform(X_test))[:, 1]
    gbm_prob = gbm.predict_proba(X_test)[:, 1]
    m_scores = test.apply(beneish_m_score, axis=1)
    beneish_pred = (m_scores > -1.78).astype(int)

    print(f"\nHeld-out test set: {len(test)} companies, {y_test.sum()} manipulators\n")

    print("--- Beneish M-Score rule (M > -1.78) ---")
    print(confusion_matrix(y_test, beneish_pred))
    print(classification_report(y_test, beneish_pred, zero_division=0))

    print("--- Logistic Regression ---")
    logreg_pred = (logreg_prob >= 0.5).astype(int)
    print(confusion_matrix(y_test, logreg_pred))
    print(classification_report(y_test, logreg_pred, zero_division=0))

    print("--- LightGBM ---")
    gbm_pred = (gbm_prob >= 0.5).astype(int)
    print(confusion_matrix(y_test, gbm_pred))
    print(classification_report(y_test, gbm_pred, zero_division=0))

    return scaler, logreg, gbm, train, test


def shap_summary(gbm, train, path):
    explainer = shap.TreeExplainer(gbm)
    sv = explainer.shap_values(train[FEATURE_COLS])
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure()
    shap.summary_plot(sv, train[FEATURE_COLS], show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary to {path}")


def main():
    df = pd.read_csv(f"{DATA_DIR}/real_beneish_dataset.csv")
    print(f"Loaded real dataset: {len(df)} companies, {df['label'].sum()} manipulators "
          f"({df['label'].mean()*100:.1f}% base rate)\n")

    print("=== Stratified 5-Fold Cross-Validation (real data) ===")
    cv_results = cross_validate(df)
    cv_summary = summarize_cv(cv_results)
    print(json.dumps(cv_summary, indent=2))

    scaler, logreg, gbm, train, test = held_out_test(df)

    shap_summary(gbm, train, f"{OUT_DIR}/shap_summary_real_data.png")

    import joblib
    joblib.dump({"scaler": scaler, "logreg": logreg, "gbm": gbm, "features": FEATURE_COLS},
                f"{OUT_DIR}/trained_models_real_data.joblib")

    with open(f"{OUT_DIR}/model_results_real_data.json", "w") as f:
        json.dump({"n_companies": len(df), "n_manipulators": int(df["label"].sum()),
                    "cv_5fold": cv_summary}, f, indent=2)

    print(f"\nSaved model to {OUT_DIR}/trained_models_real_data.joblib")
    print(f"Saved results to {OUT_DIR}/model_results_real_data.json")


if __name__ == "__main__":
    main()
