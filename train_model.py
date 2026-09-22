"""
Beneish M-Score fraud detector, trained on real financial ratio data
(1,239 Indian companies, 39 confirmed manipulators).

Compares the classic rule-based M-Score cutoff against a couple of ML models
to see whether learning the weights from data actually beats the fixed
formula from the original 1999 paper.
"""

import json
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
import joblib

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]


def beneish_score(row):
    return (-4.84 + 0.92 * row.DSRI + 0.528 * row.GMI + 0.404 * row.AQI
            + 0.892 * row.SGI + 0.115 * row.DEPI - 0.172 * row.SGAI
            + 4.679 * row.TATA - 0.327 * row.LVGI)


def score_fold(y_true, scores, is_prob=True):
    pred = (scores >= 0.5).astype(int) if is_prob else scores.astype(int)
    out = {"precision": precision_score(y_true, pred, zero_division=0),
           "recall": recall_score(y_true, pred, zero_division=0)}
    if len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    return out


def cross_validate(df, folds=5, seed=42):
    X, y = df[FEATURES].values, df["label"].values
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = {"logreg": [], "lightgbm": [], "beneish_rule": []}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(scaler.transform(X_tr), y_tr)
        results["logreg"].append(score_fold(y_te, lr.predict_proba(scaler.transform(X_te))[:, 1]))

        gbm = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  class_weight="balanced", min_child_samples=5, verbose=-1)
        gbm.fit(X_tr, y_tr)
        results["lightgbm"].append(score_fold(y_te, gbm.predict_proba(X_te)[:, 1]))

        m_scores = df.iloc[test_idx].apply(beneish_score, axis=1)
        pred = (m_scores > -1.78).astype(int).values
        results["beneish_rule"].append(score_fold(y_te, pred.astype(float), is_prob=False))

    return {model: {k: round(float(np.mean([f[k] for f in folds])), 3)
                     for k in folds[0]} for model, folds in results.items()}


def holdout_split(df, seed=42):
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    X_train, y_train = train[FEATURES].values, train["label"].values
    X_test, y_test = test[FEATURES].values, test["label"].values

    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(X_train), y_train)

    gbm = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              class_weight="balanced", min_child_samples=5, verbose=-1)
    gbm.fit(X_train, y_train)

    print(f"\nHoldout set: {len(test)} companies, {int(y_test.sum())} manipulators\n")

    for name, pred in [
        ("Beneish rule", (test.apply(beneish_score, axis=1) > -1.78).astype(int)),
        ("Logistic Regression", (lr.predict_proba(scaler.transform(X_test))[:, 1] >= 0.5).astype(int)),
        ("LightGBM", (gbm.predict_proba(X_test)[:, 1] >= 0.5).astype(int)),
    ]:
        print(f"--- {name} ---")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, zero_division=0))

    return scaler, lr, gbm, train, test


def main():
    df = pd.read_csv("data/beneish_dataset.csv")
    print(f"{len(df)} companies, {int(df['label'].sum())} manipulators "
          f"({df['label'].mean()*100:.1f}% base rate)")

    print("\n5-fold cross-validation:")
    cv = cross_validate(df)
    print(json.dumps(cv, indent=2))

    scaler, lr, gbm, train, test = holdout_split(df)

    explainer = shap.TreeExplainer(gbm)
    sv = explainer.shap_values(train[FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure()
    shap.summary_plot(sv, train[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"scaler": scaler, "logreg": lr, "gbm": gbm, "features": FEATURES},
                "outputs/model.joblib")
    with open("outputs/cv_results.json", "w") as f:
        json.dump(cv, f, indent=2)

    print("\nSaved model.joblib, cv_results.json, shap_summary.png to outputs/")


if __name__ == "__main__":
    main()
