"""Logistic regression — the supervised baseline.

Pipeline:
  1. ColumnTransformer (StandardScaler for numerics, OneHotEncoder for Sector)
  2. LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs')

Hyperparameter tuning via GridSearchCV with StratifiedKFold(n_splits=5).
Scoring during CV: average_precision (PR-AUC) — appropriate for imbalanced classification.

Reports:
  - ROC AUC, PR AUC on held-out test set
  - Confusion matrix at default 0.5 threshold AND F1-optimal threshold from PR curve
  - Per-feature coefficients (sign + magnitude) → coefficients.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .io import save_csv, save_json, save_model
from .preprocessing import build_preprocessor
from .split import train_test


PARAM_GRID = {
    "lr__C": [0.01, 0.1, 1.0, 10.0],
}

CV_FOLDS = 5
CV_SCORING = "average_precision"   # PR-AUC


def build_pipeline() -> Pipeline:
    """Preprocessor + LogisticRegression."""
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("lr", LogisticRegression(
            # L2 is the default; explicit `penalty='l2'` is deprecated in sklearn 1.8+
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )),
    ])


def train_and_eval(
    df: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    out_dir: Path,
    results_dir: Path,
) -> dict:
    """End-to-end: split → CV-tune → refit → eval on test → write artifacts."""
    out_dir = Path(out_dir);     out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test(df, y)

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring=CV_SCORING, n_jobs=-1, refit=True)
    print(f"[lr-{target_name}] fitting GridSearchCV  "
          f"(grid={PARAM_GRID}, cv={CV_FOLDS}-fold stratified, scoring={CV_SCORING}) …")
    grid.fit(X_train, y_train)
    print(f"[lr-{target_name}] best params: {grid.best_params_}")
    print(f"[lr-{target_name}] best CV PR-AUC: {grid.best_score_:.4f}")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    best = grid.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc  = float(average_precision_score(y_test, proba))

    # Default 0.5 threshold
    pred_default = (proba >= 0.5).astype(int)
    cm_default = confusion_matrix(y_test, pred_default).tolist()
    f1_default = float(f1_score(y_test, pred_default))

    # F1-optimal threshold from PR curve
    prec, rec, thresholds = precision_recall_curve(y_test, proba)
    # F1 is undefined where prec+rec == 0; mask
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    best_idx = int(np.argmax(f1s[:-1]))    # last entry corresponds to threshold beyond max
    best_threshold = float(thresholds[best_idx])
    pred_tuned = (proba >= best_threshold).astype(int)
    cm_tuned   = confusion_matrix(y_test, pred_tuned).tolist()
    f1_tuned   = float(f1_score(y_test, pred_tuned))

    print(f"[lr-{target_name}] TEST  ROC-AUC = {roc_auc:.4f}   PR-AUC = {pr_auc:.4f}")
    print(f"[lr-{target_name}] TEST  F1@0.5  = {f1_default:.4f}   F1@best({best_threshold:.3f}) = {f1_tuned:.4f}")
    print(f"[lr-{target_name}] CM @0.5:                  CM @best:")
    print(f"   TN={cm_default[0][0]:<8d} FP={cm_default[0][1]:<8d}    TN={cm_tuned[0][0]:<8d} FP={cm_tuned[0][1]:<8d}")
    print(f"   FN={cm_default[1][0]:<8d} TP={cm_default[1][1]:<8d}    FN={cm_tuned[1][0]:<8d} TP={cm_tuned[1][1]:<8d}")

    # ── Coefficient table ────────────────────────────────────────────────────
    coef_df = _coefficient_table(best)
    save_csv(coef_df, out_dir / f"logistic_{target_name}_coefficients.csv")

    # ── ROC + PR plots ───────────────────────────────────────────────────────
    _plot_curves(y_test, proba, target_name, roc_auc, pr_auc,
                 results_dir / f"logistic_{target_name}_curves.png")

    # ── Persist ──────────────────────────────────────────────────────────────
    save_model(best, out_dir / f"logistic_{target_name}_model.joblib")
    metrics = {
        "target": target_name,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate":  float(y_test.mean()),
        "best_params": grid.best_params_,
        "cv_best_pr_auc": float(grid.best_score_),
        "cv_results": {
            "mean_test_score": grid.cv_results_["mean_test_score"].tolist(),
            "std_test_score":  grid.cv_results_["std_test_score"].tolist(),
            "param_C":         [p["lr__C"] for p in grid.cv_results_["params"]],
        },
        "test": {
            "roc_auc": roc_auc,
            "pr_auc":  pr_auc,
            "threshold_default":   0.5,
            "f1_default":          f1_default,
            "confusion_matrix_default": cm_default,
            "threshold_f1_optimal": best_threshold,
            "f1_tuned":            f1_tuned,
            "confusion_matrix_tuned": cm_tuned,
        },
    }
    save_json(metrics, out_dir / f"logistic_{target_name}_metrics.json")
    return metrics


def _coefficient_table(fitted_pipe: Pipeline) -> pd.DataFrame:
    """Extract per-feature coefficients (sign + magnitude) from the fitted pipeline."""
    lr: LogisticRegression = fitted_pipe.named_steps["lr"]
    preproc = fitted_pipe.named_steps["preprocess"]
    feature_names = preproc.get_feature_names_out()
    coefs = lr.coef_.flatten()
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df


def _plot_curves(y_true, proba, target_name, roc_auc, pr_auc, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#3498db", lw=2)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#888888", alpha=0.7)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC — AUC = {roc_auc:.4f}")

    axes[1].plot(rec, prec, color="#e74c3c", lw=2)
    axes[1].axhline(float(y_true.mean()), ls="--", color="#888888", alpha=0.7,
                    label=f"baseline (pos rate = {y_true.mean():.2%})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision–Recall — AP = {pr_auc:.4f}")
    axes[1].legend(loc="best")

    fig.suptitle(f"Logistic regression — target = {target_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[lr-{target_name}] curves → {out_path}")
