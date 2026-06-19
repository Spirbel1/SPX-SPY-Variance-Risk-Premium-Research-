from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


def fit_classification_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit classification model family and return probabilities/classes."""
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for name, model in {
        "logistic": LogisticRegression(max_iter=2000),
        "logistic_l2": LogisticRegression(max_iter=2000, penalty="l2", solver="lbfgs"),
        "logistic_elasticnet": LogisticRegression(max_iter=4000, penalty="elasticnet", l1_ratio=0.5, solver="saga"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=5),
    }.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        out[name] = {"proba": prob, "pred": pred}

    return out


def evaluate_classification(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_true_s = pd.Series(y_true)
    pred_s = pd.Series(pred)
    proba_s = pd.Series(proba)
    true_pos_rate = float(y_true_s.mean()) if len(y_true_s) else np.nan
    pred_pos_rate = float(pred_s.mean()) if len(pred_s) else np.nan
    is_degenerate = float(pred_s.nunique(dropna=True) <= 1)

    metrics = {
        "accuracy": float(accuracy_score(y_true_s, pred_s)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_s, pred_s)),
        "precision": float(precision_score(y_true_s, pred_s, zero_division=0)),
        "recall": float(recall_score(y_true_s, pred_s, zero_division=0)),
        "f1": float(f1_score(y_true_s, pred_s, zero_division=0)),
        "brier": float(brier_score_loss(y_true_s, proba_s)),
        "true_positive_rate": true_pos_rate,
        "predicted_positive_rate": pred_pos_rate,
        "is_degenerate_classifier": is_degenerate,
        "n_predictions": float(len(y_true_s)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_s, proba_s))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def calibration_data(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})
