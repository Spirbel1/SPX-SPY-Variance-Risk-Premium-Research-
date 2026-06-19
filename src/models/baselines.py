from __future__ import annotations

import numpy as np
import pandas as pd


def historical_mean_regression(train_y: pd.Series, n: int) -> np.ndarray:
    """Predict constant historical mean for regression."""
    mean_val = float(train_y.mean())
    return np.full(n, mean_val)


def majority_class_baseline(train_y: pd.Series, n: int) -> np.ndarray:
    """Predict majority class for classification."""
    mode_val = int(train_y.mode().iloc[0]) if not train_y.mode().empty else 0
    return np.full(n, mode_val)
