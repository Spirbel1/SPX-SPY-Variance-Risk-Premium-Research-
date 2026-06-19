from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    index: pd.DatetimeIndex,
    initial_train_years: int,
    test_window_years: int,
    step_years: int,
) -> List[TimeSplit]:
    """Build expanding-window walk-forward splits from a datetime index."""
    if len(index) == 0:
        return []

    start = index.min()
    end = index.max()

    train_end = start + pd.DateOffset(years=initial_train_years)
    splits: List[TimeSplit] = []

    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_window_years) - pd.Timedelta(days=1)
        if test_start > end:
            break

        clipped_test_end = min(test_end, end)
        splits.append(
            TimeSplit(
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=clipped_test_end,
            )
        )
        train_end = train_end + pd.DateOffset(years=step_years)
        if train_end >= end:
            break

    return splits
