from __future__ import annotations

import pandas as pd


def validate_datetime_index(df: pd.DataFrame) -> None:
    """Ensure dataframe is time-indexed and sorted."""
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            raise ValueError("DatetimeIndex must be sorted ascending.")
        return
    if "date" in df.columns:
        d = pd.to_datetime(df["date"])
        if not d.is_monotonic_increasing:
            raise ValueError("date column must be sorted ascending.")
        return
    raise ValueError("DataFrame must have a DatetimeIndex or 'date' column.")


def validate_no_lookahead(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> None:
    """Basic guard against accidental lookahead leakage in aligned datasets."""
    validate_datetime_index(df)

    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_features or missing_targets:
        raise ValueError(f"Missing feature/target columns: {missing_features}, {missing_targets}")

    # Features should generally be available before or on same row as targets; no direct row shift checks here,
    # but we enforce that targets are not identical to current realized values for key examples.
    if "next_5d_return" in df.columns and "log_return" in df.columns:
        overlap = (df["next_5d_return"].round(12) == df["log_return"].round(12)).mean()
        if overlap > 0.95:
            raise ValueError("Potential leakage: next_5d_return appears equal to same-day returns.")


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value report by column."""
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100.0,
        }
    ).sort_values("missing_pct", ascending=False)
    return report
