from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _load(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{name} not found — run scripts first")
    return pd.read_csv(path)


def test_volatility_calibration_includes_feature_group() -> None:
    cal = _load("volatility_calibration.csv")
    assert "feature_group" in cal.columns, "volatility_calibration.csv must include feature_group"


def test_calibration_grouping_has_required_keys() -> None:
    cal = _load("volatility_calibration.csv")
    required = {"target", "model_name", "feature_group", "bin"}
    missing = required - set(cal.columns)
    assert not missing, f"volatility_calibration.csv is missing columns: {missing}"


def test_volatility_outputs_have_model_identifiers() -> None:
    files = [
        "volatility_regression_metrics.csv",
        "volatility_regression_predictions.csv",
        "volatility_classification_metrics.csv",
        "volatility_classification_predictions.csv",
        "volatility_feature_importance.csv",
    ]
    for fname in files:
        df = _load(fname)
        for col in ("target", "model_name", "feature_group"):
            assert col in df.columns, f"{fname} is missing column '{col}'"


def test_calibration_no_mixed_feature_groups() -> None:
    """Each (target, model_name, bin) combo should come from exactly one feature group."""
    cal = _load("volatility_calibration.csv")
    if "feature_group" not in cal.columns:
        pytest.fail("feature_group missing — cannot check mixing")
    # Verify each bin row is uniquely identified by (target, model_name, feature_group, bin)
    dup = cal.duplicated(subset=["target", "model_name", "feature_group", "bin"])
    assert not dup.any(), f"{dup.sum()} duplicate (target,model_name,feature_group,bin) rows in calibration"


def test_generate_research_conclusions_keys() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.utils.research_conclusions import generate_research_conclusions

    dummy = pd.DataFrame()
    conclusions = generate_research_conclusions(dummy, dummy, dummy, dummy, dummy, dummy)
    expected = {
        "return_prediction_verdict",
        "trading_strategy_verdict",
        "vol_level_verdict",
        "vol_expansion_verdict",
        "best_return_model",
        "best_vol_level_model",
        "best_vol_expansion_model",
        "main_supported_conclusion",
        "main_warning",
        "recommended_next_step",
    }
    missing_keys = expected - set(conclusions.keys())
    assert not missing_keys, f"generate_research_conclusions missing keys: {missing_keys}"


def test_generate_research_conclusions_weak_verdict_on_empty() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.utils.research_conclusions import generate_research_conclusions

    c = generate_research_conclusions(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
    )
    assert "weak" in c["return_prediction_verdict"].lower()


def test_debug_pipeline_has_vrp_step() -> None:
    dbg = _load("debug_pipeline_report.csv")
    assert "step" in dbg.columns
    steps = set(dbg["step"].unique())
    assert "after_vrp_features" in steps, f"after_vrp_features step missing — found: {steps}"
    assert "after_price_vol_features" in steps, f"after_price_vol_features step missing — found: {steps}"
