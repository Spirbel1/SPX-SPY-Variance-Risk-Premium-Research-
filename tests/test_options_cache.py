from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.databento_options_daily import download_spy_options_daily, load_spy_options_pricing_cache
from src.options.cache import cache_file_paths, default_cache_paths, ensure_cache_dirs, load_manifest, save_manifest, upsert_cache_record


def test_cache_loader_does_not_call_databento(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    paths = default_cache_paths(root)
    ensure_cache_dirs(paths)
    files = cache_file_paths(paths, "2026-01-01", "2026-01-10", symbol="SPY")

    pd.DataFrame({"date": ["2026-01-02"], "market_price": [1.23]}).to_parquet(files["pricing_snapshot"], index=False)

    # If loader tried importing databento, this would fail the test.
    monkeypatch.setattr("src.data.databento_options_daily._import_databento", lambda: (_ for _ in ()).throw(RuntimeError("should not import")))

    got = load_spy_options_pricing_cache("2026-01-01", "2026-01-10", project_root=root)
    assert got is not None
    assert len(got) == 1


def test_download_uses_cache_when_exists(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    paths = default_cache_paths(root)
    ensure_cache_dirs(paths)
    files = cache_file_paths(paths, "2026-01-01", "2026-01-10", symbol="SPY")

    pd.DataFrame({
        "date": ["2026-01-02"],
        "underlying": ["SPY"],
        "option_symbol": ["X"],
        "expiration": ["2026-01-20"],
        "strike": [100.0],
        "option_type": ["call"],
        "market_price": [1.0],
        "market_price_source": ["close"],
        "underlying_price": [101.0],
        "underlying_price_source": ["test"],
        "DTE": [18],
        "T_years": [18 / 365.0],
        "moneyness": [100.0 / 101.0],
        "log_moneyness": [0.0],
        "intrinsic_value": [1.0],
        "extrinsic_value": [0.0],
        "implied_vol_market": [0.2],
        "iv_solver_status": ["ok"],
    }).to_parquet(files["pricing_snapshot"], index=False)

    manifest = load_manifest(files["manifest"])
    upsert_cache_record(
        manifest,
        symbol="SPY",
        parent_symbol="SPY.OPT",
        start_date="2026-01-01",
        end_date="2026-01-10",
        dataset="OPRA.PILLAR",
        schemas_downloaded=["ohlcv-1d"],
        row_counts={"pricing_rows": 1},
        processed_file_paths=[str(files["pricing_snapshot"])],
        source_files_used=[],
    )
    save_manifest(files["manifest"], manifest)

    monkeypatch.setattr("src.data.databento_options_daily._read_api_key", lambda *_args, **_kwargs: "dummy")
    monkeypatch.setattr("src.data.databento_options_daily._import_databento", lambda: (_ for _ in ()).throw(RuntimeError("should not be called")))

    result = download_spy_options_daily(
        "2026-01-01",
        "2026-01-10",
        force_refresh=False,
        project_root=root,
    )
    assert result.used_cache is True
    assert result.pricing_snapshot_path.exists()
