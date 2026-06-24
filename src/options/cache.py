from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OptionsCachePaths:
    raw_dir: Path
    processed_dir: Path
    manifest_dir: Path


def default_cache_paths(project_root: Path) -> OptionsCachePaths:
    return OptionsCachePaths(
        raw_dir=project_root / "data" / "raw" / "databento" / "options",
        processed_dir=project_root / "data" / "processed" / "options_pricing",
        manifest_dir=project_root / "data" / "cache_manifest",
    )


def ensure_cache_dirs(paths: OptionsCachePaths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)


def _fmt(date_str: str) -> str:
    return str(date_str).replace("-", "")


def cache_file_paths(
    paths: OptionsCachePaths,
    start_date: str,
    end_date: str,
    symbol: str = "SPY",
) -> dict[str, Path]:
    s = _fmt(start_date)
    e = _fmt(end_date)
    base = symbol.lower()
    return {
        "ohlcv": paths.raw_dir / f"{base}_options_ohlcv_1d_{s}_{e}.parquet",
        "definitions": paths.raw_dir / f"{base}_options_definitions_{s}_{e}.parquet",
        "statistics": paths.raw_dir / f"{base}_options_statistics_1d_{s}_{e}.parquet",
        "pricing_snapshot": paths.processed_dir / f"{base}_options_pricing_snapshot_{s}_{e}.parquet",
        "manifest": paths.manifest_dir / "cache_manifest.json",
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"records": []}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"records": []}


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def record_key(symbol: str, parent_symbol: str, start_date: str, end_date: str, dataset: str) -> str:
    return f"{symbol}|{parent_symbol}|{start_date}|{end_date}|{dataset}"


def get_cache_record(
    manifest: dict[str, Any],
    *,
    symbol: str,
    parent_symbol: str,
    start_date: str,
    end_date: str,
    dataset: str,
) -> dict[str, Any] | None:
    key = record_key(symbol, parent_symbol, start_date, end_date, dataset)
    for rec in manifest.get("records", []):
        if rec.get("record_key") == key:
            return rec
    return None


def upsert_cache_record(
    manifest: dict[str, Any],
    *,
    symbol: str,
    parent_symbol: str,
    start_date: str,
    end_date: str,
    dataset: str,
    schemas_downloaded: list[str],
    row_counts: dict[str, int],
    processed_file_paths: list[str],
    source_files_used: list[str],
    module_version: str = "options_pricing_v1",
) -> dict[str, Any]:
    rec = {
        "record_key": record_key(symbol, parent_symbol, start_date, end_date, dataset),
        "symbol": symbol,
        "parent_symbol": parent_symbol,
        "start_date": start_date,
        "end_date": end_date,
        "dataset": dataset,
        "schemas_downloaded": schemas_downloaded,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": row_counts,
        "processed_file_paths": processed_file_paths,
        "source_files_used": source_files_used,
        "module_version": module_version,
    }

    records = manifest.setdefault("records", [])
    key = rec["record_key"]
    replaced = False
    for i, existing in enumerate(records):
        if existing.get("record_key") == key:
            records[i] = rec
            replaced = True
            break
    if not replaced:
        records.append(rec)
    return rec
