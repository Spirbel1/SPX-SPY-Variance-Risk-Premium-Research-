from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_performance(df: pd.DataFrame, ret_col: str, equity_col: str) -> dict:
    rets = df[ret_col].dropna()
    if rets.empty:
        return {
            "ret_col": ret_col,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
            "active_start_date": pd.NaT,
            "active_end_date": pd.NaT,
            "n_obs": 0,
        }

    ann_ret = (1.0 + rets.mean()) ** TRADING_DAYS - 1.0
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = ann_ret / downside if downside and downside > 0 else np.nan
    mdd = max_drawdown(df[equity_col].dropna())
    calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan

    active_dates = pd.to_datetime(df.loc[df[ret_col].notna(), "date"])

    return {
        "ret_col": ret_col,
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino) if pd.notna(sortino) else np.nan,
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "max_drawdown": float(mdd),
        "total_return": float(df[equity_col].dropna().iloc[-1] - 1.0),
        "active_start_date": active_dates.min() if not active_dates.empty else pd.NaT,
        "active_end_date": active_dates.max() if not active_dates.empty else pd.NaT,
        "n_obs": int(len(rets)),
    }


def performance_by_year(df: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    d = df[df[ret_col].notna()].copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    rows = []
    for year, g in d.groupby("year"):
        total = (1.0 + g[ret_col]).prod() - 1.0
        rows.append({"year": int(year), "return": float(total)})
    return pd.DataFrame(rows)


def performance_by_vix_regime(df: pd.DataFrame, ret_col: str, vix_col: str = "vix") -> pd.DataFrame:
    d = df[df[ret_col].notna()].copy()
    q1, q2 = d[vix_col].quantile([0.33, 0.66])
    d["vix_regime"] = pd.cut(
        d[vix_col],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "medium", "high"],
    )
    rows = []
    for regime, g in d.groupby("vix_regime", observed=True):
        total = (1.0 + g[ret_col]).prod() - 1.0
        rows.append({"vix_regime": regime, "return": float(total)})
    return pd.DataFrame(rows)


def performance_by_vrp_decile(df: pd.DataFrame, ret_col: str, vrp_col: str = "VRP_21d") -> pd.DataFrame:
    d = df[df[ret_col].notna()].copy().dropna(subset=[vrp_col])
    d["vrp_decile"] = pd.qcut(d[vrp_col], q=10, labels=False, duplicates="drop")
    rows = []
    for decile, g in d.groupby("vrp_decile"):
        total = (1.0 + g[ret_col]).prod() - 1.0
        rows.append({"vrp_decile": int(decile), "return": float(total)})
    return pd.DataFrame(rows)
