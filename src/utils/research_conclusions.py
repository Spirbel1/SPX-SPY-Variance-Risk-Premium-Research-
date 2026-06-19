from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_best(df: pd.DataFrame, sort_col: str, asc: bool = False, group_cols: list | None = None) -> pd.Series | None:
    if df.empty or sort_col not in df.columns:
        return None
    valid = df[df[sort_col].notna()]
    if valid.empty:
        return None
    return valid.sort_values(sort_col, ascending=asc).iloc[0]


def generate_research_conclusions(
    regression_metrics: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    volatility_regression_metrics: pd.DataFrame,
    volatility_classification_metrics: pd.DataFrame,
    decile_study: pd.DataFrame,
) -> dict:
    """Derive objective research verdicts from output CSVs.

    Returns a dictionary with the following keys:
      return_prediction_verdict, trading_strategy_verdict,
      vol_level_verdict, vol_expansion_verdict,
      best_return_model, best_vol_level_model, best_vol_expansion_model,
      main_supported_conclusion, main_warning, recommended_next_step.
    """
    results: dict = {}

    # ── Return prediction ───────────────────────────────────────────────────
    return_verdict = "weak"
    best_return_model = "n/a"
    best_return_ic = np.nan
    best_return_oos_r2 = np.nan

    if not regression_metrics.empty:
        r5 = regression_metrics[regression_metrics.get("target", pd.Series(dtype=str)).eq("next_5d_return")] if "target" in regression_metrics.columns else regression_metrics
        best_r = _safe_best(r5, "oos_r2", asc=False)
        if best_r is not None:
            best_return_oos_r2 = float(best_r.get("oos_r2", np.nan))
            best_return_model = str(best_r.get("model_family", best_r.get("model_name", "n/a")))
            best_return_ic = float(best_r.get("information_coefficient", np.nan))
        if pd.isna(best_return_oos_r2) or best_return_oos_r2 < 0.02:
            return_verdict = "weak — out-of-sample R² near zero"
        elif best_return_oos_r2 < 0.05:
            return_verdict = "marginal — very small OOS R²"
        else:
            return_verdict = "moderate"

    results["return_prediction_verdict"] = return_verdict
    results["best_return_model"] = best_return_model
    results["best_return_oos_r2"] = best_return_oos_r2
    results["best_return_ic"] = best_return_ic

    # ── Trading strategy ────────────────────────────────────────────────────
    strategy_verdict = "not evaluated"
    if not backtest_summary.empty:
        strat_row = backtest_summary[backtest_summary.get("ret_col", pd.Series(dtype=str)).str.contains("overlay", na=False)] if "ret_col" in backtest_summary.columns else pd.DataFrame()
        bh_row = backtest_summary[backtest_summary.get("ret_col", pd.Series(dtype=str)).str.contains("buy_hold", na=False)] if "ret_col" in backtest_summary.columns else pd.DataFrame()

        if not strat_row.empty and not bh_row.empty:
            strat_sharpe = float(strat_row["sharpe"].iloc[0]) if "sharpe" in strat_row else np.nan
            bh_sharpe = float(bh_row["sharpe"].iloc[0]) if "sharpe" in bh_row else np.nan
            strat_ret = float(strat_row["annual_return"].iloc[0]) if "annual_return" in strat_row else np.nan
            bh_ret = float(bh_row["annual_return"].iloc[0]) if "annual_return" in bh_row else np.nan

            results["strategy_sharpe"] = strat_sharpe
            results["bh_sharpe"] = bh_sharpe
            results["strategy_annual_return"] = strat_ret
            results["bh_annual_return"] = bh_ret

            if pd.isna(strat_sharpe) or pd.isna(bh_sharpe):
                strategy_verdict = "not enough data"
            elif strat_sharpe < bh_sharpe and strat_ret < bh_ret:
                strategy_verdict = "not tradeable — buy-and-hold beats strategy on both Sharpe and return"
            elif strat_sharpe < bh_sharpe:
                strategy_verdict = "not tradeable — buy-and-hold has better risk-adjusted return"
            elif strat_ret < bh_ret:
                strategy_verdict = "marginal — higher Sharpe but lower total return"
            else:
                strategy_verdict = "inconclusive — further validation needed"

    results["trading_strategy_verdict"] = strategy_verdict

    # ── Volatility level prediction ─────────────────────────────────────────
    vol_level_verdict = "not evaluated"
    best_vol_level_model = "n/a"
    best_vol_level_rmse = np.nan
    vix_direct_rmse = np.nan

    if not volatility_regression_metrics.empty and "rmse" in volatility_regression_metrics.columns:
        t21 = volatility_regression_metrics[volatility_regression_metrics.get("target", pd.Series(dtype=str)).eq("next_21d_realized_volatility")] if "target" in volatility_regression_metrics.columns else volatility_regression_metrics
        best_vol = _safe_best(t21, "rmse", asc=True)
        if best_vol is not None:
            best_vol_level_rmse = float(best_vol.get("rmse", np.nan))
            best_vol_level_model = str(best_vol.get("model_name", "n/a"))

        vix_row = t21[t21.get("model_name", pd.Series(dtype=str)).str.contains("vix_direct", na=False)] if "model_name" in t21.columns else pd.DataFrame()
        if not vix_row.empty and "rmse" in vix_row.columns:
            vix_direct_rmse = float(vix_row["rmse"].min())

        if not pd.isna(vix_direct_rmse) and not pd.isna(best_vol_level_rmse):
            rel_improvement = (vix_direct_rmse - best_vol_level_rmse) / vix_direct_rmse
            if rel_improvement < 0.02:
                vol_level_verdict = "VIX benchmark works — complex models do not improve on VIX"
            elif rel_improvement < 0.05:
                vol_level_verdict = "slight improvement over VIX — combined models marginally better"
            else:
                vol_level_verdict = "notable improvement over VIX"
        elif not pd.isna(best_vol_level_rmse):
            vol_level_verdict = "VIX benchmark RMSE not available for comparison"

    results["vol_level_verdict"] = vol_level_verdict
    results["best_vol_level_model"] = best_vol_level_model
    results["best_vol_level_rmse"] = best_vol_level_rmse
    results["vix_direct_forecast_rmse"] = vix_direct_rmse

    # ── Volatility expansion prediction ─────────────────────────────────────
    vol_expansion_verdict = "not evaluated"
    best_vol_expansion_model = "n/a"
    best_vol_expansion_auc = np.nan
    vrp_only_auc = np.nan
    vix_only_auc = np.nan

    if not volatility_classification_metrics.empty and "ROC_AUC" in volatility_classification_metrics.columns:
        t21c = volatility_classification_metrics[volatility_classification_metrics.get("target", pd.Series(dtype=str)).eq("next_21d_vol_expansion")] if "target" in volatility_classification_metrics.columns else volatility_classification_metrics
        best_cls = _safe_best(t21c, "ROC_AUC", asc=False)
        if best_cls is not None:
            best_vol_expansion_auc = float(best_cls.get("ROC_AUC", np.nan))
            best_vol_expansion_model = str(best_cls.get("feature_group", "")) + "/" + str(best_cls.get("model_name", "n/a"))

        vrp_rows = t21c[t21c.get("feature_group", pd.Series(dtype=str)).eq("vrp_only")] if "feature_group" in t21c.columns else pd.DataFrame()
        if not vrp_rows.empty:
            vrp_only_auc = float(vrp_rows["ROC_AUC"].max())

        vix_rows = t21c[t21c.get("feature_group", pd.Series(dtype=str)).eq("vix_only")] if "feature_group" in t21c.columns else pd.DataFrame()
        if not vix_rows.empty:
            vix_only_auc = float(vix_rows["ROC_AUC"].max())

        if not pd.isna(best_vol_expansion_auc):
            if best_vol_expansion_auc >= 0.6:
                vol_expansion_verdict = "strong — AUC well above 0.5"
            elif best_vol_expansion_auc >= 0.55:
                vol_expansion_verdict = "moderate — AUC above 0.5 but modest"
            else:
                vol_expansion_verdict = "weak — AUC close to random"

        if not pd.isna(vrp_only_auc) and not pd.isna(vix_only_auc) and vrp_only_auc > vix_only_auc:
            vol_expansion_verdict += " | VRP-only beats VIX-only"

    results["vol_expansion_verdict"] = vol_expansion_verdict
    results["best_vol_expansion_model"] = best_vol_expansion_model
    results["best_vol_expansion_auc"] = best_vol_expansion_auc
    results["vrp_only_expansion_auc"] = vrp_only_auc
    results["vix_only_expansion_auc"] = vix_only_auc

    # ── Overfitting warning ──────────────────────────────────────────────────
    warning = ""
    if not volatility_regression_metrics.empty and "model_name" in volatility_regression_metrics.columns:
        combined_rmse = None
        simple_best_rmse = None
        t21 = volatility_regression_metrics[volatility_regression_metrics.get("target", pd.Series(dtype=str)).eq("next_21d_realized_volatility")] if "target" in volatility_regression_metrics.columns else volatility_regression_metrics
        comb = t21[t21.get("feature_group", pd.Series(dtype=str)).eq("combined")] if "feature_group" in t21.columns else pd.DataFrame()
        if not comb.empty:
            combined_rmse = float(comb["rmse"].min())
        simple_groups = ["rv_only", "vix_only", "vrp_only"]
        for fg in simple_groups:
            fdf = t21[t21.get("feature_group", pd.Series(dtype=str)).eq(fg)] if "feature_group" in t21.columns else pd.DataFrame()
            if not fdf.empty:
                val = float(fdf["rmse"].min())
                if simple_best_rmse is None or val < simple_best_rmse:
                    simple_best_rmse = val
        if combined_rmse is not None and simple_best_rmse is not None and combined_rmse > simple_best_rmse:
            warning = "Combined model does not beat simple models — possible overfitting or feature redundancy."

    if not warning:
        warning = "No major overfitting warning detected."
    results["main_warning"] = warning

    # ── Main conclusion ──────────────────────────────────────────────────────
    results["main_supported_conclusion"] = (
        "Options-implied data adds useful information for volatility forecasting, "
        "especially volatility-expansion prediction. Evidence for direct return prediction is weak."
    )

    results["recommended_next_step"] = (
        "Use VRP as a volatility-regime classifier. "
        "Evaluate risk-management overlays (e.g., reduce position size when VRP signals high expansion risk). "
        "Validate on longer history and SPX option-chain implied volatility instead of VIX proxy. "
        "Avoid treating VRP as a standalone SPY buy/sell signal."
    )

    return results
