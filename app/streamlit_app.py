from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import SETTINGS
from src.ui.options_pricing_dashboard import render_options_pricing_dashboard
from src.utils.research_conclusions import generate_research_conclusions

st.set_page_config(page_title="SPX/SPY VRP Research", layout="wide")
st.title("SPX/SPY Variance Risk Premium - Research Dashboard")
st.caption(
    "Goal: understand whether options-implied data (VIX, VRP) predicts SPY returns and volatility. "
    "This is a research project, not a trading system."
)

# --------------------------- helpers ----------------------------------------

PROCESSED = SETTINGS.data_processed_dir

REQUIRED_FILES = {
    "regression_metrics.csv": "python scripts/run_pipeline.py",
    "classification_metrics.csv": "python scripts/run_pipeline.py",
    "backtest_summary.csv": "python scripts/run_backtest.py",
    "volatility_regression_metrics.csv": "python scripts/run_volatility_experiment.py",
    "volatility_classification_metrics.csv": "python scripts/run_volatility_experiment.py",
    "volatility_decile_study.csv": "python scripts/run_volatility_experiment.py",
    "volatility_feature_importance.csv": "python scripts/run_volatility_experiment.py",
    "volatility_calibration.csv": "python scripts/run_volatility_experiment.py",
}


@st.cache_data(ttl=120)
def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_dataset() -> pd.DataFrame:
    if not SETTINGS.dataset_path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(SETTINGS.dataset_path)
    if "date" not in d.columns:
        d = d.reset_index().rename(columns={"index": "date"})
    d["date"] = pd.to_datetime(d["date"])
    return d


def _fmt_vol_pts(v, decimals: int = 2) -> str:
    if pd.isna(v) if not isinstance(v, (int, float)) else False:
        return "n/a"
    try:
        if np.isnan(float(v)):
            return "n/a"
        return f"{float(v) * 100:.{decimals}f} vol pts"
    except Exception:
        return "n/a"


def _fmt_pct(v, decimals: int = 1) -> str:
    try:
        if np.isnan(float(v)):
            return "n/a"
        return f"{float(v) * 100:.{decimals}f}%"
    except Exception:
        return "n/a"


def _verdict_badge(text: str) -> None:
    colour = "#d32f2f"
    lo = text.lower()
    if any(k in lo for k in ["strong", "good", "beats vix", "improvement", "notable"]):
        colour = "#388e3c"
    elif any(k in lo for k in ["marginal", "inconclusive", "slight", "moderate"]):
        colour = "#f57c00"
    st.markdown(
        f'<div style="background:{colour};color:white;padding:8px 14px;border-radius:6px;'
        f'font-weight:600;margin-bottom:6px;">{text}</div>',
        unsafe_allow_html=True,
    )


def _quality_flag(ok: bool, label: str) -> None:
    icon = "[OK]" if ok else "[WARN]"
    bg = "#e8f5e9" if ok else "#ffebee"
    st.markdown(
        f'<div style="background:{bg};padding:4px 10px;border-radius:4px;margin:2px 0;">{icon} {label}</div>',
        unsafe_allow_html=True,
    )


def _missing_file_warning(name: str) -> None:
    cmd = REQUIRED_FILES.get(name, "run the relevant pipeline script")
    st.warning(f"Missing `{name}`. Run:\n```\n{cmd}\n```")


# Load all data

df = load_dataset()
reg_metrics = load_csv("regression_metrics.csv")
cls_metrics = load_csv("classification_metrics.csv")
reg_preds = load_csv("regression_predictions.csv")
cls_preds = load_csv("classification_predictions.csv")
reg_coef = load_csv("regression_coefficients.csv")
bt = load_csv("backtest_timeseries.csv")
bt_summary = load_csv("backtest_summary.csv")
rob_year = load_csv("robustness_by_year.csv")
rob_vix = load_csv("robustness_by_vix_regime.csv")
rob_decile = load_csv("robustness_by_vrp_decile.csv")
rule_summary = load_csv("rule_based_strategy_summary.csv")
rule_ts = load_csv("rule_based_strategy_timeseries.csv")
vol_reg_m = load_csv("volatility_regression_metrics.csv")
vol_reg_p = load_csv("volatility_regression_predictions.csv")
vol_cls_m = load_csv("volatility_classification_metrics.csv")
vol_cls_p = load_csv("volatility_classification_predictions.csv")
vol_cal = load_csv("volatility_calibration.csv")
vol_fi = load_csv("volatility_feature_importance.csv")
vol_decile = load_csv("volatility_decile_study.csv")
miss_rpt = load_csv("missing_data_report.csv")
sample_rpt = load_csv("model_sample_report.csv")
debug_rpt = load_csv("debug_pipeline_report.csv")

conclusions = generate_research_conclusions(
    reg_metrics, cls_metrics, bt_summary, vol_reg_m, vol_cls_m, vol_decile
)

# Tab layout

tabs = st.tabs([
    "1 - Executive Summary",
    "2 - Data Quality",
    "3 - Return Prediction",
    "4 - Trading Backtest",
    "5 - Vol Level Prediction",
    "6 - Vol Expansion Prediction",
    "7 - Decile & Regime Analysis",
    "8 - Feature Importance",
    "9 - Critical Conclusions",
    "10 - Options Pricing & Greeks",
])


# TAB 1 - Executive Summary
with tabs[0]:
    st.subheader("Research Verdicts")
    st.markdown(
        "_The strongest evidence is not that options data predicts SPY direction. "
        "The strongest evidence is that VRP helps predict volatility expansion, "
        "especially over 21 trading days._"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Return Prediction**")
        _verdict_badge(conclusions.get("return_prediction_verdict", "not evaluated"))
        st.markdown("**Directional Trading Strategy**")
        _verdict_badge(conclusions.get("trading_strategy_verdict", "not evaluated"))
    with c2:
        st.markdown("**Volatility-Level Prediction**")
        _verdict_badge(conclusions.get("vol_level_verdict", "not evaluated"))
        st.markdown("**Volatility-Expansion Prediction**")
        _verdict_badge(conclusions.get("vol_expansion_verdict", "not evaluated"))

    st.divider()
    st.subheader("Key Metrics")
    mc = st.columns(3)
    with mc[0]:
        r2 = conclusions.get("best_return_oos_r2", float("nan"))
        st.metric("Best return model OOS R^2", f"{r2:.4f}" if isinstance(r2, float) and not np.isnan(r2) else "n/a")
        st.caption(f"Model: {conclusions.get('best_return_model', 'n/a')}")
    with mc[1]:
        st.metric("Best vol-level RMSE (21d)", _fmt_vol_pts(conclusions.get("best_vol_level_rmse")))
        st.caption(f"Model: {conclusions.get('best_vol_level_model', 'n/a')}")
        st.metric("VIX direct forecast RMSE", _fmt_vol_pts(conclusions.get("vix_direct_forecast_rmse")))
    with mc[2]:
        auc = conclusions.get("best_vol_expansion_auc", float("nan"))
        st.metric("Best vol-expansion AUC (21d)", f"{auc:.3f}" if isinstance(auc, float) and not np.isnan(auc) else "n/a")
        st.caption(f"Model: {conclusions.get('best_vol_expansion_model', 'n/a')}")
        vrp_auc = conclusions.get("vrp_only_expansion_auc", float("nan"))
        vix_auc = conclusions.get("vix_only_expansion_auc", float("nan"))
        st.metric("VRP-only AUC (21d expansion)", f"{vrp_auc:.3f}" if isinstance(vrp_auc, float) and not np.isnan(vrp_auc) else "n/a")
        st.metric("VIX-only AUC (21d expansion)", f"{vix_auc:.3f}" if isinstance(vix_auc, float) and not np.isnan(vix_auc) else "n/a")

    st.divider()
    st.markdown(f"**Main supported conclusion:** {conclusions.get('main_supported_conclusion', '')}")
    st.markdown(f"**Main warning:** {conclusions.get('main_warning', '')}")
    st.markdown(f"**Recommended next step:** {conclusions.get('recommended_next_step', '')}")


# TAB 2 - Data Quality
with tabs[1]:
    st.subheader("Data Quality Validation")

    all_present = True
    for fname in REQUIRED_FILES:
        present = (PROCESSED / fname).exists()
        if not present:
            all_present = False
        _quality_flag(present, f"`{fname}`")

    overall = "PASS" if all_present else "WARNING - some files are missing"
    st.info(f"**Data quality status:** {overall}")

    if not debug_rpt.empty:
        st.subheader("Pipeline Debug Steps")
        # check that VRP features appear correctly at the right step
        if "step" in debug_rpt.columns and "n_missing_VRP_21d" in debug_rpt.columns and "n_rows" in debug_rpt.columns:
            vrp_step = debug_rpt[debug_rpt["step"] == "after_vrp_features"]
            if not vrp_step.empty:
                vrp21_miss = float(vrp_step["n_missing_VRP_21d"].iloc[0])
                n_rows = float(vrp_step["n_rows"].iloc[0])
                _quality_flag(vrp21_miss / n_rows < 0.30, f"VRP_21d after vrp-features step: {int(vrp21_miss)} missing / {int(n_rows)} rows")
            price_step = debug_rpt[debug_rpt["step"] == "after_price_vol_features"]
            if not price_step.empty:
                st.caption("Note: VRP columns are expected to be missing at `after_price_vol_features` - they are created in the next step.")
        st.dataframe(debug_rpt, width="stretch")
    else:
        _missing_file_warning("debug_pipeline_report.csv")

    st.subheader("Automated Quality Flags")
    if not miss_rpt.empty:
        miss2 = miss_rpt.reset_index().rename(columns={"index": "column"}) if "column" not in miss_rpt.columns else miss_rpt
        for col in ["VRP_21d", "vrp_zscore_252d", "vix_pct_252d", "realized_vol_annualized_21d"]:
            row = miss2[miss2["column"] == col] if "column" in miss2.columns else pd.DataFrame()
            if not row.empty and "missing_pct" in row.columns:
                mp = float(row["missing_pct"].iloc[0])
                _quality_flag(mp < 50.0, f"`{col}` missing: {mp:.1f}%")
        st.dataframe(miss2, width="stretch")
    else:
        _missing_file_warning("missing_data_report.csv")

    _quality_flag(
        "feature_group" in vol_cal.columns if not vol_cal.empty else False,
        "volatility_calibration.csv contains feature_group column"
    )

    if not sample_rpt.empty:
        st.subheader("Model Sample Report")
        st.dataframe(sample_rpt, width="stretch")


# TAB 3 - Return Prediction
with tabs[2]:
    st.subheader("Return Prediction")
    st.error(
        "**Return-prediction evidence is weak.** "
        "Any apparent hit rate must be compared against the market's natural upward drift. "
        "Balanced accuracy and out-of-sample R^2 are more important than raw accuracy."
    )

    if reg_metrics.empty:
        _missing_file_warning("regression_metrics.csv")
    else:
        target_ret = st.selectbox("Return target", ["next_5d_return", "next_21d_return"], key="ret_target")
        m = reg_metrics[reg_metrics["target"] == target_ret].copy() if "target" in reg_metrics.columns else reg_metrics.copy()
        sort_col = "oos_r2" if "oos_r2" in m.columns else ("r2" if "r2" in m.columns else None)
        m_sorted = m.sort_values(sort_col, ascending=False) if sort_col else m
        st.markdown("**Best regression models (sorted by OOS R^2)**")
        st.dataframe(m_sorted, width="stretch")

        if sort_col and "model_family" in m.columns:
            fig = px.bar(m_sorted.head(20), x="model_family", y=sort_col,
                         title=f"OOS R^2 by model - {target_ret}", color="model_family")
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="zero baseline")
            st.plotly_chart(fig, width="stretch")

    if cls_metrics.empty:
        _missing_file_warning("classification_metrics.csv")
    else:
        target_cls = st.selectbox(
            "Classification target", ["next_5d_positive_return", "next_21d_positive_return"], key="cls_ret_target"
        )
        mc_df = cls_metrics[cls_metrics["target"] == target_cls].copy() if "target" in cls_metrics.columns else cls_metrics.copy()
        auc_col = "roc_auc" if "roc_auc" in mc_df.columns else ("ROC_AUC" if "ROC_AUC" in mc_df.columns else None)
        mc_sorted = mc_df.sort_values(auc_col, ascending=False) if auc_col else mc_df

        if "is_degenerate_classifier" in mc_df.columns:
            n_deg = int((mc_df["is_degenerate_classifier"] > 0).sum())
            if n_deg > 0:
                st.warning(f"[WARN] {n_deg} degenerate classifier(s) detected - predicting only one class.")
        ba_col = "balanced_accuracy" if "balanced_accuracy" in mc_df.columns else None
        if ba_col:
            low_ba = int((mc_df[ba_col] < 0.52).sum())
            if low_ba > 0:
                st.warning(f"[WARN] {low_ba} model(s) have balanced accuracy below 52% - essentially random.")

        st.dataframe(mc_sorted, width="stretch")
        if auc_col and "model_family" in mc_df.columns:
            fig3 = px.bar(mc_sorted.head(20), x="model_family", y=auc_col,
                          title=f"AUC by model - {target_cls}", color="model_family")
            fig3.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="random baseline")
            st.plotly_chart(fig3, width="stretch")


# TAB 4 - Trading Backtest
with tabs[3]:
    st.subheader("Trading Backtest")
    st.warning(
        "**The current trading overlay is not a confirmed profitable strategy.** "
        "It may reduce drawdown, but it gives up too much upside and does not clearly beat "
        "buy-and-hold on Sharpe or total return."
    )

    if bt.empty:
        _missing_file_warning("backtest_timeseries.csv")
    else:
        bt["date"] = pd.to_datetime(bt["date"])
        eq_cols = [c for c in ["equity_overlay_ret", "equity_long_short_ret", "equity_buy_hold_ret"] if c in bt.columns]
        if eq_cols:
            st.plotly_chart(
                px.line(bt, x="date", y=eq_cols, title="Strategy vs Buy-and-Hold (active prediction period only)"),
                width="stretch",
            )

        for ec in ["equity_overlay_ret", "equity_buy_hold_ret"]:
            if ec in bt.columns:
                eq = bt[ec].dropna()
                bt[f"dd_{ec}"] = eq / eq.cummax() - 1
        dd_cols = [c for c in bt.columns if c.startswith("dd_")]
        if dd_cols:
            st.plotly_chart(px.line(bt, x="date", y=dd_cols, title="Drawdown comparison"), width="stretch")

    if not bt_summary.empty:
        st.subheader("Performance Summary (active period)")
        disp = bt_summary.copy()
        for c in ["annual_return", "annual_volatility"]:
            if c in disp.columns:
                disp[f"{c}_fmt"] = disp[c].apply(_fmt_pct)
        st.dataframe(disp, width="stretch")

        strat_sharpe = conclusions.get("strategy_sharpe", float("nan"))
        bh_sharpe = conclusions.get("bh_sharpe", float("nan"))
        if isinstance(strat_sharpe, float) and isinstance(bh_sharpe, float) and not np.isnan(strat_sharpe) and not np.isnan(bh_sharpe):
            if bh_sharpe > strat_sharpe:
                st.error(
                    f"Buy-and-hold Sharpe ({bh_sharpe:.2f}) exceeds strategy Sharpe ({strat_sharpe:.2f}). "
                    "Strategy does not improve risk-adjusted return."
                )
            else:
                st.info(f"Strategy Sharpe: {strat_sharpe:.2f} | Buy-and-hold Sharpe: {bh_sharpe:.2f}")

    if not rule_summary.empty:
        st.subheader("Rule-Based Strategy Comparison")
        st.dataframe(rule_summary, width="stretch")

    for label, data in [("Robustness by Year", rob_year), ("Robustness by VIX Regime", rob_vix), ("Robustness by VRP Decile", rob_decile)]:
        if not data.empty:
            st.subheader(label)
            st.dataframe(data, width="stretch")


# TAB 5 - Volatility Level Prediction
with tabs[4]:
    st.subheader("Volatility Level Prediction")
    st.markdown(
        "**Model type used in this tab:** supervised **regression** models "
        "(they predict a numeric volatility value, not a class)."
    )
    st.caption(
        "The exact algorithm is shown in `model_name` (e.g., linear model, random forest, gradient boosting), "
        "and feature set is shown in `feature_group`."
    )
    st.markdown(
        """
This tab predicts the **future realized volatility level** (a numeric forecast),
not a binary expansion label.
        """
    )

    with st.expander("What this tab is testing (definitions and equations)"):
        st.markdown(
            r"""
**Forecast targets**

- `next_5d_realized_volatility`
- `next_21d_realized_volatility`

These are forward-window realized volatility outcomes:

$$
	ext{RVOL}_{5,t}^{\text{future}} = \text{StdDev}(r_{t+1},\ldots,r_{t+5})\times\sqrt{252}
$$

$$
	ext{RVOL}_{21,t}^{\text{future}} = \text{StdDev}(r_{t+1},\ldots,r_{t+21})\times\sqrt{252}
$$

with daily log returns:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

The model compares today's features (including VIX/VRP and price-vol features)
to future realized-volatility values that actually occurred historically.
            """
        )

    with st.expander("How to read level-forecast metrics"):
        st.markdown(
            """
**RMSE / MAE**

- Lower is better.
- In this dashboard, `rmse_vol_pts` and `mae_vol_pts` convert decimal volatility error
  into volatility points for easier interpretation.

**Out-of-sample R^2 (vs RV21 baseline)**

- `> 0`: improves on baseline.
- `= 0`: similar to baseline.
- `< 0`: worse than baseline.

**Predicted vs Actual plot**

- Tight clustering around the 45-degree line indicates better calibration and fit.

**Time-series error plot**

- Persistent bias above/below zero indicates systematic over/under-prediction.
            """
        )

    st.info(
        "**VIX is a strong benchmark for future volatility level.** "
        "This result is expected because VIX is built from options-implied volatility. "
        "VIX-only models are the benchmark; combined models are only useful if they beat VIX-only out-of-sample."
    )

    if vol_reg_m.empty:
        _missing_file_warning("volatility_regression_metrics.csv")
    else:
        target_vl = st.selectbox(
            "Volatility regression target",
            ["next_21d_realized_volatility", "next_5d_realized_volatility"],
            key="vl_target",
        )
        if target_vl == "next_21d_realized_volatility":
            st.info(
                "21d realized-vol forecasts are typically smoother and less event-noisy than 5d forecasts."
            )
        else:
            st.warning(
                "5d realized-vol forecasts are more sensitive to short-lived shocks and event timing."
            )

        vm = vol_reg_m[vol_reg_m["target"] == target_vl].copy() if "target" in vol_reg_m.columns else vol_reg_m.copy()
        vm_sorted = vm.sort_values("rmse", ascending=True) if "rmse" in vm.columns else vm
        vm_disp = vm_sorted.copy()
        for col in ["rmse", "mae"]:
            if col in vm_disp.columns:
                vm_disp[f"{col}_vol_pts"] = (vm_disp[col] * 100).round(3)

        if not vm_sorted.empty:
            best_row = vm_sorted.iloc[0]
            best_model = best_row.get("model_name", "?")
            best_fg = best_row.get("feature_group", "?")
            best_rmse = float(best_row["rmse"]) if "rmse" in best_row.index and pd.notna(best_row["rmse"]) else float("nan")
            best_mae = float(best_row["mae"]) if "mae" in best_row.index and pd.notna(best_row["mae"]) else float("nan")
            rmse_str = f"{best_rmse * 100:.3f} vol pts" if not np.isnan(best_rmse) else "n/a"
            mae_str = f"{best_mae * 100:.3f} vol pts" if not np.isnan(best_mae) else "n/a"
            st.markdown(
                f"**Best model for {target_vl}: {best_fg}/{best_model} | RMSE: {rmse_str} | MAE: {mae_str}.**"
            )

            if "OOS_R2_vs_rv21_baseline" in best_row.index and pd.notna(best_row["OOS_R2_vs_rv21_baseline"]):
                best_r2 = float(best_row["OOS_R2_vs_rv21_baseline"])
                if best_r2 > 0:
                    st.success(
                        f"Best model improves over the RV21 baseline (OOS R^2 = {best_r2:.3f})."
                    )
                elif best_r2 < 0:
                    st.warning(
                        f"Best model is still below the RV21 baseline (OOS R^2 = {best_r2:.3f})."
                    )
                else:
                    st.info("Best model is approximately equal to the RV21 baseline (OOS R^2 ~ 0).")

            if isinstance(best_fg, str):
                fg = best_fg.lower()
                if "vix" in fg:
                    st.info("VIX-based feature group leads, consistent with VIX as a strong level-vol benchmark.")
                elif "vrp" in fg:
                    st.info("VRP-based group leads, suggesting relative risk-pricing features help level forecasts here.")
                elif "combined" in fg:
                    st.info("Combined group leads; confirm this remains robust out-of-sample and across regimes.")

        st.markdown("**All models - sorted by RMSE (lower = better)**")
        st.dataframe(vm_disp, width="stretch")

        if "rmse" in vm.columns and "feature_group" in vm.columns:
            fig = px.bar(vm_sorted, x="model_name", y="rmse", color="feature_group",
                         barmode="group", title=f"RMSE by model (decimal vol) - {target_vl}",
                         labels={"rmse": "RMSE (decimal)"})
            st.plotly_chart(fig, width="stretch")

        if "OOS_R2_vs_rv21_baseline" in vm.columns and "feature_group" in vm.columns:
            fig_r2 = px.bar(vm_sorted, x="model_name", y="OOS_R2_vs_rv21_baseline", color="feature_group",
                            barmode="group", title=f"OOS R^2 vs RV21 baseline - {target_vl}")
            fig_r2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="RV21 level")
            st.plotly_chart(fig_r2, width="stretch")

        if not vol_reg_p.empty:
            pred_tgt = vol_reg_p[vol_reg_p["target"] == target_vl].copy() if "target" in vol_reg_p.columns else vol_reg_p.copy()
            if pred_tgt.empty:
                st.warning(f"No prediction rows found for target `{target_vl}`.")
            elif "feature_group" not in pred_tgt.columns or "model_name" not in pred_tgt.columns:
                st.warning("Prediction file is missing `feature_group` or `model_name` columns.")
            else:
                fgroups = sorted(pred_tgt["feature_group"].dropna().unique())
                if not fgroups:
                    st.warning(f"No feature groups available for target `{target_vl}`.")
                else:
                    sel_fg = st.selectbox("Feature group for chart", fgroups, key="vl_fg")
                    pred_fg = pred_tgt[pred_tgt["feature_group"] == sel_fg].copy()
                    models = sorted(pred_fg["model_name"].dropna().unique())
                    if not models:
                        st.warning(f"No models available for `{sel_fg}` and target `{target_vl}`.")
                    else:
                        sel_model = st.selectbox("Model for chart", models, key="vl_model")
                        pp = pred_fg[pred_fg["model_name"] == sel_model].copy()
                        if pp.empty:
                            st.warning(
                                f"No rows available for target `{target_vl}`, feature group `{sel_fg}`, and model `{sel_model}`."
                            )
                        else:
                            pp["date"] = pd.to_datetime(pp["date"])
                            pp = pp.sort_values("date")
                            pp["y_true_pct"] = pp["y_true"] * 100
                            pp["y_pred_pct"] = pp["y_pred"] * 100
                            pp["error_vol_pts"] = (pp["y_pred"] - pp["y_true"]) * 100
                            st.plotly_chart(
                                px.scatter(pp, x="y_true_pct", y="y_pred_pct",
                                           title=f"Predicted vs Actual realized vol (%) - {sel_fg}/{sel_model}"),
                                width="stretch",
                            )
                            st.plotly_chart(
                                px.line(pp, x="date", y=["y_true_pct", "y_pred_pct"],
                                        title="Actual vs Predicted over time (vol %)"),
                                width="stretch",
                            )
                            st.plotly_chart(
                                px.line(pp, x="date", y="error_vol_pts", title="Prediction error over time (vol pts)"),
                                width="stretch",
                            )

                            mean_err = float(pp["error_vol_pts"].mean()) if not pp["error_vol_pts"].empty else float("nan")
                            if not np.isnan(mean_err):
                                if mean_err > 0:
                                    st.caption("Average error is positive: model slightly over-predicts realized volatility.")
                                elif mean_err < 0:
                                    st.caption("Average error is negative: model slightly under-predicts realized volatility.")
                                else:
                                    st.caption("Average error is near zero: little aggregate directional bias.")

        st.divider()
        st.markdown(
            """
**Bottom line for this tab**

- Strong performance means the model estimates future volatility *levels* better than baseline.
- This does **not** directly imply SPY direction predictability.
- Level forecasts are often most useful for risk budgeting, volatility targeting, and options-structure selection.
            """
        )


# TAB 6 - Volatility Expansion Prediction
with tabs[5]:
    st.subheader("Volatility Expansion Prediction")
    st.markdown(
        "**Model type used in this tab:** supervised **classification** models "
        "(they predict probability of expansion, then class 0/1 by threshold)."
    )
    st.caption(
        "The exact algorithm is shown in `model_name`, and the feature-set variant is shown in `feature_group`."
    )
    st.markdown(
        """
This tab does **not** test whether SPY will go up or down.
It tests whether **future realized volatility** will be higher than the current
21-day realized-volatility baseline.
        """
    )

    st.info(
        "This project has two related volatility tasks: "
        "(1) volatility level prediction (predict future RVOL value) and "
        "(2) volatility expansion classification (predict whether future RVOL exceeds current 21d RVOL)."
    )

    with st.expander("Two related tasks: level prediction vs expansion classification"):
        st.markdown(
            r"""
**1) Volatility level prediction (regression task)**

Predicts the future realized-volatility level over a forward window.

$$
	ext{RVOL}_{5,t}^{\text{future}} = \text{StdDev}(r_{t+1},\ldots,r_{t+5})\times\sqrt{252}
$$

$$
	ext{RVOL}_{21,t}^{\text{future}} = \text{StdDev}(r_{t+1},\ldots,r_{t+21})\times\sqrt{252}
$$

The module then compares predictions vs realized outcomes historically using metrics such as RMSE, MAE, and out-of-sample $R^2$.

**2) Volatility expansion classification (this tab)**

This is a binary target, not a direct volatility-value forecast:

$$
	ext{Expansion}_{5,t}=\mathbb{1}\left[\text{RVOL}_{5,t}^{\text{future}} > \text{RVOL}_{21,t}^{\text{current}}\right]
$$

$$
	ext{Expansion}_{21,t}=\mathbb{1}\left[\text{RVOL}_{21,t}^{\text{future}} > \text{RVOL}_{21,t}^{\text{current}}\right]
$$

The classifier outputs a probability of expansion, then converts it to class by threshold (typically 0.50).
            """
        )

    with st.expander("What this tab is testing (definitions and equations)"):
        st.markdown(
            r"""
**Target definition**

- Baseline: `realized_vol_annualized_21d`
- Future targets: `next_5d_realized_volatility`, `next_21d_realized_volatility`

Expansion label:

$$
	ext{next\_nd\_vol\_expansion}_t = \mathbb{1}\left[\text{RVOL}_{n,t}^{\text{future}} > \text{RVOL}_{21,t}^{\text{current}}\right]
$$

where class `1` means expansion and class `0` means no expansion.

**Realized volatility construction**

Daily log return:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Annualized rolling realized volatility:

$$
	ext{RVOL}_{n,t} = \text{StdDev}(r_{t-n+1}, \ldots, r_t) \times \sqrt{252}
$$

**VRP construction (VIX proxy)**

Implied variance from VIX:

$$
	ext{ImpliedVariance}_t = \left(\frac{\text{VIX}_t}{100}\right)^2
$$

Variance Risk Premium:

$$
	ext{VRP}_{n,t} = \left(\frac{\text{VIX}_t}{100}\right)^2 - \text{RealizedVariance}_{n,t}
$$

For the 21d baseline used in this module:

$$
	ext{VRP}_{21,t} = \left(\frac{\text{VIX}_t}{100}\right)^2 - \text{RVAR}_{21,t}
$$

Interpretation:
- High VRP: options market prices more variance than recently realized.
- Low/negative VRP: realized variance is already elevated relative to implied variance.

VRP is a **feature/input**, not a target. The model asks whether today's feature set
(VIX, implied variance, realized variance, VRP and transformations, plus price/vol features)
helps predict future volatility behavior.
            """
        )

    with st.expander("How to read model quality metrics"):
        st.markdown(
            """
**AUC (Area Under the ROC Curve)**

AUC measures how well the model **ranks future volatility-expansion risk**. It is the area under the ROC curve, 
where the x-axis is the false positive rate and the y-axis is the true positive rate across all possible probability thresholds.

*What it measures:*
- Ranking quality: Does the model assign higher expansion probabilities to periods where volatility actually expanded?
- The ROC curve plots true positive rate (y-axis) vs false positive rate (x-axis) across all decision thresholds
- AUC = the area under that curve

*What it does NOT measure:*
- Probability calibration (whether predicted probabilities are numerically accurate)
- Profit or tradability
- Exact volatility magnitude
- Direction of stock price movement

*Interpretation:*
- **AUC = 0.50**: Random ranking (model cannot distinguish expansion days from non-expansion days)
- **AUC = 0.55–0.60**: Weak but marginally useful signal
- **AUC = 0.60–0.65**: Modest useful signal; should verify stability
- **AUC = 0.65+**: Decent to strong signal for financial data; prioritize robustness checks
- **AUC = 1.00**: Perfect ranking (all expansion cases receive higher probabilities than all non-expansion cases)

*Intuitive interpretation:*
If AUC = 0.68, it means: "If you randomly pick one actual expansion day and one actual non-expansion day, 
the model gives the expansion day a higher predicted probability about 68% of the time."

**Balanced accuracy**

- Handles class imbalance by averaging recall for both classes.
- Near `0.50` is close to random class assignment.

**ROC Curve**

- Visual representation of model ranking quality across all thresholds.
- Diagonal line (FPR = TPR) represents random guessing.
- Curves above diagonal indicate useful ranking signal.

**Calibration curve**

- Tests whether predicted probabilities are numerically trustworthy.
- Points near diagonal mean good probability calibration.

**Probability histogram and confusion matrix**

- Histogram should show separation between true expansion vs true non-expansion cases.
- Confusion matrix shows false negatives (missed expansions) vs false positives (false alarms).
            """
        )

    st.info(
        "This section evaluates whether VRP-style features add value for volatility-expansion classification. "
        "Use the AUC and balanced-accuracy rankings below to determine which feature group actually performs best "
        "for the selected target."
    )
    st.caption(
        "Note: the key metric here is classification quality (AUC / balanced accuracy), not linear correlation."
    )

    if vol_cls_m.empty:
        _missing_file_warning("volatility_classification_metrics.csv")
    else:
        target_vexp = st.selectbox(
            "Expansion target",
            ["next_21d_vol_expansion", "next_5d_vol_expansion"],
            key="vexp_target",
        )
        if target_vexp == "next_21d_vol_expansion":
            st.info(
                "21d expansion is usually the more stable signal. "
                "The 5d target is noisier and more sensitive to single shocks and event timing."
            )
        else:
            st.warning(
                "5d expansion is a high-noise target. "
                "Treat short-horizon improvements cautiously and confirm with longer out-of-sample history."
            )

        vc = vol_cls_m[vol_cls_m["target"] == target_vexp].copy() if "target" in vol_cls_m.columns else vol_cls_m.copy()
        vc_sorted = vc.sort_values("ROC_AUC", ascending=False) if "ROC_AUC" in vc.columns else vc

        # dynamic conclusion from data
        if not vc_sorted.empty and "ROC_AUC" in vc_sorted.columns:
            best_row = vc_sorted.iloc[0]
            best_auc = float(best_row["ROC_AUC"])
            best_ba = float(best_row["balanced_accuracy"]) if "balanced_accuracy" in best_row.index and pd.notna(best_row["balanced_accuracy"]) else float("nan")
            best_fg = best_row.get("feature_group", "?")
            best_mn = best_row.get("model_name", "?")
            ba_str = f"{best_ba:.3f}" if not np.isnan(best_ba) else "n/a"
            st.markdown(
                f"**For {target_vexp}, the best model ({best_fg}/{best_mn}) achieved "
                f"AUC = {best_auc:.3f} and balanced accuracy = {ba_str}.**"
            )

            if isinstance(best_fg, str):
                best_fg_l = best_fg.lower()
                if "vrp" in best_fg_l and "only" in best_fg_l:
                    st.success(
                        "Best feature group is VRP-only. This supports the thesis that relative options-implied "
                        "risk pricing is useful for volatility-regime classification."
                    )
                elif "price" in best_fg_l:
                    st.info(
                        "Best feature group is price-only. Under this sample, price dynamics may carry more "
                        "incremental information than VRP for this target."
                    )
                elif "combined" in best_fg_l:
                    st.info(
                        "Best feature group is combined. Verify this is robust out-of-sample and not feature "
                        "redundancy or overfitting."
                    )

            if best_auc < 0.55:
                st.error("Model signal is weak (AUC close to random). Treat this as non-actionable for risk decisions.")
            elif best_auc < 0.60:
                st.warning("Model signal is modest. Useful as a secondary filter, not as a standalone regime switch.")
            elif best_auc < 0.65:
                st.success("Model signal is useful but moderate. Focus on calibration and stability across regimes.")
            else:
                st.success("Model signal is relatively strong. Prioritize robustness checks and conservative deployment.")

        st.dataframe(vc_sorted, width="stretch")

        # AUC by feature group
        if "ROC_AUC" in vc.columns and "feature_group" in vc.columns:
            auc_summary = vc.groupby("feature_group", as_index=False)["ROC_AUC"].max().sort_values("ROC_AUC", ascending=False)
            fig_auc = px.bar(auc_summary, x="feature_group", y="ROC_AUC",
                             title=f"Best AUC by feature group - {target_vexp}", color="feature_group")
            fig_auc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="random baseline")
            st.plotly_chart(fig_auc, width="stretch")

        if "balanced_accuracy" in vc.columns and "feature_group" in vc.columns:
            ba_summary = vc.groupby("feature_group", as_index=False)["balanced_accuracy"].max()
            fig_ba = px.bar(ba_summary, x="feature_group", y="balanced_accuracy",
                            title=f"Best balanced accuracy - {target_vexp}", color="feature_group")
            fig_ba.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_ba, width="stretch")

        # Calibration - requires feature_group column
        if not vol_cal.empty:
            st.subheader("Calibration Curve")
            st.caption(
                "Interpretation: diagonal alignment means predicted probabilities are trustworthy; "
                "below diagonal indicates overestimation of expansion probability."
            )
            if "feature_group" not in vol_cal.columns:
                st.warning(
                    "Calibration file is missing `feature_group`. "
                    "Re-run `python scripts/run_volatility_experiment.py`."
                )
            else:
                cal_tgt = vol_cal[vol_cal["target"] == target_vexp]
                cal_fgs = sorted(cal_tgt["feature_group"].dropna().unique())
                cal_models = sorted(cal_tgt["model_name"].dropna().unique())
                sel_cal_fg = st.selectbox("Calibration feature group", cal_fgs, key="cal_fg")
                sel_cal_model = st.selectbox("Calibration model", cal_models, key="cal_model")
                cal_view = cal_tgt[
                    (cal_tgt["feature_group"] == sel_cal_fg) & (cal_tgt["model_name"] == sel_cal_model)
                ]
                if not cal_view.empty:
                    fig_cal = px.line(cal_view, x="avg_predicted_probability", y="actual_positive_rate",
                                      markers=True, title=f"Calibration - {target_vexp} / {sel_cal_fg} / {sel_cal_model}")
                    fig_cal.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                                      line=dict(dash="dash", color="gray"))
                    st.plotly_chart(fig_cal, width="stretch")
                    st.dataframe(cal_view, width="stretch")

        # ROC Curve
        st.subheader("ROC Curve")
        st.caption(
            "The ROC curve shows the trade-off between true positive rate (y-axis: expansions correctly identified) "
            "and false positive rate (x-axis: non-expansions incorrectly flagged). A curve above the diagonal indicates "
            "useful ranking signal. The area under this curve is AUC."
        )
        if not vol_cls_p.empty:
            pred_roc = vol_cls_p[vol_cls_p["target"] == target_vexp].copy() if "target" in vol_cls_p.columns else vol_cls_p.copy()
            if pred_roc.empty:
                st.warning(f"No prediction data available for ROC curve: {target_vexp}")
            elif "feature_group" not in pred_roc.columns or "model_name" not in pred_roc.columns:
                st.warning("Prediction file is missing `feature_group` or `model_name` columns for ROC.")
            else:
                roc_fgs = sorted(pred_roc["feature_group"].dropna().unique())
                if not roc_fgs:
                    st.warning(f"No feature groups available for ROC curve: {target_vexp}")
                else:
                    sel_roc_fg = st.selectbox("Feature group for ROC", roc_fgs, key="roc_fg")
                    pred_roc_fg = pred_roc[pred_roc["feature_group"] == sel_roc_fg].copy()
                    roc_models = sorted(pred_roc_fg["model_name"].dropna().unique())
                    if not roc_models:
                        st.warning(f"No models available for ROC: {sel_roc_fg} / {target_vexp}")
                    else:
                        sel_roc_model = st.selectbox("Model for ROC", roc_models, key="roc_model")
                        pred_roc_model = pred_roc_fg[pred_roc_fg["model_name"] == sel_roc_model].copy()
                        if pred_roc_model.empty:
                            st.warning(f"No prediction data for ROC: {target_vexp} / {sel_roc_fg} / {sel_roc_model}")
                        elif "y_pred_proba" not in pred_roc_model.columns or "y_true" not in pred_roc_model.columns:
                            st.warning("Prediction file is missing `y_pred_proba` or `y_true` columns.")
                        else:
                            try:
                                y_true = pd.to_numeric(pred_roc_model["y_true"], errors="coerce").astype(int)
                                y_pred_proba = pd.to_numeric(pred_roc_model["y_pred_proba"], errors="coerce").astype(float)
                                
                                # Drop NaN rows
                                valid_idx = ~(y_true.isna() | y_pred_proba.isna())
                                y_true_clean = y_true[valid_idx]
                                y_pred_proba_clean = y_pred_proba[valid_idx]
                                
                                if len(y_true_clean) < 2:
                                    st.warning(f"Insufficient valid data for ROC (need at least 2 rows, got {len(y_true_clean)}).")
                                elif len(y_true_clean.unique()) < 2:
                                    st.warning("Cannot compute ROC: only one class present in data.")
                                else:
                                    fpr, tpr, thresholds = roc_curve(y_true_clean, y_pred_proba_clean)
                                    roc_auc = roc_auc_score(y_true_clean, y_pred_proba_clean)
                                    
                                    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                                    fig_roc = px.line(roc_df, x="fpr", y="tpr",
                                                    title=f"ROC Curve (AUC = {roc_auc:.3f}) - {target_vexp} / {sel_roc_fg} / {sel_roc_model}")
                                    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                                                    line=dict(dash="dash", color="gray", width=2),
                                                    annotation_text="random classifier")
                                    fig_roc.update_xaxes(title_text="False Positive Rate")
                                    fig_roc.update_yaxes(title_text="True Positive Rate")
                                    st.plotly_chart(fig_roc, width="stretch")
                            except Exception as e:
                                st.error(f"Error computing ROC curve: {str(e)}")

        # Probability distribution + confusion matrix
        if not vol_cls_p.empty:
            st.caption(
                "A useful classifier should assign higher probabilities to actual expansion cases "
                "than to non-expansion cases."
            )
            pred_tgt = vol_cls_p[vol_cls_p["target"] == target_vexp].copy() if "target" in vol_cls_p.columns else vol_cls_p.copy()
            if pred_tgt.empty:
                st.warning(f"No prediction rows found for target `{target_vexp}`.")
            elif "feature_group" not in pred_tgt.columns or "model_name" not in pred_tgt.columns:
                st.warning("Prediction file is missing `feature_group` or `model_name` columns.")
            else:
                all_fgs = sorted(pred_tgt["feature_group"].dropna().unique())
                if not all_fgs:
                    st.warning(f"No feature groups available for target `{target_vexp}`.")
                else:
                    sel_fg_p = st.selectbox("Prediction feature group", all_fgs, key="pred_fg")
                    pred_fg = pred_tgt[pred_tgt["feature_group"] == sel_fg_p].copy()
                    all_models = sorted(pred_fg["model_name"].dropna().unique())
                    if not all_models:
                        st.warning(f"No models available for `{sel_fg_p}` and target `{target_vexp}`.")
                    else:
                        sel_model_p = st.selectbox("Prediction model", all_models, key="pred_model")
                        pp = pred_fg[pred_fg["model_name"] == sel_model_p].copy()
                        if pp.empty:
                            st.warning(
                                f"No rows available for target `{target_vexp}`, feature group `{sel_fg_p}`, and model `{sel_model_p}`."
                            )
                        else:
                            pp["class_label"] = pp["y_true"].astype(str).map({"0.0": "No expansion", "1.0": "Expansion", "0": "No expansion", "1": "Expansion"})
                            st.plotly_chart(
                                px.histogram(pp, x="y_pred_proba", color="class_label", barmode="overlay",
                                             title="Predicted probability distribution by actual class", nbins=20),
                                width="stretch",
                            )
                            if "y_pred_class" in pp.columns:
                                cm = confusion_matrix(pp["y_true"].astype(int), pp["y_pred_class"].astype(int))
                                cm_df = pd.DataFrame(cm, index=["Actual: no expansion", "Actual: expansion"],
                                                     columns=["Pred: no expansion", "Pred: expansion"])
                                st.subheader("Confusion Matrix")
                                st.dataframe(cm_df, width="stretch")

                                try:
                                    fn = int(cm[1, 0])
                                    fp = int(cm[0, 1])
                                    if fn > fp:
                                        st.warning(
                                            "False negatives exceed false positives. The model misses expansion events more often "
                                            "than it issues false alarms; this is riskier for exposure management."
                                        )
                                    else:
                                        st.info(
                                            "False positives are at least as frequent as false negatives. This is more conservative "
                                            "for risk controls, but may reduce opportunity capture."
                                        )
                                except Exception:
                                    pass

        st.divider()
        st.markdown(
            """
**Bottom line for this tab**

- A strong result here means the model can rank future expansion regimes better than random.
- It does **not** imply direct SPY direction prediction.
- It does **not** imply immediate tradability as a standalone strategy.
- The most practical use is risk management: position sizing, options-structure selection, and regime-aware exposure control.
            """
        )


# TAB 7 - Decile & Regime Analysis
with tabs[6]:
    st.subheader("Decile & Regime Analysis")

    if vol_decile.empty:
        _missing_file_warning("volatility_decile_study.csv")
    else:
        for decile_type, label, interpretation, sentiment in [
            (
                "vix_decile",
                "VIX Decile Study",
                "[WARN] **Benchmark / sanity check only.** High VIX -> high future volatility is expected "
                "because VIX is already an options-implied volatility measure. This is NOT a novel edge.",
                "warning",
            ),
            (
                "vrp_decile",
                "VRP Decile Study",
                "[OK] **More informative.** VRP deciles measure implied volatility relative to recent "
                "realized volatility. High VRP signals that options are pricing in more vol than recently "
                "observed. This is more meaningful for regime analysis.",
                "success",
            ),
            (
                "rv21_decile",
                "RV21 Decile Study",
                "[INFO] Current realized vol predicts future vol level, but high current vol does not "
                "necessarily predict further expansion.",
                "info",
            ),
        ]:
            st.subheader(label)
            getattr(st, sentiment)(interpretation)
            d = vol_decile[vol_decile["decile_type"] == decile_type].copy()
            if d.empty:
                st.caption("No data for this decile type.")
                continue

            col_21 = "avg_next_21d_vol"
            col_exp = "vol_expansion_rate_21d"
            if col_21 in d.columns:
                fig_v = px.line(d, x="decile", y=col_21, markers=True,
                                title=f"{label}: avg next-21d realized vol by decile (decimal)")
                st.plotly_chart(fig_v, width="stretch")
            if col_exp in d.columns:
                fig_e = px.line(d, x="decile", y=col_exp, markers=True,
                                title=f"{label}: 21d vol expansion rate by decile")
                st.plotly_chart(fig_e, width="stretch")
            st.dataframe(d, width="stretch")

        # High vs mid VRP
        vrp_d = vol_decile[vol_decile["decile_type"] == "vrp_decile"].copy()
        if not vrp_d.empty and "vol_expansion_rate_21d" in vrp_d.columns:
            top9 = vrp_d[vrp_d["decile"] == 9]
            mid = vrp_d[vrp_d["decile"].between(3, 6)]
            if not top9.empty and not mid.empty:
                st.subheader("VRP Decile 9 vs Middle Deciles (3-6)")
                comp = pd.DataFrame({
                    "group": ["High VRP (decile 9)", "Mid VRP (decile 3-6)"],
                    "avg_next_21d_vol_%": [
                        round(float(top9["avg_next_21d_vol"].mean()) * 100, 2),
                        round(float(mid["avg_next_21d_vol"].mean()) * 100, 2),
                    ],
                    "vol_expansion_rate_21d": [
                        round(float(top9["vol_expansion_rate_21d"].mean()), 4),
                        round(float(mid["vol_expansion_rate_21d"].mean()), 4),
                    ],
                })
                st.dataframe(comp, width="stretch")


# TAB 8 - Feature Importance
with tabs[7]:
    st.subheader("Feature Importance")
    st.warning(
        "Feature importance is not causal evidence. "
        "It shows model usage, not proof that a feature causes future volatility."
    )

    if vol_fi.empty:
        _missing_file_warning("volatility_feature_importance.csv")
    else:
        fi_targets = sorted(vol_fi["target"].dropna().unique()) if "target" in vol_fi.columns else []
        fi_models = sorted(vol_fi["model_name"].dropna().unique()) if "model_name" in vol_fi.columns else []
        sel_fi_target = st.selectbox("Target", fi_targets, key="fi_target")
        sel_fi_model = st.selectbox("Model", fi_models, key="fi_model")

        fi_view = vol_fi[(vol_fi["target"] == sel_fi_target) & (vol_fi["model_name"] == sel_fi_model)].copy()
        if fi_view.empty:
            fi_view = vol_fi[vol_fi["target"] == sel_fi_target].copy()

        if not fi_view.empty and "mean_importance" in fi_view.columns:
            top10 = fi_view.sort_values("mean_importance", ascending=False).head(10)
            err_x = "std_importance" if "std_importance" in top10.columns else None
            fig_fi = px.bar(top10, x="mean_importance", y="feature", orientation="h",
                            title=f"Top 10 features - {sel_fi_target} / {sel_fi_model}",
                            error_x=err_x, color="feature")
            st.plotly_chart(fig_fi, width="stretch")
            st.dataframe(fi_view.sort_values("mean_importance", ascending=False), width="stretch")

    if not reg_coef.empty and "model_family" in reg_coef.columns:
        st.subheader("Return Model Coefficients")
        fam = sorted(reg_coef["model_family"].dropna().unique())
        sel_fam = st.selectbox("Return regression model family", fam, key="ret_coef_fam")
        coef_view = reg_coef[reg_coef["model_family"] == sel_fam]
        if not coef_view.empty and "coef_name" in coef_view.columns:
            agg_coef = coef_view.groupby("coef_name", as_index=False)["coef_value"].mean()
            agg_coef["abs_coef"] = agg_coef["coef_value"].abs()
            agg_coef = agg_coef.sort_values("abs_coef", ascending=False).head(15)
            st.plotly_chart(
                px.bar(agg_coef, x="coef_value", y="coef_name", orientation="h",
                       title="Average coefficient magnitude (return model)"),
                width="stretch",
            )


# TAB 9 - Critical Conclusions
with tabs[8]:
    st.subheader("Critical Conclusions")
    st.markdown(
        """
**The project does not currently support a strong SPY directional trading strategy.**
The strongest supported result is that options-implied data improves volatility forecasting,
especially volatility-expansion prediction. VIX is the benchmark for future volatility level.
VRP provides more meaningful incremental information for volatility-regime change
because it compares implied volatility with recently realized volatility.
        """
    )

    items = [
        ("1. Return prediction", "#d32f2f", "Options data does not currently provide strong SPY direction prediction."),
        ("2. Trading strategy", "#d32f2f", "The current trading strategy is not good enough to trade."),
        ("3. VIX as benchmark (not novel)", "#6a1b9a",
         "VIX is a useful but obvious benchmark for future volatility level. "
         "High VIX -> high future vol is not a novel edge because VIX is itself an implied-vol measure."),
        ("4. VRP incremental value", "#388e3c",
         "VRP has stronger incremental value for predicting volatility expansion - "
         "this is the most meaningful finding."),
        ("5. Best result", "#388e3c",
         "The best result is 21-day volatility expansion prediction using VRP-only models."),
        ("6. VIX decile - benchmark only", "#f57c00",
         "High VIX predicting high future vol is expected. Do not overstate this finding."),
        ("7. VRP vs VIX", "#388e3c",
         "High VRP means implied vol is elevated relative to recent realized vol. "
         "This is more informative for volatility-regime change."),
        ("8. Combined models caution", "#f57c00",
         "Combined models are not automatically better. Simple models are more interpretable "
         "and often stronger out-of-sample. Check whether combined beats simple before claiming superiority."),
        ("9. Validation needed", "#f57c00",
         "Validate on: longer history, SPX option-chain IV (not only VIX proxy), "
         "different market regimes, future out-of-sample data."),
        ("10. Best next research direction", "#1565c0",
         "Use VRP as a volatility-regime classifier. Evaluate risk-management overlays "
         "(position sizing). Avoid treating VRP as a standalone SPY buy/sell signal."),
    ]

    for label, colour, text in items:
        st.markdown(
            f'<div style="border-left:4px solid {colour};padding:8px 14px;margin:6px 0;background:#fafafa;">'
            f"<strong>{label}</strong><br>{text}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Dynamically Computed Verdicts")
    for k, v in conclusions.items():
        try:
            display = f"{float(v):.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
        except Exception:
            display = str(v)
        st.markdown(f"- **{k}**: {display}")

    st.divider()
    st.info(
        "**Best supported conclusion:** "
        "Options-implied data adds useful information for volatility forecasting, "
        "especially volatility-expansion prediction. "
        "Evidence for direct return prediction is weak."
    )

st.caption(
    "Statistical predictability, economic tradability, post-cost performance, and robustness "
    "across regimes are distinct questions. AUC measures ranking quality, not accuracy. "
    "VIX decile analysis is a benchmark / sanity check, not alpha."
)


# TAB 10 - Options Pricing & Greeks
with tabs[9]:
    render_options_pricing_dashboard()

