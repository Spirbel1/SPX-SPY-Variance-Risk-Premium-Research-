# spx_spy_vrp_research

A reproducible Python research app to test whether variance risk premium (VRP) signals improve prediction of future SPX/SPY returns and volatility.

## Research Question

Does an options-implied variance proxy (MVP: VIX from FRED) add predictive value for future SPX/SPY returns beyond price-only and realized-volatility baselines?

## Data Sources

- FRED `VIXCLS` (implied volatility proxy)
- FRED `SP500` (SPX close proxy)
- Yahoo Finance `SPY` OHLCV (optional)

## Project Layout

- `data/raw/`: cached raw downloads
- `data/processed/vrp_dataset.parquet`: engineered dataset
- `src/`: loaders, features, models, validation, backtest, charts
- `scripts/run_pipeline.py`: build the dataset and run walk-forward modeling
- `scripts/run_backtest.py`: run signal backtest and save outputs
- `app/streamlit_app.py`: interactive dashboard
- `tests/`: unit tests for VRP and target shifting

## Installation

1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: copy `.env.example` to `.env` and set API keys/settings.

## Run Pipeline

```bash
python scripts/run_pipeline.py
```

This command downloads/caches data, builds features/targets, validates lookahead safety, saves:
- `data/processed/vrp_dataset.parquet`
- walk-forward model outputs in `data/processed/`

## Run Backtest

```bash
python scripts/run_backtest.py
```

This command builds a weekly-signal strategy and saves:
- `data/processed/backtest_timeseries.csv`
- `data/processed/backtest_summary.csv`
- `data/processed/trade_log.csv`

## Volatility Prediction Experiment

This experiment evaluates volatility forecasting, not direct return prediction. A useful volatility model may help with risk management, options strategy selection, trade filtering, or position sizing, even if it does not produce a profitable standalone SPY trading strategy.

### Goal

Test whether options-implied volatility and the variance risk premium help forecast future realized volatility and volatility expansion better than realized-volatility-only baselines.

### Targets

- `next_5d_realized_volatility`
- `next_21d_realized_volatility`
- `next_5d_vol_expansion`
- `next_21d_vol_expansion`

### Feature Groups

- Historical realized-volatility baseline
- VIX / implied-volatility baseline
- VRP-only features
- Extreme-VRP regime features
- Combined model features

### Models and Metrics

The experiment runs walk-forward regression and classification models, plus direct baselines such as VIX, RV21, and an average VIX/RV21 forecast. It reports standard regression metrics, classification metrics, calibration, feature importance, and decile studies.

### How to Run

```bash
python scripts/run_volatility_experiment.py
```

### How to Interpret Results

- VIX is expected to be strong for predicting future volatility level.
- VRP is expected to be more useful for volatility expansion and regime classification.
- Raw VRP may not have a linear relationship with future volatility.
- Extreme VRP values may matter more than average VRP values.

The outputs are written to `data/processed/`:
- `volatility_regression_metrics.csv`
- `volatility_regression_predictions.csv`
- `volatility_classification_metrics.csv`
- `volatility_classification_predictions.csv`
- `volatility_calibration.csv`
- `volatility_feature_importance.csv`
- `volatility_decile_study.csv`

## Interpretation Guidance

Do not claim directional predictability unless out-of-sample evidence supports it. Keep these separate:
- Statistical predictability
- Economic tradability
- Performance after transaction costs
- Robustness across years and regimes

This experiment evaluates volatility forecasting, not direct return prediction. A useful volatility model may help with risk management, options strategy selection, trade filtering, or position sizing, even if it does not produce a profitable standalone SPY trading strategy.

## Launch Streamlit

```bash
streamlit run app/streamlit_app.py
```

Dashboard includes tabs for data overview, VRP features, regressions, classification, backtest results, and robustness checks.

## Options Pricing & Greeks Add-on

The dashboard includes a tab called **Options Pricing & Greeks** for daily SPY options pricing analytics.

### What It Does

- Uses Databento daily options data (no intraday, no tick, no 0DTE).
- Downloads raw daily data only when you click **Download / Update Databento Cache**.
- Saves raw and processed data locally as Parquet.
- Loads local cache on later runs without re-downloading.
- Computes implied volatility, Black-Scholes-Merton theoretical price, and Greeks.
- Compares market price vs theoretical price under different volatility modes.

### API Key Setup

Set Databento API key with either method:

1. Environment variable:

```bash
set DATABENTO_API_KEY=your_key_here
```

2. Streamlit secrets:

```toml
# .streamlit/secrets.toml
DATABENTO_API_KEY = "your_key_here"
```

The app never hard-codes or prints your key.

### Cache-First Workflow

1. Open the **Options Pricing & Greeks** tab.
2. Choose a date range (default 30 days, warning above 30, blocked above 90 unless explicitly confirmed).
3. Click **Download / Update Databento Cache** once.
4. On future runs, click **Load from cache** to use local files only.

Local cache paths:

- `data/raw/databento/options/`
- `data/processed/options_pricing/`
- `data/cache_manifest/cache_manifest.json`

This module is designed so widget changes and chart interactions do not trigger Databento API calls.

### Cost Controls (Current Defaults)

To reduce Databento usage cost, the downloader now limits contracts before requesting daily OHLCV:

- DTE window: `7` to `45`
- Moneyness band: `0.90` to `1.10`
- Max contract symbols: `1500`

This keeps the first implementation focused on near-ATM, near-term contracts while preserving daily-only behavior.

## MVP Limitation

VIX is an index-level implied volatility proxy, not a full model-free implied variance from a full SPX option chain. It is useful for MVP research but can diverge from option-chain-based variance estimates.

## Upgrade Path to Paid Option-Chain Data

Use `src/data_loaders/csv_options_loader.py` adapters to ingest standardized option-chain CSV/API extracts from ORATS, Cboe DataShop, ThetaData, OptionMetrics, or Databento.

The file includes placeholders/utilities for:
- ATM 30-day IV
- 25-delta put and call IV
- 25-delta risk reversal and skew
- 30d-60d term structure
- chain-based implied variance approximation

## Interpreting the Research Dashboard

> Do not judge the project only by trading returns.

1. **Return prediction and volatility prediction are separate tasks.** A model that predicts volatility well does not necessarily predict direction.
2. **VIX is the benchmark for volatility-level forecasting.** VIX is derived from options-implied volatility, so it naturally forecasts future realized volatility. This is a *sanity check*, not a novel finding.
3. **VRP is the main signal for volatility-expansion forecasting.** VRP measures the *gap* between implied and recently realized volatility. When this gap is large, the market is pricing in more risk than recently observed — this is more informative for regime detection.
4. **AUC measures ranking quality, not accuracy.** An AUC of 0.60 means the model ranks days correctly 60% of the time, not that it is 60% accurate.
5. **VIX decile analysis is expected and should not be overstated.** "High VIX → high future vol" is well-known and trivially true. It confirms data correctness; it is not alpha.
6. **The strongest current conclusion is about volatility-regime classification, not SPY direction.** The best supported result is that VRP-only features help predict 21-day volatility expansion.
7. **Combined models are not automatically better.** Feature redundancy and overfitting can make combined models worse out-of-sample than simple VIX-only or VRP-only baselines.
8. **Main interpretation:**
   > The project does not currently support a strong SPY directional trading strategy.
   > The strongest supported result is that options-implied data improves volatility forecasting,
   > especially volatility-expansion prediction. VIX is the benchmark for future volatility level.
   > VRP provides more meaningful incremental information for volatility-regime change
   > because it compares implied volatility with recently realized volatility.

## Testing

```bash
pytest -q
```
