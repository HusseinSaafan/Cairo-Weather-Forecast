# Cairo Weather Forecast

An end-to-end time-series forecasting project that prepares daily Cairo weather data, runs exploratory analysis and statistical tests, trains multiple forecasting models, and automatically selects a champion model based on RMSE.

## Disclaimer

This project is for learning and portfolio demonstration.

It is not intended for production weather operations or safety-critical forecasting.

The focus is on practical time-series workflow design: data preparation, EDA, model training, model comparison, and reproducible experimentation.

## Project Objective

- Build a clean forecasting pipeline from raw weather data to model selection
- Perform time-series EDA (trend, seasonality, autocorrelation)
- Train and evaluate multiple forecasting approaches
- Compare models using MSE, MAE, and RMSE
- Persist a champion model artifact for downstream use

## Dataset Overview

Raw dataset path:
- `database/raw/Cairo-Weather.csv`

Forecasting target used in the pipeline:
- `temperature_2m_mean (°C)`

Time column:
- `time`

Split strategy used by preprocessing:
- Train: all rows before the final 365 days
- Test: last 365 days
- Validation: not used

## Project Structure

```text
Cairo-Weather-Forecast/
├── src/
│   ├── main.py                                # End-to-end pipeline entry point
│   ├── ingestion_preprocessing/
│   │   └── clean_split.py                     # Clean raw data and create train/test
│   ├── eda/
│   │   ├── initial_eda.py                     # Initial descriptive EDA and plots
│   │   ├── eda.py                             # ACF/PACF on original series
│   │   ├── eda_log.py                         # ACF/PACF on log-transformed series
│   │   ├── decomp.py                          # Additive vs multiplicative decomposition
│   │   ├── decomp_log.py                      # Additive decomposition on log(target)
│   │   ├── test.py                            # Ljung-Box + ADF + decomposition residual tests
│   │   └── test_log.py                        # Ljung-Box + ADF on log-transformed series
│   ├── modeling/
│   │   ├── sarima.py                          # SARIMA baseline on original target
│   │   ├── sarima_log.py                      # SARIMA on log(target)
│   │   ├── man_sarima.py                      # Manual seasonal differencing + ARIMA
│   │   ├── man_sarima_log.py                  # Manual differencing on log(target)
│   │   ├── auto_arima.py                      # Auto-ARIMA on original target
│   │   ├── auto_arima_log.py                  # Auto-ARIMA on log(target)
│   │   ├── prophet.py                         # Prophet model with CV-based tuning
│   │   └── champ_model.py                     # Compare metrics and save champion model
│   └── utils/
│       ├── config.py                          # Shared logger configuration
│       └── helpers.py                         # Shared data-loading utilities
├── database/
│   ├── raw/
│   │   └── Cairo-Weather.csv                  # Raw source data
│   ├── train.csv                              # Generated train split
│   └── test.csv                               # Generated test split (last 365 days)
├── artifacts/
│   ├── *_metrics.json                         # Per-model evaluation metrics
│   ├── model_comparison.json                  # Full leaderboard + champion details
│   └── champ.pkl                              # Champion model artifact
├── figures/
│   ├── eda/                                   # EDA figures
│   ├── modeling/                              # Model diagnostics
│   └── model_eval/                            # Actual vs predicted and comparison charts
├── logs/                                      # Timestamped logs and summaries
├── requirements.txt
└── README.md
```

## Pipeline Workflow

### 1. Data Cleaning and Split

- Load raw weather data
- Keep only `time` and `temperature_2m_mean (°C)`
- Sort by time and handle missing target values
- Create train/test split where test is the most recent 365 days

### 2. Exploratory Data Analysis (EDA)

- Initial descriptive statistics and distribution analysis
- Rolling statistics and outlier views
- ACF/PACF plots on original and log-transformed target
- Seasonal decomposition on original and log scales
- Statistical tests (Ljung-Box and ADF)

### 3. Model Training

The project trains multiple forecasting models:

- SARIMA
- SARIMA (log target)
- Manual seasonal differencing + ARIMA
- Manual seasonal differencing + ARIMA (log target)
- Auto-ARIMA
- Auto-ARIMA (log target)
- Prophet

Each model writes:

- Model artifact (`.pkl`)
- Metrics JSON (`mse`, `mae`, `rmse`)
- Diagnostic/evaluation plots
- Run logs

### 4. Champion Selection

`src/modeling/champ_model.py`:

- Reads all `*_metrics.json` files in `artifacts/`
- Sorts models by RMSE (lower is better)
- Saves the best artifact as `artifacts/champ.pkl`
- Writes `artifacts/model_comparison.json`

## Results Summary

Based on the current `artifacts/model_comparison.json`:

- Champion model: Prophet
- Champion RMSE: 2.1583
- Champion MAE: 1.6456
- Champion MSE: 4.6581

Top leaderboard snapshot (sorted by RMSE):

1. Prophet (RMSE 2.1583)
2. Manual SARIMA (RMSE 3.0325)
3. Manual SARIMA Log (RMSE 3.0385)

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python -m src.main
```

This executes ingestion/preprocessing, all EDA scripts, all modeling scripts, and champion selection in sequence.

## Tech Stack

- pandas, numpy: data manipulation
- matplotlib: visualization
- statsmodels: SARIMA/ARIMA and statistical tests
- scipy: signal/statistical utilities
- scikit-learn: regression error metrics
- pmdarima: Auto-ARIMA
- prophet: forecasting model and CV diagnostics

## Skills Demonstrated

- Time-series preprocessing and splitting strategy design
- Seasonality/trend analysis and decomposition
- Stationarity/autocorrelation diagnostics
- Multi-model forecasting and comparison
- Metric-driven champion model selection
- Modular Python pipeline design and logging
- Reproducible dependency management

## Notes

- Dependencies are pinned in `requirements.txt`.
- The project was tested with Python 3.9.6.
- `.pkl` artifacts are ignored except `artifacts/champ.pkl`.
