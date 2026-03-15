# Vehicle Sale Price Prediction Project

This repository predicts vehicle sale prices (`Sold_Amount`) using the provided datasets:

- `DatiumTrain.rpt`
- `DatiumTest.rpt`

The workflow includes:

- Exploratory Data Analysis (EDA) with publication-quality plots
- Feature engineering and strict removal of disallowed features
- Training and comparison of multiple regression models
- Metrics tracking and experiment logging
- Diagnostics for model underperformance by segment

## 1) Environment Setup (Mac Air M4 Friendly)

The project uses standard Python libraries that run on Apple Silicon.

```bash
cd "/Users/mdrashidulislam/Desktop/Projects/Data Science Task"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note:
- If matplotlib cache permission warnings appear, run commands with `MPLCONFIGDIR=.cache/matplotlib`.

## 2) Run EDA

```bash
MPLCONFIGDIR=.cache/matplotlib python src/run_eda.py
```

Generated files:
- `outputs/eda/data_quality_summary.json`
- `outputs/eda/eda_dashboard.png`
- `outputs/eda/missingness_profile.png`
- `outputs/eda/segment_landscape.png`
- `outputs/eda/train_test_shift.png`
- `outputs/eda/correlation_focus.png`

## 3) Train and Compare Models

```bash
MPLCONFIGDIR=.cache/matplotlib python src/train_models.py
```

This will:
- Train 4 candidates (baseline + 3 predictive models)
- Use temporal holdout validation (80/20 split by `Sold_Date`)
- Use training-fit preprocessing decisions (rare-category grouping, outlier clipping, high-missing/high-cardinality removal)
- Log all experiments to:
  - `logs/experiments.jsonl`
  - `logs/experiments.csv`
- Save model artifacts and diagnostics:
  - `outputs/models/validation_results.csv`
  - `outputs/models/validation_results.md`
  - `outputs/models/best_model.joblib`
  - `outputs/models/run_summary.json`
  - `outputs/models/validation_best_model_residuals_vs_pred.png`
  - `outputs/models/validation_best_model_actual_vs_pred.png`
  - `outputs/models/segment_diagnostics_*.csv`
- Save predictions:
  - `outputs/predictions/test_predictions.csv`

## 4) Project Report

See:
- `reports/project_report.md`

The report documents:
- data quality concerns
- feature design decisions
- model experimentation and encoding strategy
- evaluation metrics and diagnostics
- final model choice and rationale

## 5) Important Constraint Enforced

These forbidden fields are explicitly excluded from training:

- `AvgWholesale`
- `AvgRetail`
- `GoodWholesale`
- `GoodRetail`
- `TradeMin`
- `TradeMax`
- `PrivateMax`
