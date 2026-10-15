# Walmart sales experiment

The configured source is [aslanahmedov/walmart-sales-forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast), version 1. The training file has 421,570 store/department observations on 143 weekly dates from February 5, 2010 through October 26, 2012.

```bash
uv run python scripts/prepare_data.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima evaluation.cv.enabled=true
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs -m sweep=arima_orders
```

Run from the repository root. Acquisition is selected by `data.source`; the forecasting loader is selected by `data.loader`. Walmart column names, store/department filters, and aggregation rules are configuration values.

The scripts have no Walmart default; `--config-dir` explicitly selects this experiment. To start another dataset, copy the independent [example experiment](../example/README.md). It remains usable if this Walmart directory is removed.

`sweep=arima_orders` varies both `order` and `seasonal_order`: three choices crossed with two choices produce six jobs. `model/arima.yaml` supplies constructor defaults, and the sweep overrides both orders. It is not restricted to a nonseasonal model. See the template for a four-job example and a second sweep that selects explicit order pairs.

The default sums sales across all stores/departments, reserves the final 20% of weeks, and optionally runs three eight-week CV folds inside preceding history. The holiday flag is a future-known calendar feature. The explicit seasonal model uses a 52-week cycle; inspect convergence diagnostics when training windows contain few cycles.

The run's `plots/cv.png` shows all folds together with their preceding history and training cutoffs. `plots/holdout.png` shows training history, held-out actuals, and the forecast. Plots use the saved `series.csv` and `predictions.csv`.

`scripts/eda.py` explores the full configured weekly series and writes tables and plots under `results/eda`. It includes series history, trailing statistics, target distributions, differencing, ACF/PACF, robust STL with period 52, seasonal profiles, and ADF/KPSS diagnostics. Override the `eda` settings or data filters in the command or config; rerunning replaces the current EDA output.

Model and sweep manifests live in `configs/model` and `configs/sweep`. Consolidated metric reports and comparison PNGs live under `results`. Detailed run files live under `artifacts/runs/<run-name>`. Rerunning a configuration replaces its previous files; distinct sweep configurations remain separate. Sweep preset names select jobs and do not appear as result groups. These generated outputs are ignored by Git and can be discarded while tinkering.
