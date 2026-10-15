# ForgeML

ForgeML runs YAML-configured experiments with task-specific data loading, models, and evaluation. The current implementation supports univariate forecasting with optional known future features: naive, ARIMA, seasonal ARIMA, ARIMA with regressors, and Auto-ARIMA.

## Setup and run

From the repository root, using Python 3.12:

```bash
uv sync
uv run python scripts/prepare_data.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima evaluation.cv.enabled=true
```

The Walmart example uses the real [Walmart Sales Forecast dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast). Its source handle, selected files, CSV columns, and aggregation rules live in `experimentation/walmart_sales/configs/config.yaml`.

`prepare_data.py` materializes the configured raw source. EDA and training read local data and do not download implicitly. There is no dataset built into the scripts: select an experiment with `--config-dir experimentation/<name>/configs`. Deleting the Walmart directory does not affect other experiments.

## Start a different experiment

Copy the self-contained [example experiment](experimentation/example/README.md):

```bash
cp -R experimentation/example experimentation/my_forecast
mkdir -p experimentation/my_forecast/data
```

Set `experiment.name: my_forecast` in its main config, provide your real CSV, and set its column names, frequency, evaluation sizes, and seasonal periods. The template assumes daily observations with a seven-day cycle; adapt those settings to your data. It contains no dummy dataset and has its own model and sweep manifests.

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs --cfg job --resolve
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs model=arima
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs -m sweep=arima_orders
```

The first command only prints the resolved config. The example ARIMA sweep crosses two `order` values with two `seasonal_order` values, producing four jobs. `sweep=arima_pairs` demonstrates two explicitly paired choices instead. The template README explains each merge step and all terminal commands.

## Responsibilities and extension boundaries

```text
src/forgeml/
├── data/
│   └── sources.py                   # Shared raw acquisition: local files or Kaggle
└── tasks/
    ├── registry.py                  # Dispatch to the selected task pipeline
    └── forecasting/
        ├── factory.py               # Construct models from YAML manifests
        ├── pipeline.py              # Forecasting experiment orchestration
        ├── dataloaders/
        │   ├── base.py              # ForecastData and BaseForecastDataLoader
        │   ├── csv.py               # CSV decoding and forecasting preparation
        │   └── factory.py           # Select the configured forecasting loader
        ├── eda/
        │   ├── pipeline.py          # Explore the configured target series
        │   ├── diagnostics.py       # STL, correlations, stationarity, summaries
        │   └── plots.py             # Series and diagnostic PNGs
        ├── models/
        │   ├── base.py              # BaseForecaster: fit, predict, describe_fit
        │   ├── inputs.py            # Important shared model input checks
        │   ├── naive.py
        │   ├── arima.py
        │   └── auto_arima.py
        └── evaluation/
            ├── metrics.py
            ├── plots.py
            ├── reporting.py
            ├── results.py
            └── splitting.py
```

Raw acquisition copies bytes and is independent of ML task. The forecasting CSV loader prepares a chronological target and aligned features; it has no Walmart-specific column names. Its factory converts OmegaConf settings to ordinary Python options before constructing the loader.

Each future task should own its data contract, loaders, model interface, and pipeline. Add its pipeline, report, and EDA entry points to `tasks/registry.py`; that registry does not impose a universal trainer or forecasting data shapes on classification or vision. Only the forecasting task is implemented now.

The forecasting pipeline calls `BaseForecaster.fit(history, exog)` and `predict(horizon, future_exog)` for every model. A future Transformer adapter can construct training windows, fit scaling, create batches, and train its network inside `fit`. Those operations receive only that fold's training history. It must return a one-dimensional forecast array from `predict`; the pipeline's split and evaluation logic then stays the same. PyTorch and Transformer training are not implemented yet.

## Configure data

The Walmart manifest declares:

```yaml
task: forecasting

data:
  loader: csv
  source:
    provider: kaggle
    handle: aslanahmedov/walmart-sales-forecast/versions/1
    directory: experimentation/walmart_sales/data
    files: [train.csv]
  path: ${data.source.directory}/train.csv
  date_column: Date
  target_column: Weekly_Sales
  frequency: W-FRI
  filters: {}
  known_future_columns: [IsHoliday]
  aggregate:
    Weekly_Sales: sum
    IsHoliday: max
```

The Walmart config forecasts total weekly sales across all stores/departments. Use `filters: {Store: 1, Dept: 1}` in YAML to select one series, or:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs +data.filters.Store=1 +data.filters.Dept=1
```

For another local CSV, configure `source.provider: local`, its directory, and the appropriate CSV fields. Aggregation can be null when every date already represents one observation. For a different input format, implement `BaseForecastDataLoader.load()`, return `ForecastData`, register the loader, and select it through `data.loader`.

CSV preparation sorts dates, checks missing periods, and performs only explicit deterministic filtering/aggregation. Learned transformations and model-specific batching belong inside training on each fold. `known_future_columns` must contain features actually available at the forecast origin, such as calendar flags.

## Exploratory data analysis

```bash
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs +data.filters.Store=1 +data.filters.Dept=1
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs eda.max_lags=30 eda.rolling_window=8
```

EDA explores the target produced by the configured forecasting loader, with the same filters and aggregation as training. It runs independently of model fitting and saves one current report under `results/eda`. Repeating the command replaces its managed files without adding attempts or model runs. For another experiment, pass its `--config-dir` just as with training.

The main config contains its analysis settings:

```yaml
eda:
  directory: ${results.directory}/eda
  seasonal_period: 52
  stl_robust: true
  max_lags: 60
  rolling_window: 13
  difference_lag: 1
  start_date: null
  end_date: null
```

The default analyzes the whole prepared series. Optional `start_date` and `end_date` restrict the analysis without changing training. Configure `seasonal_period` for the dataset, such as 12 for an annual cycle in monthly observations; null disables STL and seasonal differencing. The Walmart period of 52 is an approximate annual cycle in weekly observations. Cycle positions start at the first selected date rather than calendar January 1.

Generated plots are:

- `series.png`: the complete selected history and points beyond 1.5 × IQR.
- `rolling.png`: trailing mean, standard deviation, and mean ± one standard deviation.
- `distribution.png`: target histogram and boxplot.
- `differences.png`: differences at the configured lag and seasonal lag.
- `acf.png` and `pacf.png`: original, differenced, seasonally differenced, and available STL residual series, with 95% reference bands.
- `stl.png`: observed values, trend, seasonality, and residuals from additive [statsmodels STL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html).
- `seasonality.png`: calendar month distributions and overlaid configured cycles.

Plots live in `results/eda/plots`. The accompanying `series.csv`, `rolling.csv`, `differences.csv`, `autocorrelation.csv`, `outliers.csv`, and optional `stl.csv`/`seasonality.csv` contain the plotted numbers. `summary.json` and `summary.csv` describe the time span, frequency, distribution, and analysis settings; `config.yaml` saves the resolved configuration.

`stationarity.csv` contains ADF/KPSS statistics, p-values, lags, critical values, and diagnostic notes. Their null hypotheses have opposite meanings: ADF tests a unit root; KPSS tests level stationarity with a constant. The hypotheses are included in each row; these diagnostics do not automatically choose a model. See the [statsmodels ADF/KPSS guide](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html).

Lag counts adapt to each series' length. Constant or short series produce explicit notes instead of stopping other analyses. This pipeline conservatively requires two seasonal cycles and regular dates for STL; when unavailable, `stl.png` displays the reason and there is no `stl.csv`. IQR flags never remove observations. EDA uses the prepared-data contract, so malformed values or missing periods rejected by the loader still need to be addressed in the input data.

## Evaluation and plots

Holdout can use a fraction or an observation count. CV uses expanding training windows configured by `n_splits`, `horizon`, `step`, and `min_train_size`.

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs evaluation.holdout.test_size=26
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs evaluation.holdout.enabled=false evaluation.cv.enabled=true
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs evaluation.cv.enabled=true
```

When both are enabled, the final holdout is reserved first and CV runs entirely inside development history. Every window fits a fresh model. Pooled CV scores include every validation forecast, including repeated dates when windows overlap. Choose model settings using CV, then assess the selected configuration on the holdout.

Each run saves the prepared observations in `series.csv` and forecasts in `predictions.csv`. Its plots include:

- `plots/cv.png`: all completed CV folds on one timeline, with the observed history, a different forecast color per fold, colored training cutoffs, and shaded forecast periods. The timeline ends at the last CV validation date, excluding a later final holdout.
- `plots/holdout.png`: the training history leading into the held-out observations and forecast, with the training cutoff marked.

Each CV fold is trained on the observations from the beginning of the displayed history through its own cutoff. The shaded initial history is the first fold's training period; later folds grow beyond it. Overlapping fold forecasts remain distinct lines.

Saved series and window boundaries let plots be rebuilt without reading the source dataset or retraining:

```bash
uv run python scripts/consolidate_results.py --config experimentation/walmart_sales/configs/config.yaml --refresh-plots
```

## Results layout and naming

```text
experimentation/walmart_sales/
├── configs/
│   ├── config.yaml
│   ├── model/
│   └── sweep/
├── data/                            # Configured raw files
├── results/
│   ├── mae.csv
│   ├── rmse.csv
│   ├── mape.csv
│   ├── smape.csv
│   ├── wape.csv
│   ├── runs.csv
│   ├── eda/                         # EDA tables, summary, config and plots
│   └── plots/                       # One metric comparison PNG per CSV
└── artifacts/
    ├── runs/<run-name>/
    │   ├── run.json
    │   ├── config.yaml
    │   ├── series.csv
    │   ├── metrics.json
    │   ├── metrics.csv
    │   ├── predictions.csv
    │   └── plots/
    │       ├── cv.png
    │       └── holdout.png
    └── sweeps/                      # Hydra launch manifests
```

Run folders describe model settings, filters, and evaluation, such as `arima__order-1-1-1__all-data__holdout-20pct__cv-3x8`. Here `cv-3x8` means three folds forecasting eight observations each. There are no study folders or attempt counters. You can also choose a name:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=naive run.name=last_value
```

Rerunning the same model, data configuration, and split settings replaces its entire previous artifact folder. Each configuration has one current result. Changing only the metric list, or updating data at the configured CSV path, refreshes that result too. A failed rerun replaces the previous success and cannot leave its old predictions or plots behind.

Different model parameters, data configurations, or split settings remain separate, even when launched by different sweep presets. If two distinct configurations have the same descriptive name, a stable `__config-<fingerprint>` suffix distinguishes them; their full settings are in `config.yaml`. The suffix identifies configuration differences, never execution count. Use distinct model labels or `run.name` values when you want explicit names for those variants.

Every selected metric has a consolidated CSV and comparison PNG. A score row represents one run's pooled CV result or holdout result; per-fold scores remain in the artifact. `runs.csv` indexes all runs, including failures. Only completed runs enter comparisons.

Compare rows sharing a `comparison_id`: it fingerprints the prepared target values and evaluated window boundaries. Different targets, filters that change the series, or evaluation windows stay distinguishable in CSVs and separate plot panels. Reports can be rebuilt from current artifacts with `scripts/consolidate_results.py`.

During development, generated data, artifacts, and reports are ignored by Git. You can remove the generated output folders and run experiments again. Saved artifact formats describe the current implementation.

## Models and sweeps

Model manifests contain a registry `name`, readable `label`, `use_exog`, and ordinary constructor `params`:

```yaml
# @package model
name: arima
label: arima
use_exog: false
params:
  order: [1, 1, 1]
  seasonal_order: [0, 0, 0, 0]
  trend: null
```

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=naive
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima/seasonal
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima/with_exog
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima/auto
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs model=arima 'model.params.order=[2,1,1]'
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs -m sweep=arima_orders
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs -m sweep=model_comparison
```

Auto-ARIMA uses pmdarima to select orders inside each training window. Search bounds live in `model.params.search`, and selected orders, AIC, convergence, and fit warnings are saved. Weekly Walmart data uses an approximate annual period of 52 in the explicit seasonal manifest.

The Walmart `arima_orders` preset crosses three `order` choices with two `seasonal_order` choices, producing six jobs. Its base is the general `model/arima.yaml` manifest; both parameters are passed to the same ARIMA implementation. `model_comparison` selects five complete model manifests. These filenames select jobs and do not create result groups or additional copies of an existing configuration.

To define your own sweep, copy `configs/sweep/arima_orders.yaml` to `configs/sweep/my_orders.yaml` and edit `hydra.sweeper.params`. For example:

```yaml
# @package _global_
defaults:
  - override /model: arima

evaluation:
  cv:
    enabled: true

hydra:
  sweeper:
    params:
      model.params.order: "[1,1,0],[2,1,0]"
      model.params.seasonal_order: "[0,0,0,0],[1,0,0,52]"
```

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs -m sweep=my_orders
```

This produces four combinations through Hydra's [standard basic sweeper](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/). Each combination keeps its latest result. `-m` expands choices into separate jobs; selecting the YAML alone loads shared settings for one run. CLI values can replace sweep choices. To request specific paired orders instead of every combination, use the dictionary-sweep pattern in [the template's arima_pairs.yaml](experimentation/example/configs/sweep/arima_pairs.yaml). Keep model-specific parameter sweeps separate from comparisons that include models with different constructor arguments.

## Hydra's role

Hydra is not required to read YAML. We use its composition, terminal overrides, and Cartesian grid launcher instead of maintaining those mechanisms ourselves. It composes the main config and model/sweep groups, applies overrides, and launches multirun jobs. `# @package model` puts nested ARIMA manifests directly under `cfg.model`; sweep files use `# @package _global_` to modify shared settings. OmegaConf resolves interpolations and factories convert settings into Python values.

The main defaults load a model, merge the main file at `_self_`, then optionally merge a selected sweep. `model=arima` changes the selected manifest; `'model.params.order=[2,1,0]'` overrides one value. `--config-dir` selects the experiment's config directory, and `--config-name` selects its main YAML filename without the extension. The scripts have no Walmart default. The [template guide](experimentation/example/README.md#how-hydra-builds-the-config) walks through the order and provides a four-job grid table.

The thin training script dispatches `cfg.task` and passes launch metadata to the task. The report CLI also dispatches to that task's reporter, which owns its artifact format. Concrete models and data loaders do not depend on Hydra. `hydra.job.chdir: false` keeps paths relative to the launch directory; ForgeML owns descriptive run artifacts and Hydra keeps launch manifests under `artifacts/sweeps`.

Inspect the composed config without fitting:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs sweep=arima_orders --cfg job --resolve
```

That prints the base job config, not every sweep combination. Inspect the selected sweep axes with:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs sweep=arima_orders --cfg hydra --package hydra.sweeper
```

MLflow is a separate tool for tracking runs and artifacts; it does not replace config composition or the grid launcher. This implementation currently saves CSV/JSON/PNG outputs and does not log to MLflow.

## Metrics and checks

Available metrics are MAE, MSE, RMSE, MAPE, SMAPE, and WAPE. Percentage metrics are reported as percentages. MAPE excludes zero actuals and is undefined when all actuals are zero. SMAPE treats a zero/zero pair as zero. WAPE is undefined when all actuals are zero. Undefined values are JSON null and blank CSV cells.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src scripts tests
```

Tests use small deterministic local CSVs and mocked acquisition to verify data contracts, metrics, leakage boundaries, plots, registry dispatch, and saved reports. Production experiments use the configured real dataset.
