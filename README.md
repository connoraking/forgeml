# ForgeML

Config-driven forecasting experiments with shared data preparation, chronological evaluation, EDA, and plots. Current model families are naive and ARIMA. ARIMA supports fixed orders or automatic order selection, seasonal terms, and future-known regressors.

## Run an experiment

From the repository root, with Python 3.12:

```bash
uv sync
uv run python scripts/prepare_data.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs
```

The training command selects enabled families according to their manifests and trains new or unfinished configurations. Completed configurations are reused when their fit settings and input data match; changed metric selections are scored from saved forecasts. There is no built-in dataset default. The Walmart configuration uses the real [Walmart Sales Forecast dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast); preparation downloads the configured raw files, and training only reads local data.

## Config structure

```text
experimentation/<experiment>/configs/
├── config.yaml
└── models/
    ├── naive/
    │   ├── default.yaml
    │   └── manifest.yaml
    └── arima/
        ├── default.yaml
        └── manifest.yaml
```

The main `config.yaml` owns the dataset, evaluation plan, EDA settings, output paths, and family switches. Its final section controls what runs:

```yaml
models:
  naive:
    enabled: true
  arima:
    enabled: true
```

Disable every ARIMA configuration, including automatic order searches:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs models.arima.enabled=false
```

A family's `default.yaml` contains its registry name and settings. ARIMA's `selection: fixed` uses `params.order`, `params.seasonal_order`, and `params.trend`. `selection: auto` uses `search` instead; fixed `params` are ignored during automatic selection. Its `manifest.yaml` includes those defaults and owns its sweep:

```yaml
# models/arima/manifest.yaml
defaults:
  - default
  - _self_

sweep:
  enabled: true
  parameters:
    params.order:
      - [0, 1, 0]
      - [1, 1, 0]
    params.seasonal_order:
      - [0, 0, 0, 0]
      - [1, 0, 0, 52]
```

This runs four combinations for ARIMA. A complete bracketed order is one choice. `sweep.enabled: false` runs `default.yaml` once. For specific combinations, use `variants` instead of independent grid axes:

```yaml
sweep:
  enabled: true
  parameters: {}
  variants:
    - params:
        order: [1, 1, 1]
        seasonal_order: [0, 0, 0, 0]
    - params:
        order: [1, 0, 0]
        seasonal_order: [1, 0, 0, 52]
    - use_exog: true
    - selection: auto
    - selection: auto
      use_exog: true
```

Each variant merges into the family defaults. Grid axes, if present, are then applied to each variant. No `-m` is needed: the forecasting experiment runner expands the per-family manifests. Ordinary YAML lists define choices; there are no encoded sweep strings or separate sweep preset names.

### Hand-picked configurations and later iterations

Append a case to `models/arima/manifest.yaml` under `sweep.variants`. For seasonal `(1,0,0) × (1,0,0,52)` with exogenous variables:

```yaml
sweep:
  enabled: true
  parameters: {}
  variants:
    # Keep earlier entries here and append more cases as you explore.
    - selection: fixed
      use_exog: true
      params:
        order: [1, 0, 0]
        seasonal_order: [1, 0, 0, 52]
```

Each case inherits unspecified values from `default.yaml`, independently of other cases. Exogenous inputs come from `data.known_future_columns`; they must be available at forecast time. The Walmart manifest already includes this case and uses `IsHoliday`.

Run the usual training command again. Completed matching cases are skipped, failed/incomplete cases are retried, and new cases are fitted. Earlier results stay available even if their entries are removed from the current manifest. Cache checks compare prepared target values and dates, plus regressor values and columns for exogenous models. Editing unused automatic search settings does not invalidate fixed-order fits.

```yaml
# config.yaml
execution:
  rerun: false
```

To deliberately refit the cases selected by today's manifest, pass `execution.rerun=true`. Other saved configurations remain intact. To run just one hand-picked case from the terminal:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs \
  models.naive.enabled=false models.arima.sweep.enabled=false \
  models.arima.selection=fixed models.arima.use_exog=true \
  'models.arima.params.order=[1,0,0]' \
  'models.arima.params.seasonal_order=[1,0,0,52]'
```

Training automatically refreshes aggregate metric CSVs and forecast plots using all saved results. You can also rebuild them independently:

```bash
uv run python scripts/analyze.py --config-dir experimentation/walmart_sales/configs
```

Analysis reads saved fit artifacts only and never trains models or reloads raw data. It rebuilds both `results/cv` and `results/holdout`, their per-model plots, metric plots, and overlaid/stacked forecast comparisons, including the combined CV-plus-holdout views. Repeating analysis replaces the generated reports rather than appending duplicate rows. EDA remains a separate command.

### Automatic order selection

Auto-ARIMA is a parameter selection method within `models.arima`. There is one ARIMA manifest and one family enable switch. For a single automatic seasonal search:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs models.arima.sweep.enabled=false models.arima.selection=auto
```

ARIMA's `default.yaml` keeps both sets of settings together:

```yaml
name: arima
selection: fixed
use_exog: false
params:
  order: [1, 1, 1]
  seasonal_order: [0, 0, 0, 0]
  trend: null
search:
  seasonal: true
  m: 52
  max_p: 2
  max_q: 2
  max_P: 1
  max_Q: 1
```

Use `selection: auto` in a sweep variant to compare automatic selection against fixed orders. Search settings can be overridden per variant or swept with axes such as `search.m: [7, 14]`. Grid axes apply to every variant, so keep fixed order grids and automatic search grids in separate sweeps when their settings differ. Each automatic fit searches only its own training history.

Reports record `model_type: arima` and a separate `selection` value (`fixed` or `auto`). The `auto_arima__order-...__seasonal-...` directory prefix identifies the selection method and preserves the actual fitted orders.

## Hydra's role

Hydra composes the main config with `models/<family>/manifest.yaml` and each manifest's `default.yaml`. `_self_` means merge the settings in the current file after its listed defaults. Terminal overrides apply to the composed config. OmegaConf resolves references such as `${experiment.root}`.

The small forecasting experiment runner reads the resulting family switches and grids. Hydra does not determine fitted model names, result directories, or model-selection metrics. It is optional as an architectural choice; here it supplies config composition and terminal overrides. MLflow tracking is not implemented.

Inspect the resolved config without reading data or fitting:

```bash
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs --cfg job --resolve
```

`--config-dir` selects the experiment directory; `--config-name other_config` selects a different main YAML filename. Run from the repository root so relative data/output paths have the documented meaning, or use absolute paths.

## New datasets

Copy the independent [example experiment](experimentation/example/README.md):

```bash
cp -R experimentation/example experimentation/my_forecast
mkdir -p experimentation/my_forecast/data
```

Set `experiment.name: my_forecast`, provide your real CSV, and configure its date/target columns, frequency, filters, aggregation, forecast horizon, and seasonal periods. The template contains no dummy data and has no dependency on Walmart. Its example settings assume daily observations and a weekly cycle of seven observations; adapt those values to your data.

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs
uv run python scripts/eda.py --config-dir experimentation/my_forecast/configs
```

## Evaluation and naming

Holdout reserves the final fraction or observation count. Expanding-window CV stays entirely before that holdout. Each window fits a fresh model, including a fresh Auto-ARIMA search. CV scores pool all validation predictions; overlapping windows retain each forecast for the repeated dates.

Detailed results and metric comparisons are separated by evaluation:

```text
results/
├── cv/
│   ├── runs.csv
│   ├── mae.csv, rmse.csv, rmsle.csv, mape.csv, smape.csv, wape.csv
│   ├── <fitted-model-name>/
│   │   ├── config.yaml, run.json
│   │   ├── metrics.json, metrics.csv, orders.csv
│   │   ├── series.csv, predictions.csv
│   │   └── plots/
│   │       ├── forecast.png
│   │       └── cv_and_holdout.png
│   └── plots/
│       ├── mae.png, rmse.png, ...
│       ├── forecast_overlay.png
│       └── forecast_stacked.png
├── holdout/                         # Same structure, holdout scores only
├── plots/
│   ├── forecast_overlay.png         # All models, CV and holdout together
│   └── forecast_stacked.png         # One model per panel, both evaluations
└── eda/
```

ARIMA directories use actual fitted orders, for example:

- `arima__order-1-0-0__seasonal-1-0-0-52`
- `auto_arima__order-0-1-1__seasonal-0-1-1-52`

The automatic version uses the same convention with `auto_` at the front. `__exog` identifies a model using future-known regressors. Names contain no dates, attempt counters, split sizes, or search bounds.

All of a model's CV folds stay together. If Auto-ARIMA selects different orders, its CV folder lists `fold-1__order-...__seasonal-...`, then the other folds. `orders.csv` maps every fold to its selected orders and convergence status. Very long descriptions span multiple path components without truncating orders.

Every model has a split-only forecast plot and a plot showing its CV plus holdout forecasts. Forecast comparisons include every model overlaid and every model stacked in separate panels. All timelines show prior observed training history; fold forecasts remain separate lines. CV-only plots stop before the final holdout. Combined plots show both evaluations for inspection; their scores remain separate.

Model comparisons separate differing targets/evaluation windows. Failed runs remain indexed, with completed windows preserved, and do not enter pooled metric rankings. Internal fit artifacts and resolved configs live under `artifacts/configurations`; those internal keys are separate from user-facing fitted model names. New launches retain earlier configurations. Disabling a family or removing a variant only changes what is selected for training; its saved results remain in comparisons. A forced rerun replaces that configuration's previous output without adding attempt counters.

Rebuild reports and all plots without fitting:

```bash
uv run python scripts/analyze.py --config-dir experimentation/walmart_sales/configs
```

## EDA

The separate EDA pipeline uses the configured data loader and writes `results/eda`. It includes full-series, trailing mean/std, distribution, differencing, ACF/PACF, additive STL, and seasonal-profile plots. CSVs retain plotted values; summary JSON/CSV and ADF/KPSS reports describe the prepared target. Short/constant series produce notes for unavailable diagnostics. IQR flags do not remove observations.

Configure analysis windows and seasonal periods under `eda`. The loader enforces the prepared-series contract, including finite values and a regular configured frequency; EDA does not silently clean missing periods.

## Extension boundaries

`data/sources.py` acquires raw files independently of ML task. Forecasting owns its loaders, model factory, fit/predict interface, experiment expansion, evaluation, and EDA. `tasks/registry.py` dispatches to each task's own pipeline and reporters. Other tasks should define their own data contracts.

Future forecasting adapters, including Transformers, can prepare training windows, fit transforms, and train within `fit(history, exog)`, then return a one-dimensional forecast from `predict(horizon, future_exog)`. Their parameters and sweep choices belong in their own default/manifest files. Transformer training is not implemented yet.

Available metrics: MAE, MSE, RMSE, RMSLE, MAPE, SMAPE, WAPE. Percentage metrics use percentages; undefined values are JSON null and blank CSV cells. MAPE excludes zero actuals, SMAPE treats zero/zero as zero, and WAPE is undefined for an all-zero target.

RMSLE is `sqrt(mean((log1p(predicted) - log1p(actual))²))`, using natural logarithms. It supports zeros and is unitless. Any negative actual or prediction makes RMSLE undefined for that fold or pooled evaluation: JSON null, a blank CSV cell, and an Undefined plot label. Forecasts are not clipped and observations are not dropped.

Add `rmsle` under `evaluation.metrics`. Analysis applies the current metric list to all saved forecasts and updates per-fold scores, pooled CV/holdout scores, aggregate CSVs, and metric plots without retraining. Changing the metric list also preserves completed fits on subsequent training launches.
