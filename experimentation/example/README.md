# Forecasting experiment template

Copy this directory to start an experiment using real data. No script depends on Walmart, and the template contains no generated dataset.

## Setup

From the repository root:

```bash
cp -R experimentation/example experimentation/my_forecast
mkdir -p experimentation/my_forecast/data
```

In `configs/config.yaml`, set `experiment.name: my_forecast`, place your real CSV at `data/series.csv` or change `data.path`, and set date/target columns, frequency, filters, and aggregation. Adapt evaluation lengths and seasonal periods; this template assumes daily observations with a weekly cycle of seven. CV and holdout are enabled, so provide enough training history for both or disable one.

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs --cfg job --resolve
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs
uv run python scripts/eda.py --config-dir experimentation/my_forecast/configs
uv run python scripts/prepare_data.py --config-dir experimentation/my_forecast/configs
```

Preparation checks the configured local source directory; it does not fabricate data. Other source providers, such as Kaggle, can be configured under `data.source`. New input formats need a loader implementation; other CSV datasets only need config changes.

## What controls execution

The main file's final section enables model families:

```yaml
models:
  naive:
    enabled: true
  arima:
    enabled: true
```

Each `models/<family>/default.yaml` contains model constructor settings. Each `manifest.yaml` loads its default and controls its own sweep. ARIMA includes both fixed and automatic order selection; there is no separate Auto-ARIMA family. Set `selection: fixed` to use `params`, or `selection: auto` to use `search` from the same ARIMA default file.

The ARIMA manifest demonstrates a normal YAML grid:

```yaml
defaults:
  - default
  - _self_

sweep:
  enabled: false
  parameters:
    params.order:
      - [0, 1, 0]
      - [1, 1, 0]
    params.seasonal_order:
      - [0, 0, 0, 0]
      - [1, 0, 0, 7]
```

Set `enabled: true` to run all four combinations. Each complete order is one choice. Disable it to run `default.yaml` once. Explicit paired combinations can instead be listed under `sweep.variants`, each merging settings into the default. The Walmart ARIMA manifest shows that pattern, including `selection: auto` variants alongside fixed-order variants. To sweep automatic search settings, use `selection: auto` as the default and grid keys such as `search.m` or `search.max_p` instead of `params.order`.

```bash
# Disable all ARIMA configurations, including automatic searches.
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs models.arima.enabled=false

# Enable the ARIMA grid from the terminal.
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs models.arima.sweep.enabled=true

# Run one automatic seasonal search (the example default uses m=7).
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs models.arima.sweep.enabled=false models.arima.selection=auto

# Change a default parameter for this invocation.
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs 'models.arima.params.order=[2,1,0]'
```

A parameter already listed in an enabled grid takes its values from the grid. Edit the grid, or disable the sweep when you want one terminal override to be the sole choice. Feature-based models need configured `data.known_future_columns` that are available at forecast time; `use_exog: true` enables those features.

## Adding configurations over time

Append hand-picked entries to a family's `sweep.variants`, with `sweep.enabled: true` and `sweep.parameters: {}`. For example, adapt the seasonal period to your data:

```yaml
sweep:
  enabled: true
  parameters: {}
  variants:
    - selection: fixed
      use_exog: true
      params:
        order: [1, 0, 0]
        seasonal_order: [1, 0, 0, 7]
```

Set `data.known_future_columns` before enabling exogenous variables. Each variant inherits family defaults independently.

`execution.rerun: false` makes repeated training launches incremental: matching completed cases are skipped, failed/incomplete cases are retried, and additions are trained. Data changes are detected from the prepared observations and any regressors used. Removed variants and disabled families keep their saved results. To refit selected cases intentionally, pass `execution.rerun=true`.

Training refreshes aggregate CSVs and forecast plots automatically. The analysis command in the Results section performs the same rebuild from all saved fit artifacts, without model training or raw data access.

## Hydra

The main `defaults` list composes each model family's manifest. Each manifest loads its `default.yaml`; `_self_` merges the current file's settings afterward. The main file then sets data/evaluation settings and family switches. Terminal overrides apply to that composed object. `${experiment.root}` references another config value.

The experiment runner expands each manifest's `sweep.parameters` grid and `variants`; no Hydra `-m` flag is needed. Hydra supplies composition and terminal overrides, while the forecasting pipeline controls fitting, naming, and reports. `--cfg job --resolve` prints the final settings without reading data or fitting models.

Relative config directories and data/output paths are interpreted from the terminal's working directory. Run from the repository root or supply absolute paths. Use `--config-name other_config` for another main YAML filename.

## Results

`results/cv` and `results/holdout` independently contain model directories, a run index, one CSV per metric, and comparison plots. Model directory names show fitted orders; automatic models use the same name with `auto_` prefixed. CV folds stay together, with varying automatic orders listed by fold in the directory name and in `orders.csv`.

Each model has its split plot and a CV-plus-holdout plot. Both split directories contain overlaid and stacked all-model forecasts. `results/plots` contains the all-model views with CV and holdout together. Observed training history is visible throughout, and scores remain separate by evaluation.

The underlying config/fit artifacts are kept under `artifacts/configurations`. Generated files are ignored by Git. EDA remains under `results/eda` and does not train models.

```bash
uv run python scripts/analyze.py --config-dir experimentation/my_forecast/configs
```

This rebuilds the saved result tables and plots without rerunning fits.

The metric list includes `rmsle` (Root Mean Squared Logarithmic Error). Analysis recomputes the configured metrics from saved forecasts, so adding a metric requires no retraining. RMSLE uses natural `log1p`, supports zeros, and returns an undefined score if any actual or prediction is negative.
