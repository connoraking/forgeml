# Walmart sales experiment

The source is [aslanahmedov/walmart-sales-forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast), version 1: 421,570 rows, aggregated into 143 weekly observations from February 5, 2010 through October 26, 2012. Column names, filters, aggregation, source, and output paths are configured in `configs/config.yaml`.

```bash
uv run python scripts/prepare_data.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/eda.py --config-dir experimentation/walmart_sales/configs
uv run python scripts/train.py --config-dir experimentation/walmart_sales/configs
```

## Model controls

The main config ends with `models.naive.enabled` and `models.arima.enabled`. Disabling ARIMA skips both fixed orders and automatic order searches.

Each family has `configs/models/<family>/default.yaml` for constructor parameters and `manifest.yaml` for sweep controls. The ARIMA default contains `selection: fixed`, fixed `params`, and automatic `search` settings. Its manifest includes `selection: auto` variants in the same sweep. With these settings, the manifest selects seven configurations:

- Naive once.
- Four explicit ARIMA variants: `(1,1,1)` and seasonal `(1,0,0) × (1,0,0,52)`, each with and without holiday regressors.
- Seasonal Auto-ARIMA with and without holiday regressors.

No `-m` or separate sweep preset is needed. A family's `sweep.enabled: false` runs its default once. `sweep.parameters` defines independent YAML grid axes; `sweep.variants` defines specific combinations. Each variant merges into the defaults, then any grid axes apply.

For a single automatic run, pass `models.arima.sweep.enabled=false models.arima.selection=auto` to the training command. Both automatic variants use annual `m=52` seasonality and search ordinary and seasonal orders. The Canova–Hansen test chooses seasonal differencing, retaining `D=0` when its sample requirement is unmet in early short folds. The optimizer allows 200 iterations. Selected orders and convergence are recorded per evaluation window.

`IsHoliday` is the dataset's provided future-known flag. It does not flag the biggest pre-Christmas sales weeks, so it is not a dedicated Christmas lead-up feature. Annual terms can capture the recurring year-end pattern; model scores determine how well each approach forecasts it.

## Adding later iterations

Append hand-picked combinations to `configs/models/arima/manifest.yaml` under `sweep.variants`. The seasonal exogenous case is already included:

```yaml
- selection: fixed
  use_exog: true
  params:
    order: [1, 0, 0]
    seasonal_order: [1, 0, 0, 52]
```

Run the usual training command again: `execution.rerun: false` skips completed cases with matching data and settings, retries failed/incomplete cases, and trains additions. Previous configurations remain in comparisons even if removed from today's manifest. Pass `execution.rerun=true` to deliberately refit selected configurations. Disabling a family stops it from training without deleting its results.

Training updates all aggregate reports automatically. `scripts/analyze.py` below refreshes CSVs and forecast/metric plots from every saved configuration without fitting models or requiring raw data. EDA plots are handled by `scripts/eda.py` separately.

## Evaluation and plots

The final 20% is held out. Three eight-week CV windows stay before that holdout. The first validation window covers November 4 through December 23, 2011; the holdout covers April 20 through October 26, 2012.

`results/cv` and `results/holdout` contain separate fitted-model directories and metric CSVs. Automatic names use actual selected orders, never search bounds. CV folds stay together; if orders vary, the folder lists each fold's orders and `orders.csv` gives the mapping.

Every model has `plots/forecast.png` for its split and `plots/cv_and_holdout.png` for both evaluations. Each split's `plots` directory contains all-model overlays, stacked forecast panels, and metric comparisons. `results/plots` contains all-model overlaid and stacked CV-plus-holdout views. Every timeline includes preceding observed history.

```bash
uv run python scripts/analyze.py --config-dir experimentation/walmart_sales/configs
```

The independent [example experiment](../example/README.md) can be copied for another dataset. None of the scripts defaults to Walmart.

The metric list includes `rmsle` (Root Mean Squared Logarithmic Error). Analysis recomputes the configured metrics from saved forecasts, so adding a metric requires no retraining. RMSLE uses natural `log1p`, supports zeros, and returns an undefined score if any actual or prediction is negative.
