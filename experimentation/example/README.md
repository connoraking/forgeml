# Forecasting experiment template

This is a self-contained starting point for a real dataset. It does not use or import the Walmart experiment, and it contains no generated sample data. Deleting `experimentation/walmart_sales` does not affect this experiment or the CLI entry points.

## Create an experiment

Run from the repository root:

```bash
cp -R experimentation/example experimentation/my_forecast
mkdir -p experimentation/my_forecast/data
```

Edit `experimentation/my_forecast/configs/config.yaml`:

- Set `experiment.name: my_forecast`. Its default root becomes `experimentation/my_forecast`.
- Place your real CSV at `data/series.csv`, or change `data.path` to its location.
- Set `date_column`, `target_column`, `frequency`, and any filters/aggregation rules.
- Set holdout/CV sizes for your history and forecast horizon.
- Adjust `eda.seasonal_period`, the seasonal model's period, and the sweep periods. The template assumes daily observations with a weekly cycle of 7; those values are examples of settings to adapt, not inferred properties of your data.

No Python files need to change for another dataset using the CSV loader. Other input formats require a loader implementation. Only the forecasting task is currently implemented.

The optional `model=arima_with_exog` manifest requires `data.known_future_columns` to list features available at forecast time. The default model-comparison sweep excludes it because the template starts with no future features.

## Terminal commands

```bash
# Inspect the final configuration without reading data or fitting a model.
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs --cfg job --resolve

# A single model, chosen from this experiment's model directory.
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs model=arima

# Enable both CV and the final holdout.
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs model=arima evaluation.cv.enabled=true

# Use the same experiment for EDA and local-source preparation.
uv run python scripts/eda.py --config-dir experimentation/my_forecast/configs
uv run python scripts/prepare_data.py --config-dir experimentation/my_forecast/configs

# Rebuild saved evaluation reports without fitting models.
uv run python scripts/consolidate_results.py --config experimentation/my_forecast/configs/config.yaml --refresh-plots
```

The training, EDA, and preparation scripts accept `--config-dir` and default to the filename `config.yaml`. Use `--config-name another_config` for another main filename in that directory. The report script takes the YAML file directly with `--config`.

Relative config directories and data/output paths are interpreted from your terminal's working directory. Run these commands from the repository root. You can supply absolute paths and override `experiment.root=/absolute/path/to/my_forecast` for another location. Model and sweep choices are looked up inside the selected config directory, independently of the current dataset's name.

## How Hydra builds the config

The main file starts with:

```yaml
defaults:
  - model: naive
  - _self_
  - sweep: null
```

1. `model: naive` loads `model/naive.yaml` under `cfg.model`.
2. `_self_` merges the main file's data, evaluation, and output settings.
3. `sweep: null` selects no sweep unless you request one. `sweep=arima_orders` loads `sweep/arima_orders.yaml` after the main settings. That file can enable CV and select ARIMA as the base model.
4. Terminal value overrides apply to the composed config. `model=arima` changes the selected model manifest; `model.params.order=[2,1,0]` changes a parameter. Quote overrides containing brackets in your shell.
5. `${experiment.name}` and similar expressions reference other config values. Changing the experiment name updates the root and derived output paths. `--cfg job --resolve` shows the final values.

The last two defaults in the full main file only disable Hydra's automatic log files. The `hydra` section keeps working-directory changes off and stores its sweep manifest under `artifacts/sweeps`. The forecasting pipeline determines result names and saves CSV/JSON/PNG outputs.

`# @package _global_` in a sweep means its `evaluation` and `hydra` sections merge at the root. `override /model: arima` changes the selected model defaults. These directives control composition; they do not create a separate kind of model.

Hydra is optional as an architectural choice: plain YAML plus a custom loader could do the same job. We use Hydra's composition, terminal overrides, and basic sweeper so those mechanisms do not need a separate implementation. These are [standard Hydra features](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/).

## Sweep both ARIMA orders

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs -m sweep=arima_orders
```

The saved sweep contains:

```yaml
hydra:
  sweeper:
    params:
      model.params.order: "[0,1,0],[1,1,0]"
      model.params.seasonal_order: "[0,0,0,0],[1,0,0,7]"
```

This produces every combination:

| Job | `order` | `seasonal_order` |
| --- | --- | --- |
| 1 | `[0,1,0]` | `[0,0,0,0]` |
| 2 | `[0,1,0]` | `[1,0,0,7]` |
| 3 | `[1,1,0]` | `[0,0,0,0]` |
| 4 | `[1,1,0]` | `[1,0,0,7]` |

`order` is `[p,d,q]`; `seasonal_order` is `[P,D,Q,period]`. One bracketed list is one choice. The outer YAML string contains Hydra's comma-separated choice syntax. `[0,0,0,0]` disables the seasonal component.

`model/arima.yaml` supplies defaults for the same ARIMA implementation used by the seasonal manifest. The grid overrides both constructor parameters. A manifest's filename does not restrict which orders can be swept.

You can also supply both axes directly:

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs -m model=arima 'model.params.order=[0,1,0],[1,1,0]' 'model.params.seasonal_order=[0,0,0,0],[1,0,0,7]'
```

Without `-m`, a bracketed list such as `'model.params.order=[2,1,0]'` is one parameter value, not a grid. Selecting a saved sweep without `-m` loads its shared settings but does not expand its parameter choices. The basic launcher executes jobs sequentially on this machine.

`--cfg job --resolve` prints the base job, not the expanded grid. To inspect the saved axes without launching jobs:

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs sweep=arima_orders --cfg hydra --package hydra.sweeper
```

## Sweep specific pairs

When you want selected pairs instead of every combination:

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs -m sweep=arima_pairs
```

`arima_pairs.yaml` sweeps two dictionaries under `model.params`. Each dictionary changes `order` and `seasonal_order` together, producing exactly two jobs. Unspecified parameters keep their manifest values.

## Compare models or create another sweep

```bash
uv run python scripts/train.py --config-dir experimentation/my_forecast/configs -m sweep=model_comparison
```

`model_comparison.yaml` selects complete model manifests. Keep an ARIMA parameter grid separate from a comparison that includes naive or another model with different constructor arguments. A future Transformer manifest can sweep its own parameters through the same mechanism after its model adapter is implemented.

To create another sweep, copy a file in `configs/sweep`, change its model selection and choices, and pass its filename without `.yaml` as `sweep=your_sweep`. No source-code registration is needed for a new sweep.

## Outputs and tracking

Detailed files go under `artifacts/runs/<descriptive-name>`; consolidated metrics and plots go under `results`. Each configuration keeps its latest output. Hydra's numbered terminal jobs are launch indices, not attempts or result-folder names.

MLflow is a separate tracking tool. It does not replace config composition or this grid launcher. Current tracking consists of the saved CSV/JSON/PNG artifacts; MLflow logging is not implemented.
