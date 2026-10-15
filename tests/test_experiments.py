"""Metrics, leakage boundaries, CSV preparation, and saved experiment artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from forgeml.tasks.forecasting.dataloaders import load_data
from forgeml.tasks.forecasting.evaluation.metrics import evaluate
from forgeml.tasks.forecasting.evaluation.splitting import plan_evaluation
from forgeml.tasks.forecasting.experiments import expand_experiments
from forgeml.tasks.forecasting.pipeline import run_experiment

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experimentation/example/configs"


@pytest.fixture
def config(tmp_path):
    # Small CSVs exercise experiment mechanics without downloading data during pytest.
    dates = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    rows = pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "Weekly_Sales": np.repeat(np.arange(1, 31, dtype=float), 2),
            "IsHoliday": np.repeat(np.arange(30) % 4 == 0, 2),
        }
    )
    path = tmp_path / "sales.csv"
    rows.sample(frac=1, random_state=42).to_csv(path, index=False)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")

    cfg.data.path = str(path)
    cfg.data.date_column = "Date"
    cfg.data.target_column = "Weekly_Sales"
    cfg.data.frequency = "W-FRI"
    cfg.data.known_future_columns = ["IsHoliday"]
    cfg.data.aggregate = {"Weekly_Sales": "sum", "IsHoliday": "max"}
    cfg.results.directory = str(tmp_path / "results")
    cfg.artifacts.directory = str(tmp_path / "artifacts")
    cfg.evaluation.cv.n_splits = 2
    cfg.evaluation.cv.horizon = 3
    cfg.evaluation.cv.step = 3
    cfg.evaluation.cv.min_train_size = 10

    cfg.evaluation.cv.enabled = False
    return expand_experiments(cfg)[0]


def test_metrics_and_zero_sales_are_explicit() -> None:
    metrics = evaluate(
        np.array([0, 10, 20]),
        np.array([0, 12, 16]),
        ["mae", "mse", "rmse", "mape", "smape", "wape"],
    )

    assert metrics["mae"] == pytest.approx(2)
    assert metrics["mse"] == pytest.approx(20 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(20 / 3))
    assert metrics["mape"] == pytest.approx(20)
    assert metrics["smape"] == pytest.approx(100 * (4 / 22 + 8 / 36) / 3)
    assert metrics["wape"] == pytest.approx(20)

    all_zero = evaluate(np.zeros(2), np.zeros(2), ["mape", "smape", "wape"])

    assert all_zero == {"mape": None, "smape": 0.0, "wape": None}

    with pytest.raises(ValueError, match="matching"):
        evaluate(np.array([1, 2]), np.array([1]), ["rmse"])


def test_loader_aggregates_and_sorts_real_csv_shape(config) -> None:
    data = load_data(config.data)

    assert data.target.index.is_monotonic_increasing
    assert len(data.target) == 30
    assert data.target.iloc[-1] == 60
    assert data.exog.columns.tolist() == ["IsHoliday"]


def test_loader_rejects_missing_weeks(config) -> None:
    path = Path(config.data.path)
    rows = pd.read_csv(path)
    rows = rows.loc[rows.Date != sorted(rows.Date.unique())[5]]
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing periods"):
        load_data(config.data)


def test_cv_never_uses_final_holdout(config) -> None:
    config.evaluation.cv.enabled = True
    windows = plan_evaluation(30, config.evaluation)
    holdout = windows[-1]

    assert holdout.evaluation == "holdout"
    assert holdout.test_start == 24
    assert holdout.test_end == 30

    for window in windows[:-1]:
        assert window.train_end <= window.test_start
        assert window.test_end <= holdout.test_start

    assert windows[0].train_end < windows[1].train_end


@pytest.mark.parametrize("holdout, cv", [(True, False), (False, True), (True, True)])
def test_pipeline_persists_the_configured_evaluation(config, holdout, cv) -> None:
    config.evaluation.holdout.enabled = holdout
    config.evaluation.cv.enabled = cv
    report = run_experiment(config)
    directory = Path(report["artifact_directory"])

    assert report["status"] == "completed"
    assert (directory / "config.yaml").is_file()
    assert (directory / "run.json").is_file()
    assert (directory / "metrics.json").is_file()
    assert (directory / "metrics.csv").is_file()

    observed = pd.read_csv(directory / "series.csv")
    assert len(observed) == 30

    predictions = pd.read_csv(directory / "predictions.csv")
    expected = ({"holdout"} if holdout else set()) | ({"cv"} if cv else set())

    assert set(predictions.evaluation) == expected
    assert set(report["summary"]) == expected
    assert predictions.predicted.notna().all()

    # Every naive forecast equals the observation just before its own test window.
    data = load_data(config.data)

    for window in plan_evaluation(len(data.target), config.evaluation):
        plot = (
            Path(config.results.directory)
            / window.evaluation
            / "naive/plots/forecast.png"
        )
        assert plot.is_file()
        rows = predictions.loc[
            (predictions.evaluation == window.evaluation)
            & (predictions.fold == window.fold)
        ]
        np.testing.assert_allclose(
            rows.predicted, data.target.iloc[window.train_end - 1]
        )

    for scope in expected:
        for metric in config.evaluation.metrics:
            assert (
                Path(config.results.directory) / scope / "plots" / f"{metric}.png"
            ).is_file()


def test_failed_run_keeps_its_config_and_error(config) -> None:
    config.data.path = str(Path(config.data.path).with_name("missing.csv"))

    with pytest.raises(FileNotFoundError):
        run_experiment(config)

    directory = next(
        (Path(config.artifacts.directory) / "configurations").rglob("run.json")
    ).parent
    report = json.loads((directory / "run.json").read_text())

    assert (directory / "config.yaml").is_file()
    assert report["status"] == "failed"
    assert report["error"]["type"] == "FileNotFoundError"


def test_missing_targets_are_not_silently_aggregated(config) -> None:
    path = Path(config.data.path)
    rows = pd.read_csv(path)
    rows.loc[0, "Weekly_Sales"] = np.nan
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="nonfinite"):
        load_data(config.data)


def test_failed_fold_keeps_completed_predictions(config, monkeypatch) -> None:
    from forgeml.tasks.forecasting import pipeline
    from forgeml.tasks.forecasting.models import NaiveForecaster

    config.evaluation.cv.enabled = True
    calls = 0

    def create_model(_config):
        nonlocal calls
        calls += 1

        if calls == 2:
            raise RuntimeError("A later fold could not be fitted.")

        return NaiveForecaster()

    monkeypatch.setattr(pipeline, "create_forecaster", create_model)

    with pytest.raises(RuntimeError, match="later fold"):
        run_experiment(config)

    directory = next(
        (Path(config.artifacts.directory) / "configurations").rglob("run.json")
    ).parent
    report = json.loads((directory / "run.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    predictions = pd.read_csv(directory / "predictions.csv")

    assert report["status"] == "failed"
    assert len(metrics["windows"]) == 1
    assert set(predictions.fold) == {1}
    assert len(predictions) == config.evaluation.cv.horizon


def test_zero_cv_step_is_rejected(config) -> None:
    config.evaluation.cv.enabled = True
    config.evaluation.cv.step = 0

    with pytest.raises(ValueError, match="CV"):
        plan_evaluation(30, config.evaluation)


def test_reports_compare_variants_without_duplicate_rows(config) -> None:
    from omegaconf import OmegaConf

    from forgeml.tasks.forecasting.evaluation.reporting import consolidate_reports

    first = run_experiment(config)
    config.model = OmegaConf.create(
        {
            "name": "arima",
            "use_exog": False,
            "params": {"order": [0, 1, 0]},
        }
    )
    second = run_experiment(config)

    # Rebuilding the same reports must not append a second copy of either result.
    consolidate_reports(config.results.directory, config.artifacts.directory)

    for metric in config.evaluation.metrics:
        rows = pd.read_csv(Path(config.results.directory) / "holdout" / f"{metric}.csv")

        assert set(rows.model_type) == {"naive", "arima"}
        assert len(rows) == 2
        assert rows.comparison_id.nunique() == 1

        for report in (first, second):
            actual = rows.loc[rows.run_id == report["run_id"], metric].iloc[0]
            assert actual == pytest.approx(report["summary"]["holdout"][metric])

    artifact = Path(second["artifact_directory"])
    metadata = json.loads((artifact / "run.json").read_text())

    assert artifact.parent.name == "configurations"
    assert metadata["model"] == "arima"
    assert metadata["parameters"] == {"order": [0, 1, 0]}
    assert metadata["started_at"]


def test_reports_keep_cv_holdout_and_changed_data_distinguishable(config) -> None:
    config.evaluation.cv.enabled = True
    run_experiment(config)

    # Another configured dataset has its own current output and comparison context.
    rows = pd.read_csv(config.data.path)
    rows.Weekly_Sales *= 2
    config.data.path = str(Path(config.data.path).with_name("other_sales.csv"))
    rows.to_csv(config.data.path, index=False)
    run_experiment(config)

    for scope in ("cv", "holdout"):
        scores = pd.read_csv(Path(config.results.directory) / scope / "rmse.csv")
        assert len(scores) == 2
        assert scores.comparison_id.nunique() == 2
        assert scores.artifact_path.nunique() == 2


def test_failed_runs_are_indexed_but_do_not_enter_metric_reports(config) -> None:
    config.data.path = str(Path(config.data.path).with_name("missing.csv"))

    with pytest.raises(FileNotFoundError):
        run_experiment(config)

    directory = Path(config.results.directory) / "holdout"
    runs = pd.read_csv(directory / "runs.csv")

    assert runs.status.tolist() == ["failed"]

    for metric in config.evaluation.metrics:
        assert pd.read_csv(directory / f"{metric}.csv").empty


def test_run_names_describe_settings_and_replace_previous_output(config) -> None:
    from forgeml.tasks.forecasting.evaluation.results import (
        configuration_key,
        create_run_directory,
        fitted_name,
    )

    first = create_run_directory(config)
    second = create_run_directory(config)

    assert first.name == configuration_key(config)
    assert second == first

    assert first.parent.name == "configurations"
    name = fitted_name(
        {
            "model_type": "arima",
            "selection": "auto",
            "use_exog": False,
            "parameters": {},
        },
        {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    )
    assert name == "auto_arima__order-1-0-0__seasonal-0-0-1-52"


def test_model_manifest_controls_joint_order_grid() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=["models.arima.sweep.enabled=true"],
            return_hydra_config=True,
        )

    assert config.models.arima.name == "arima"
    assert config.evaluation.cv.enabled
    assert config.models.arima.sweep.parameters["params.order"] == [
        [0, 1, 0],
        [1, 1, 0],
    ]
