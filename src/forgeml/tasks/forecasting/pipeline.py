"""Run a configured experiment and save metrics, forecasts, and fit details."""

import hashlib
import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .dataloaders import ForecastData, load_data
from .evaluation.metrics import evaluate
from .evaluation.reporting import consolidate_reports
from .evaluation.results import create_run_directory, save_observed_series, save_results
from .evaluation.splitting import EvaluationWindow, plan_evaluation
from .factory import create_forecaster

logger = logging.getLogger(__name__)


def _run_window(
    data: ForecastData,
    window: EvaluationWindow,
    config: DictConfig,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fit a fresh model, forecast one window, and describe what was evaluated."""
    train = data.target.iloc[: window.train_end]
    test = data.target.iloc[window.test_start : window.test_end]

    train_exog = None
    future_exog = None

    if config.model.use_exog:
        if data.exog is None:
            raise ValueError(
                "This model needs data.known_future_columns in the config."
            )

        train_exog = data.exog.iloc[: window.train_end]
        future_exog = data.exog.iloc[window.test_start : window.test_end]

    # A fresh model per fold is essential: learned parameters and automatic order
    # searches must not carry information from a later validation window backwards.
    model = create_forecaster(config.model)
    started = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train, exog=train_exog)
        forecast = model.predict(len(test), future_exog=future_exog)

    metrics = evaluate(test.to_numpy(), forecast, config.evaluation.metrics)
    scores = {"evaluation": window.evaluation, "fold": window.fold, **metrics}

    predictions = pd.DataFrame(
        {
            "evaluation": window.evaluation,
            "fold": window.fold,
            "date": test.index,
            "actual": test.to_numpy(),
            "predicted": forecast,
        }
    )

    details = {
        **scores,
        "train_points": len(train),
        "test_points": len(test),
        "train_start": str(train.index[0]),
        "train_end": str(train.index[-1]),
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "model_fit": model.describe_fit(),
        "warnings": sorted({str(warning.message) for warning in caught}),
    }

    return predictions, scores, details


def run_experiment(config: DictConfig, launch_metadata: dict | None = None) -> dict:
    """Evaluate the configured plan and replace this variant's previous output."""
    directory = create_run_directory(config)
    predictions = []
    scores = []
    report = {
        "task": "forecasting",
        "experiment": config.experiment.name,
        "run_id": f"run-{uuid4().hex[:12]}",
        "run_name": directory.name,
        "model": config.model.get("label", config.model.name),
        "model_type": config.model.name,
        "parameters": OmegaConf.to_container(config.model.params, resolve=True),
        "data_config": OmegaConf.to_container(config.data, resolve=True),
        "requested_metrics": list(config.evaluation.metrics),
        "started_at": datetime.now(UTC).isoformat(),
        "launch": launch_metadata or {},
        "status": "running",
        "artifact_directory": str(directory),
        "windows": [],
        "summary": {},
    }

    print(f"Running {report['run_name']}")

    try:
        data = load_data(config.data)
        save_observed_series(directory, data.target)
        # Matching dates alone do not prove that two runs used the same target data.
        target_hash = pd.util.hash_pandas_object(data.target, index=True).to_numpy()
        report["target_fingerprint"] = hashlib.sha256(target_hash.tobytes()).hexdigest()
        report["dataset"] = {
            "path": str(Path(config.data.path).expanduser().resolve()),
            "points": len(data.target),
            "start": str(data.target.index[0]),
            "end": str(data.target.index[-1]),
            "frequency": config.data.get("frequency"),
        }
        windows = plan_evaluation(len(data.target), config.evaluation)

        for window in windows:
            forecast_rows, window_scores, details = _run_window(data, window, config)

            predictions.append(forecast_rows)
            scores.append(window_scores)
            report["windows"].append(details)

            print(f"{window.evaluation} fold={window.fold}: {window_scores}")

        # Pooled CV metrics evaluate all validation forecasts together. Overlapping
        # windows intentionally include each forecast made for a given date.
        combined = pd.concat(predictions, ignore_index=True)

        for evaluation, rows in combined.groupby("evaluation"):
            report["summary"][evaluation] = evaluate(
                rows.actual.to_numpy(),
                rows.predicted.to_numpy(),
                config.evaluation.metrics,
            )

        report["status"] = "completed"

    except Exception as error:
        # One recovery boundary keeps the original exception and preserves any
        # completed folds. We do not pretend that a failed fit was a successful run.
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save_results(directory, report, predictions, scores)
        consolidate_reports(config.results.directory, config.artifacts.directory)
        logger.error("Experiment failed; details saved in %s", directory)
        raise

    save_results(directory, report, predictions, scores)
    consolidate_reports(config.results.directory, config.artifacts.directory)
    print(f"Artifacts: {directory}")
    print(f"Metric reports: {config.results.directory}")

    return report
