"""Rebuild per-metric comparison CSVs from the saved run artifacts."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from omegaconf import DictConfig

from .metrics import METRICS
from .plots import plot_metric_comparison, plot_run_forecasts

# Stable columns also make an empty runs.csv useful when every run has failed.
RUN_COLUMNS = [
    "run_name",
    "model",
    "model_type",
    "status",
    "started_at",
    "finished_at",
    "parameters",
    "artifact_path",
    "run_id",
]


def rebuild_reports(config: DictConfig, refresh_plots: bool = False) -> None:
    """Task entry point for rebuilding forecasting reports from an experiment config."""
    consolidate_reports(
        config.results.directory,
        config.artifacts.directory,
        refresh_plots=refresh_plots,
    )


def _comparison_group(metadata: dict, windows: list[dict]) -> str:
    """Distinguish scores for different target series or evaluation windows."""
    boundaries = [
        {
            key: window.get(key)
            for key in (
                "train_start",
                "train_end",
                "train_points",
                "test_start",
                "test_end",
                "test_points",
            )
        }
        for window in windows
    ]
    identity = {"target": metadata["target_fingerprint"], "windows": boundaries}
    serialized = json.dumps(identity, sort_keys=True)
    group = hashlib.sha256(serialized.encode()).hexdigest()[:12]

    return group


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Readers see either the previous complete report or the new complete report."""
    temporary = path.with_name(f".{path.stem}-{uuid4().hex[:8]}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def consolidate_reports(
    results_directory: Path, artifacts_directory: Path, refresh_plots: bool = False
) -> None:
    """Write one comparison CSV/plot per metric and a run index.

    Rebuilding from artifacts is repeatable and never appends duplicate rows.
    A metric row is one run's pooled CV score or its final holdout score. Fold
    scores stay in the artifact, where they cannot be mistaken for pooled scores.
    """
    results_directory = Path(results_directory).expanduser().resolve()
    artifacts_directory = Path(artifacts_directory).expanduser().resolve()
    results_directory.mkdir(parents=True, exist_ok=True)

    runs = []
    metrics = {}
    requested_metrics = set()

    for metadata_path in sorted((artifacts_directory / "runs").rglob("run.json")):
        directory = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        evaluation = json.loads((directory / "metrics.json").read_text())
        plot_run_forecasts(directory, metadata, refresh=refresh_plots)
        requested_metrics.update(metadata["requested_metrics"])

        # Use a relative artifact link so an experiment folder can be moved as a unit.
        row = {key: metadata.get(key) for key in RUN_COLUMNS}
        row["run_name"] = metadata["run_name"]
        row["parameters"] = json.dumps(metadata["parameters"], sort_keys=True)
        row["artifact_path"] = os.path.relpath(directory, results_directory)
        runs.append(row)

        # Failed runs remain visible in runs.csv, but do not enter score comparisons.
        if metadata["status"] != "completed":
            continue

        for name, scores in evaluation["summary"].items():
            windows = [w for w in evaluation["windows"] if w["evaluation"] == name]
            group = _comparison_group(metadata, windows)

            for metric, value in scores.items():
                metric_row = {
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "model": row["model"],
                    "parameters": row["parameters"],
                    "evaluation": name,
                    "comparison_id": group,
                    metric: value,
                    "started_at": row["started_at"],
                    "artifact_path": row["artifact_path"],
                }
                metrics.setdefault(metric, []).append(metric_row)

    _write_csv(results_directory / "runs.csv", pd.DataFrame(runs, columns=RUN_COLUMNS))

    # Failed or deleted runs must not leave old scores looking like current reports.
    existing_metrics = {
        name for name in METRICS if (results_directory / f"{name}.csv").exists()
    }

    for metric in sorted(requested_metrics | metrics.keys() | existing_metrics):
        columns = [
            "run_name",
            "model",
            "parameters",
            "evaluation",
            "comparison_id",
            metric,
            "started_at",
            "artifact_path",
            "run_id",
        ]
        rows = metrics.get(metric, [])
        frame = pd.DataFrame(rows, columns=columns)
        _write_csv(results_directory / f"{metric}.csv", frame)
        plot_metric_comparison(frame, metric, results_directory)
