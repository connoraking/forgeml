"""Publish independent CV/holdout results and forecast comparison views."""

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .plots import (
    model_label,
    plot_forecast_comparison,
    plot_metric_comparison,
    plot_run_forecasts,
)
from .results import (
    evaluation_name,
    rescore_saved_results,
    result_directory,
    write_json,
)

RUN_COLUMNS = [
    "run_name",
    "model",
    "model_type",
    "selection",
    "status",
    "started_at",
    "finished_at",
    "parameters",
    "selected_orders",
    "artifact_path",
    "run_id",
]
_PREDICTION_COLUMNS = ["evaluation", "fold", "date", "actual", "predicted"]


def rebuild_reports(config: DictConfig) -> None:
    """Reports are rebuilt from fit artifacts without reading or fitting raw data."""
    consolidate_reports(
        config.results.directory, config.artifacts.directory, config.evaluation.metrics
    )


def _comparison_group(metadata: dict, windows: list[dict]) -> str:
    boundaries = [
        {
            key: window[key]
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
    identity = {
        "target": metadata.get("target_fingerprint", metadata["data_config"]),
        "windows": boundaries,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _read_records(artifacts: Path, metrics: Sequence[str] | None) -> list[dict]:
    records = []
    for source in sorted((artifacts / "configurations").glob("*/run.json")):
        directory = source.parent
        predictions_path = directory / "predictions.csv"
        series_path = directory / "series.csv"
        records.append(
            {
                "metadata": rescore_saved_results(directory, metrics),
                "evaluation": json.loads((directory / "metrics.json").read_text()),
                "config": OmegaConf.load(directory / "config.yaml"),
                "directory": directory,
                "predictions": (
                    pd.read_csv(predictions_path, parse_dates=["date"])
                    if predictions_path.exists()
                    else pd.DataFrame(columns=_PREDICTION_COLUMNS)
                ),
                "series": (
                    pd.read_csv(series_path, parse_dates=["date"])
                    if series_path.exists()
                    else pd.DataFrame(columns=["date", "actual"])
                ),
            }
        )
    return records


def _publish_model(record: dict, scope: str, root: Path, report_root: Path) -> dict:
    metadata = record["metadata"]
    windows = [w for w in record["evaluation"]["windows"] if w["evaluation"] == scope]
    name = evaluation_name(metadata, windows)
    directory = result_directory(root, name)
    directory.mkdir(parents=True, exist_ok=True)
    summary = record["evaluation"]["summary"].get(scope, {})

    # One directory contains all CV folds for this model. orders.csv is the
    # explicit fold-to-order mapping, including automatic selections and convergence.
    orders = [
        {
            "fold": window["fold"],
            "order": json.dumps(window["model_fit"]["order"])
            if "order" in window["model_fit"]
            else None,
            "seasonal_order": json.dumps(window["model_fit"]["seasonal_order"])
            if "seasonal_order" in window["model_fit"]
            else None,
            "converged": window["model_fit"].get("converged"),
        }
        for window in windows
    ]
    pd.DataFrame(
        orders, columns=["fold", "order", "seasonal_order", "converged"]
    ).to_csv(directory / "orders.csv", index=False)
    scores = [
        {
            "fold": window["fold"],
            **{metric: window[metric] for metric in metadata["requested_metrics"]},
        }
        for window in windows
    ]
    pd.DataFrame(scores, columns=["fold", *metadata["requested_metrics"]]).to_csv(
        directory / "metrics.csv", index=False
    )
    predictions = record["predictions"]
    predictions.loc[predictions.evaluation == scope].to_csv(
        directory / "predictions.csv", index=False
    )
    record["series"].to_csv(directory / "series.csv", index=False)
    # The public snapshot uses the same schema as an experiment config and can
    # be rerun directly: just this family, this configured case, no sweep.
    snapshot = OmegaConf.to_container(record["config"], resolve=True)
    definition = snapshot.pop("model")
    definition["enabled"] = True
    definition["sweep"] = {"enabled": False, "parameters": {}}
    snapshot["models"] = {definition["name"]: definition}
    OmegaConf.save(OmegaConf.create(snapshot), directory / "config.yaml")
    write_json(directory / "metrics.json", {"summary": summary, "windows": windows})
    write_json(
        directory / "run.json",
        {
            **metadata,
            "run_name": name,
            "evaluation": scope,
            "source_artifact": os.path.relpath(record["directory"], directory),
        },
    )
    plot_run_forecasts(record, directory, scope)

    label = model_label(record) if windows else metadata["model_type"] + " (unfitted)"
    return {
        "run_name": name,
        "model": label,
        "model_type": metadata["model_type"],
        "selection": metadata.get("selection"),
        "status": metadata["status"],
        "started_at": metadata["started_at"],
        "finished_at": metadata["finished_at"],
        "parameters": json.dumps(metadata["parameters"], sort_keys=True),
        "selected_orders": json.dumps(orders),
        "artifact_path": os.path.relpath(directory, report_root),
        "run_id": metadata["run_id"],
    }


def consolidate_reports(
    results_directory: Path,
    artifacts_directory: Path,
    metrics: Sequence[str] | None = None,
) -> None:
    """Each split owns its model folders, pooled metric CSVs, and comparison plots.

    Root-level plots show both evaluations for visual comparison only. Their
    scores remain separate and models are grouped by matching target/windows.
    """
    results = Path(results_directory).expanduser().resolve()
    artifacts = Path(artifacts_directory).expanduser().resolve()
    records = _read_records(artifacts, metrics)

    for scope in ("cv", "holdout"):
        root = results / scope
        root.mkdir(parents=True, exist_ok=True)
        # Retire earlier published names when automatic orders change on a rerun.
        for metadata in list(root.rglob("run.json")):
            shutil.rmtree(metadata.parent)
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

        rows = []
        metric_rows = {}
        requested = set()
        selected_records = [r for r in records if r["config"].evaluation[scope].enabled]
        contexts = {}
        latest = {}
        for record in sorted(
            selected_records, key=lambda r: r["metadata"]["started_at"]
        ):
            windows = [
                w for w in record["evaluation"]["windows"] if w["evaluation"] == scope
            ]
            group = _comparison_group(record["metadata"], windows)
            contexts[record["metadata"]["run_id"]] = group
            # Two searches selecting the same orders produce one current fitted
            # model, rather than duplicate directories with hidden search IDs.
            latest[(group, evaluation_name(record["metadata"], windows))] = record
        selected_records = list(latest.values())
        for record in selected_records:
            metadata = record["metadata"]
            group = contexts[metadata["run_id"]]
            model_root = root
            if len(set(contexts.values())) > 1:
                # Different datasets/splits need separate destinations as well
                # as separate comparison panels, even with identical model orders.
                dataset = Path(record["config"].data.path).stem
                model_root = root / f"data-{dataset}__{group}"
            row = _publish_model(record, scope, model_root, root)
            rows.append(row)
            requested.update(metadata["requested_metrics"])
            if metadata["status"] != "completed":
                continue
            windows = [
                w for w in record["evaluation"]["windows"] if w["evaluation"] == scope
            ]
            group = _comparison_group(metadata, windows)
            for metric, score in record["evaluation"]["summary"][scope].items():
                metric_rows.setdefault(metric, []).append(
                    {
                        **row,
                        "comparison_id": group,
                        metric: score,
                    }
                )

        pd.DataFrame(rows, columns=RUN_COLUMNS).to_csv(root / "runs.csv", index=False)
        # Removed metrics must not leave old scores on disk.
        for path in root.glob("*.csv"):
            if path.name != "runs.csv" and path.stem not in requested:
                path.unlink()
                (root / "plots" / f"{path.stem}.png").unlink(missing_ok=True)
        for metric in sorted(requested):
            frame = pd.DataFrame(
                metric_rows.get(metric, []),
                columns=[*RUN_COLUMNS, "comparison_id", metric],
            )
            frame.to_csv(root / f"{metric}.csv", index=False)
            plot_metric_comparison(frame, metric, root)
        for stacked in (False, True):
            plot_forecast_comparison(selected_records, root / "plots", scope, stacked)

    for stacked in (False, True):
        plot_forecast_comparison(records, results / "plots", stacked=stacked)
