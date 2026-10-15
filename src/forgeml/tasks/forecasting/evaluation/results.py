"""Save reproducible fit artifacts and name results using fitted model orders."""

import hashlib
import json
import math
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .metrics import evaluate


def _slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("._-") or "unnamed"


def configuration_key(config: DictConfig) -> str:
    """Identify a configured fit; filenames and display labels do not affect it."""
    model = OmegaConf.to_container(config.model, resolve=True)
    if model["name"] == "arima":
        # Changing search bounds must not invalidate fixed-order fits, and fixed
        # orders are irrelevant to a search that selects its own parameters.
        model.setdefault("selection", "fixed")
        unused = "params" if model["selection"] == "auto" else "search"
        model.pop(unused, None)
    evaluation = OmegaConf.to_container(config.evaluation, resolve=True)
    evaluation.pop("metrics", None)
    identity = {
        "task": config.task,
        "model": model,
        "data": OmegaConf.to_container(config.data, resolve=True),
        "evaluation": evaluation,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def model_metadata(model: DictConfig) -> dict:
    """Record the model family separately from how its parameters are chosen."""
    definition = OmegaConf.to_container(model, resolve=True)
    selection = definition.get("selection", "fixed") if model.name == "arima" else None
    parameters = (
        {"search": definition.get("search", {})}
        if selection == "auto"
        else definition.get("params", {})
    )
    return {
        "model_type": model.name,
        "selection": selection,
        "use_exog": model.get("use_exog", False),
        "parameters": parameters,
    }


def describe_model(model: DictConfig) -> str:
    """Describe configured settings for progress messages before fitting."""
    metadata = model_metadata(model)
    return fitted_name(metadata, metadata["parameters"])


def _model_prefix(metadata: dict) -> str:
    """Keep the automatic selection method visible in readable result names."""
    name = metadata["model_type"]
    return (
        "auto_arima"
        if name == "arima" and metadata.get("selection") == "auto"
        else name
    )


def fitted_name(metadata: dict, fitted: dict) -> str:
    """Use the same naming convention for manual ARIMA and automatic ARIMA."""
    parts = [_model_prefix(metadata)]
    if "order" in fitted:
        parts.append("order-" + "-".join(map(str, fitted["order"])))
        parts.append(
            "seasonal-" + "-".join(map(str, fitted.get("seasonal_order", [0, 0, 0, 0])))
        )
    if metadata["use_exog"]:
        parts.append("exog")
    trend = metadata["parameters"].get("trend")
    if trend is not None:
        parts.append(f"trend-{trend}")
    return "__".join(_slug(part) for part in parts)


def evaluation_name(metadata: dict, windows: list[dict]) -> str:
    """Keep CV folds together while making every selected order explicit.

    A fixed-order model has one short name. If automatic orders vary, list the
    fold assignments instead of pretending that one order was fitted everywhere.
    """
    names = [fitted_name(metadata, window["model_fit"]) for window in windows]
    if not names:
        return fitted_name(metadata, {}) + "__unfitted"
    if len(set(names)) == 1:
        return names[0]

    parts = [_model_prefix(metadata)]
    if metadata["use_exog"]:
        parts.append("exog")
    for window in windows:
        fitted = window["model_fit"]
        parts.append(f"fold-{window['fold']}")
        parts.append("order-" + "-".join(map(str, fitted["order"])))
        parts.append("seasonal-" + "-".join(map(str, fitted["seasonal_order"])))
    return "__".join(parts)


def result_directory(root: Path, name: str) -> Path:
    """Split unusually long fold descriptions across path components.

    No order is truncated or replaced by an opaque identifier. Most CV plans fit
    in one component; longer plans retain all orders in a nested readable path.
    """
    components = []
    current = ""
    for part in name.split("__"):
        candidate = f"{current}__{part}" if current else part
        if len(candidate.encode()) > 220:
            components.append(current)
            current = part
        else:
            current = candidate
    return root.joinpath(*components, current)


def create_run_directory(config: DictConfig) -> Path:
    """Keep internal configuration identities out of user-facing result names."""
    directory = (
        Path(config.artifacts.directory).expanduser().resolve()
        / "configurations"
        / configuration_key(config)
    )
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    OmegaConf.save(config, directory / "config.yaml", resolve=True)
    return directory


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_observed_series(directory: Path, target: pd.Series) -> None:
    target.rename("actual").rename_axis("date").to_csv(directory / "series.csv")


def write_json(path: Path, value: dict) -> None:
    """Publish complete JSON documents and represent undefined numbers as null."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def rescore_saved_results(directory: Path, metrics: Sequence[str] | None) -> dict:
    """Apply a changed metric selection to saved forecasts without fitting again.

    Keep training details, selected orders, warnings, and run identities intact.
    CV summaries are scored from pooled predictions, not averaged fold scores.
    """
    report = json.loads((directory / "run.json").read_text())
    if metrics is None or list(metrics) == report["requested_metrics"]:
        return report

    metrics = list(metrics)
    evaluation = json.loads((directory / "metrics.json").read_text())
    predictions_path = directory / "predictions.csv"
    predictions = (
        pd.read_csv(predictions_path)
        if predictions_path.exists()
        else pd.DataFrame(columns=["evaluation", "fold", "actual", "predicted"])
    )

    scores = []
    for window in evaluation["windows"]:
        rows = predictions.loc[
            (predictions.evaluation == window["evaluation"])
            & (predictions.fold == window["fold"])
        ]
        values = evaluate(rows.actual.to_numpy(), rows.predicted.to_numpy(), metrics)
        for previous in report["requested_metrics"]:
            window.pop(previous, None)
        window.update(values)
        scores.append(
            {"evaluation": window["evaluation"], "fold": window["fold"], **values}
        )

    evaluation["summary"] = {
        scope: evaluate(rows.actual.to_numpy(), rows.predicted.to_numpy(), metrics)
        for scope, rows in predictions.groupby("evaluation")
    }
    report["requested_metrics"] = metrics
    snapshot = OmegaConf.load(directory / "config.yaml")
    snapshot.evaluation.metrics = metrics

    write_json(directory / "metrics.json", evaluation)
    pd.DataFrame(scores, columns=["evaluation", "fold", *metrics]).to_csv(
        directory / "metrics.csv", index=False
    )
    OmegaConf.save(snapshot, directory / "config.yaml", resolve=True)
    write_json(directory / "run.json", report)
    return report


def save_results(
    directory: Path,
    report: dict,
    predictions: list[pd.DataFrame],
    scores: list[dict],
) -> None:
    """Preserve completed windows even if a later fit fails."""
    report.setdefault("finished_at", datetime.now(UTC).isoformat())
    metrics = {"windows": report["windows"], "summary": report["summary"]}
    metadata = {key: value for key, value in report.items() if key not in metrics}
    write_json(directory / "metrics.json", metrics)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            directory / "predictions.csv", index=False
        )
    if scores:
        pd.DataFrame(scores).to_csv(directory / "metrics.csv", index=False)
    write_json(directory / "run.json", metadata)
