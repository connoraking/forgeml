"""Detailed run artifacts, kept separate from experiment-level comparison reports."""

import hashlib
import json
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _slug(value: object) -> str:
    """Turn a label into a single readable directory component."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("._-") or "unnamed"


def describe_model(model: DictConfig) -> str:
    """Show explicit ARIMA orders in the name, rather than hiding them behind an ID."""
    parts = [model.get("label", model.name)]
    parameters = model.get("params", {})

    if "order" in parameters:
        parts.append("order-" + "-".join(map(str, parameters["order"])))

    seasonal = parameters.get("seasonal_order", [0, 0, 0, 0])
    if any(seasonal[:3]):
        parts.append("seasonal-" + "-".join(map(str, seasonal)))

    search = parameters.get("search")
    if search:
        parts.append(f"max-p{search.get('max_p', 3)}-q{search.get('max_q', 3)}")

    return "__".join(_slug(part) for part in parts)


def _configuration_key(config: DictConfig) -> str:
    """Identify the configured forecast, independently of a sweep or execution.

    Changing the requested metrics refreshes the same forecast's output. Changing
    model parameters, data preparation, or split settings defines another variant.
    Dataset contents are deliberately excluded: replacing a CSV refreshes its run.
    """
    settings = OmegaConf.to_container(config, resolve=True)
    evaluation = dict(settings["evaluation"])
    evaluation.pop("metrics", None)
    identity = {
        "task": settings["task"],
        "model": settings["model"],
        "data": settings["data"],
        "evaluation": evaluation,
        "name": settings.get("run", {}).get("name"),
    }

    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _run_name(config: DictConfig) -> str:
    """Describe model, series, and evaluation settings without execution counters."""
    run = config.get("run", {})

    name = run.get("name")
    if name:
        stem = _slug(name)
    else:
        filters = config.data.get("filters", {})
        scope = "-".join(f"{key}-{value}" for key, value in sorted(filters.items()))
        parts = [describe_model(config.model), _slug(scope or "all-data")]

        if config.evaluation.holdout.enabled:
            size = config.evaluation.holdout.test_size
            holdout = (
                f"{size * 100:g}pct" if isinstance(size, float) else f"{size}steps"
            )
            parts.append(f"holdout-{holdout}")

        if config.evaluation.cv.enabled:
            cv = config.evaluation.cv
            parts.append(f"cv-{cv.n_splits}x{cv.horizon}")
            if cv.get("step") not in (None, cv.horizon):
                parts.append(f"step-{cv.step}")

        stem = "__".join(parts)

    return stem[:200]


def create_run_directory(config: DictConfig) -> Path:
    """Replace a configuration's previous output, then save its config snapshot."""
    root = Path(config.artifacts.directory).expanduser().resolve() / "runs"
    directory = root / _run_name(config)
    key = _configuration_key(config)

    # The readable name highlights common settings, but cannot describe every
    # future model parameter. Separate configs with matching names rather than
    # silently overwriting a sweep variant (for example, a different ARIMA trend).
    variant = root / f"{directory.name}__config-{key[:12]}"
    snapshot = directory / "config.yaml"
    if variant.exists():
        directory = variant
    elif directory.exists():
        if not snapshot.exists() or _configuration_key(OmegaConf.load(snapshot)) != key:
            directory = variant

    # Remove the whole previous output so a failed rerun cannot retain successful
    # predictions or plots, and a changed metric list cannot leave stale files.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    # Save before loading/fitting so a failed run still has its exact configuration.
    OmegaConf.save(config, directory / "config.yaml", resolve=True)

    return directory


def _json_safe(value):
    """JSON has no NaN/Infinity. Treat undefined diagnostics as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def save_observed_series(directory: Path, target: pd.Series) -> None:
    """Keep the prepared observations so plots can show training history later."""
    target.rename("actual").rename_axis("date").to_csv(directory / "series.csv")


def _write_json(path: Path, value: dict) -> None:
    """Replace the file only after writing a complete JSON document."""
    temporary = path.with_suffix(".tmp")

    with temporary.open("w") as file:
        json.dump(_json_safe(value), file, indent=2, allow_nan=False)
        file.write("\n")

    temporary.replace(path)


def save_results(
    directory: Path,
    report: dict,
    predictions: list[pd.DataFrame],
    scores: list[dict],
) -> None:
    """Persist results and metadata, including completed folds from a failed run."""
    report.setdefault("finished_at", datetime.now(UTC).isoformat())

    # run.json is the index of this artifact. metrics.json contains evaluation data.
    metrics = {"windows": report["windows"], "summary": report["summary"]}
    metadata = {key: value for key, value in report.items() if key not in metrics}

    _write_json(directory / "metrics.json", metrics)

    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            directory / "predictions.csv", index=False
        )

    if scores:
        pd.DataFrame(scores).to_csv(directory / "metrics.csv", index=False)

    # Publishing metadata last prevents consolidation from reading an unfinished run.
    _write_json(directory / "run.json", metadata)
