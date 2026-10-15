"""Expand model manifests and add new fits to an experiment's saved results."""

import json
from itertools import product
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .dataloaders import ForecastData, load_data
from .evaluation.reporting import consolidate_reports
from .evaluation.results import configuration_key, describe_model, rescore_saved_results
from .pipeline import run_experiment


def expand_experiments(config: DictConfig) -> list[DictConfig]:
    """The main config chooses families; each manifest owns its own sweep.

    Grid keys address the model definition, for example params.order or use_exog.
    Values are ordinary YAML lists. A list-valued constructor argument therefore
    uses a list of lists, with one complete order per choice.
    """
    common = OmegaConf.to_container(config, resolve=True)
    manifests = common.pop("models")
    experiments = []
    seen = set()

    for family, definition in manifests.items():
        definition = dict(definition)
        if not definition.pop("enabled"):
            continue

        sweep = definition.pop("sweep")
        parameters = sweep.get("parameters", {}) if sweep["enabled"] else {}
        variants = sweep.get("variants", [{}]) if sweep["enabled"] else [{}]
        if any(
            not isinstance(values, list) or not values for values in parameters.values()
        ):
            raise ValueError(f"{family}: sweep parameters must be nonempty YAML lists.")

        # itertools.product with no axes produces one default configuration.
        for variant in variants:
            for choices in product(*parameters.values()):
                model = OmegaConf.merge(definition, variant)
                current = OmegaConf.create({**common, "model": model})
                for key, value in zip(parameters, choices, strict=True):
                    OmegaConf.update(current.model, key, value, merge=True)

                # Overlapping hand-picked cases and grids should fit a case once.
                key = configuration_key(current)
                if key not in seen:
                    seen.add(key)
                    experiments.append(current)

    return experiments


def _completed_report(
    config: DictConfig, datasets: dict[str, ForecastData]
) -> dict | None:
    """Reuse complete outputs only while the fit settings and input still match."""
    directory = (
        Path(config.artifacts.directory).expanduser().resolve()
        / "configurations"
        / configuration_key(config)
    )
    required = (
        "run.json",
        "config.yaml",
        "metrics.json",
        "predictions.csv",
        "series.csv",
    )
    if not all((directory / name).is_file() for name in required):
        return None

    report = json.loads((directory / "run.json").read_text())
    if report["status"] != "completed":
        return None

    # Load each distinct dataset once for cache checks, rather than once per
    # model. Compare prepared values so changes to targets or regressors count.
    data_key = json.dumps(
        OmegaConf.to_container(config.data, resolve=True), sort_keys=True
    )
    if data_key not in datasets:
        datasets[data_key] = load_data(config.data)
    fingerprint = datasets[data_key].fingerprint(config.model.use_exog)
    if report.get("input_fingerprint") != fingerprint:
        return None
    # Metrics do not change a fit's identity. Score additional metrics from the
    # stored forecasts instead of retraining a completed model for a new score.
    return rescore_saved_results(directory, config.evaluation.metrics)


def run_experiments(config: DictConfig, launch_metadata: dict | None = None) -> dict:
    """Train new/unfinished cases, retain earlier fits, and refresh all reports.

    Removing a variant or disabling a family affects what trains today. Its
    saved fits remain available to analysis. execution.rerun=true explicitly
    replaces the selected cases, without deleting other saved configurations.
    """
    experiments = expand_experiments(config)
    rerun = config.get("execution", {}).get("rerun", False)
    datasets = {}
    reports = []
    skipped = 0
    trained = 0
    print(f"Selected {len(experiments)} model configurations")

    try:
        for current in experiments:
            previous = None if rerun else _completed_report(current, datasets)
            if previous is not None:
                print(f"Skipping completed: {describe_model(current.model)}")
                reports.append(previous)
                skipped += 1
                continue

            reports.append(
                run_experiment(current, launch_metadata, refresh_reports=False)
            )
            trained += 1
    finally:
        # Include every saved fit, even configurations absent from today's
        # manifest. A failed run still leaves its completed windows inspectable.
        consolidate_reports(
            config.results.directory,
            config.artifacts.directory,
            config.evaluation.metrics,
        )

    print(f"Trained {trained}; reused {skipped} completed configurations")
    print(f"Results: {config.results.directory}")
    return {
        "status": "completed",
        "runs": reports,
        "trained": trained,
        "skipped": skipped,
    }
