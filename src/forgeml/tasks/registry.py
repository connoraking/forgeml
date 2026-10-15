"""Dispatch CLI experiments to task-specific pipelines."""

from omegaconf import DictConfig

from .forecasting.eda import run_eda as explore_forecasting
from .forecasting.evaluation.reporting import rebuild_reports as rebuild_forecasting
from .forecasting.pipeline import run_experiment as run_forecasting

_TASKS = {"forecasting": run_forecasting}
_REPORTERS = {"forecasting": rebuild_forecasting}
_EXPLORERS = {"forecasting": explore_forecasting}


def run_task(config: DictConfig, launch_metadata: dict | None = None) -> dict:
    """Run a task; each task owns its loaders, model interface, and evaluation.

    This is a small entry-point registry. It does not impose a universal trainer
    or require future classification/vision tasks to use forecasting data shapes.
    """
    name = config.task

    if name not in _TASKS:
        raise ValueError(f"Unknown task: {name!r}. Available tasks: {list(_TASKS)}")

    return _TASKS[name](config, launch_metadata=launch_metadata)


def consolidate_task_results(config: DictConfig, refresh_plots: bool = False) -> None:
    """Delegate report rebuilding to the task that produced those artifacts."""
    name = config.task

    if name not in _REPORTERS:
        raise ValueError(f"No reporter registered for task: {name!r}")

    _REPORTERS[name](config, refresh_plots=refresh_plots)


def explore_task(config: DictConfig) -> dict:
    """Delegate EDA to a task's own data contract and diagnostic pipeline."""
    name = config.task

    if name not in _EXPLORERS:
        raise ValueError(f"No EDA pipeline registered for task: {name!r}")

    return _EXPLORERS[name](config)
