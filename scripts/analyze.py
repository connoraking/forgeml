"""Rebuild aggregate metric CSVs and forecast plots from all saved experiment fits."""

from hydra import main
from omegaconf import DictConfig

from forgeml.tasks.registry import consolidate_task_results


@main(
    version_base="1.3",
    config_path=None,
    config_name="config",
)
def analyze(config: DictConfig) -> None:
    # Analysis uses saved forecasts and metrics. It needs neither the original
    # dataset nor model fitting, and ignores today's training family switches.
    consolidate_task_results(config)
    print(f"Updated metric CSVs and plots: {config.results.directory}")


if __name__ == "__main__":
    analyze()
