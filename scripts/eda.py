"""Generate exploratory plots and diagnostics for the configured experiment."""

from hydra import main
from omegaconf import DictConfig

from forgeml.tasks.registry import explore_task


@main(
    version_base="1.3",
    config_path=None,
    config_name="config",
)
def explore(config: DictConfig) -> None:
    explore_task(config)


if __name__ == "__main__":
    explore()
