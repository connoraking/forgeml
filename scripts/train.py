"""Train using an experiment selected with --config-dir and --config-name."""

from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from forgeml.tasks.registry import run_task


@main(
    version_base="1.3",
    config_path=None,
    config_name="config",
)
def train(config: DictConfig) -> None:
    # Hydra-specific information stays at the CLI boundary. The library pipeline
    # also works directly with a composed DictConfig, without a Hydra runtime.
    hydra_config = HydraConfig.get()
    launch_metadata = {
        "model_choice": hydra_config.runtime.choices.get("model"),
        "overrides": list(hydra_config.overrides.task),
        "job_number": hydra_config.job.get("num"),
    }

    run_task(config, launch_metadata=launch_metadata)


if __name__ == "__main__":
    train()
