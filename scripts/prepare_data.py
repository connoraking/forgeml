"""Materialize the raw data source configured by an experiment."""

from hydra import main
from omegaconf import DictConfig

from forgeml.data import prepare_data


@main(
    version_base="1.3",
    config_path=None,
    config_name="config",
)
def prepare(config: DictConfig) -> None:
    directory = prepare_data(config.data.source)
    print(f"Data ready at {directory}")


if __name__ == "__main__":
    prepare()
