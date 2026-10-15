"""Choose the forecasting data loader described by an experiment config."""

from omegaconf import DictConfig, OmegaConf

from .base import BaseForecastDataLoader, ForecastData
from .csv import CSVForecastDataLoader

_LOADERS = {"csv": CSVForecastDataLoader}


def create_data_loader(config: DictConfig) -> BaseForecastDataLoader:
    """Keep configuration management outside concrete loader implementations."""
    options = OmegaConf.to_container(config, resolve=True)
    name = options["loader"]

    if name not in _LOADERS:
        raise ValueError(f"Unknown forecasting data loader: {name!r}")

    return _LOADERS[name](options)


def load_data(config: DictConfig) -> ForecastData:
    """Load a configured dataset through the task's loader contract."""
    return create_data_loader(config).load()
