"""Construct a model from a YAML manifest; leave fitting to the pipeline."""

from omegaconf import DictConfig, OmegaConf

from .arima import ARIMAForecaster
from .auto_arima import AutoARIMAForecaster
from .base import BaseForecaster
from .naive import NaiveForecaster

_REGISTRY = {
    "naive": NaiveForecaster,
    "arima": ARIMAForecaster,
    "auto_arima": AutoARIMAForecaster,
}


def create_forecaster(config: DictConfig) -> BaseForecaster:
    """Convert ``cfg.model`` to normal Python values and construct the named class."""
    manifest = OmegaConf.to_container(config, resolve=True)
    name = manifest["name"]

    if name not in _REGISTRY:
        raise ValueError(f"Unknown forecasting model: {name!r}")

    # Only params reach the constructor. Pipeline settings stay in the manifest.
    parameters = manifest.get("params", {})

    return _REGISTRY[name](**parameters)
