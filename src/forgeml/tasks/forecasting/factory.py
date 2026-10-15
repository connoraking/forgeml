"""Construct a model from a YAML manifest; leave fitting to the pipeline."""

from omegaconf import DictConfig, OmegaConf

from .models.arima import ARIMAForecaster
from .models.base import BaseForecaster
from .models.naive import NaiveForecaster

_REGISTRY = {
    "naive": NaiveForecaster,
    "arima": ARIMAForecaster,
}


def create_forecaster(config: DictConfig) -> BaseForecaster:
    """Convert ``cfg.model`` to normal Python values and construct the named class."""
    manifest = OmegaConf.to_container(config, resolve=True)
    name = manifest["name"]

    if name not in _REGISTRY:
        raise ValueError(f"Unknown forecasting model: {name!r}")

    parameters = dict(manifest.get("params", {}))
    if name == "arima":
        # Automatic order selection is an ARIMA setting, not another model type.
        parameters["selection"] = manifest.get("selection", "fixed")
        parameters["search"] = manifest.get("search", {})

    return _REGISTRY[name](**parameters)
