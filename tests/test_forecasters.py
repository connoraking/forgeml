"""Model behavior and experiment-specific Hydra manifests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from forgeml.tasks.forecasting.factory import create_forecaster
from forgeml.tasks.forecasting.models import (
    ARIMAForecaster,
    AutoARIMAForecaster,
    BaseForecaster,
    NaiveForecaster,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experimentation/example/configs"


@pytest.fixture
def history() -> pd.Series:
    # A deterministic fixture for adapter tests; experiments always load real CSV data.
    rng = np.random.default_rng(42)
    return pd.Series(
        10 + rng.normal(size=60), index=pd.date_range("2020-01-01", periods=60)
    )


def test_base_forecaster_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseForecaster()


@pytest.mark.parametrize(
    "model", [NaiveForecaster, ARIMAForecaster, AutoARIMAForecaster]
)
def test_models_require_a_fit_and_positive_horizon(model, history) -> None:
    forecaster = model()

    with pytest.raises(RuntimeError):
        forecaster.predict(2)

    forecaster.fit(history)

    with pytest.raises(ValueError):
        forecaster.predict(0)


def test_naive_repeats_last_value(history) -> None:
    model = NaiveForecaster()
    model.fit(history)

    np.testing.assert_array_equal(model.predict(3), [history.iloc[-1]] * 3)

    with pytest.raises(ValueError):
        model.fit(pd.Series([1.0, np.nan]))


@pytest.mark.parametrize("model_type", [ARIMAForecaster, AutoARIMAForecaster])
def test_arima_exog_requires_aligned_features(model_type, history) -> None:
    exog = pd.DataFrame(
        {"holiday": np.arange(len(history)) % 2, "weekday": history.index.dayofweek},
        index=history.index,
    )
    model = model_type()

    with pytest.raises(ValueError, match="index"):
        model.fit(history, exog.set_axis(pd.RangeIndex(len(history))))

    model.fit(history, exog)

    with pytest.raises(ValueError, match="required"):
        model.predict(2)

    future = pd.DataFrame({"holiday": [0, 1], "weekday": [3, 4]})
    forecast = model.predict(2, future)

    assert forecast.shape == (2,)
    assert np.isfinite(forecast).all()

    # Column ordering is normalized, so this should produce identical forecasts.
    np.testing.assert_allclose(
        forecast, model.predict(2, future[["weekday", "holiday"]])
    )

    with pytest.raises(ValueError, match="columns"):
        model.predict(2, future.rename(columns={"weekday": "different"}))


def test_auto_arima_records_selected_order(history) -> None:
    model = AutoARIMAForecaster(search={"d": 0, "max_p": 1, "max_q": 1})
    model.fit(history)

    details = model.describe_fit()

    assert details["order"][0] <= 1
    assert details["order"][1] == 0
    assert details["order"][2] <= 1
    assert np.isfinite(model.predict(4)).all()


def test_factory_constructs_without_fitting() -> None:
    manifest = OmegaConf.create({"name": "arima", "params": {"order": [1, 1, 0]}})
    model = create_forecaster(manifest)

    assert model.order == (1, 1, 0)
    assert model.results is None

    with pytest.raises(ValueError, match="Unknown"):
        create_forecaster(OmegaConf.create({"name": "made_up"}))


@pytest.mark.parametrize(
    "choice",
    ["naive", "arima", "arima_seasonal", "arima_with_exog", "auto_arima"],
)
def test_manifests_compose_under_model(choice) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="config", overrides=[f"model={choice}"])

    assert "arima" not in config.model
    assert isinstance(create_forecaster(config.model), BaseForecaster)


def test_hydra_parameter_override() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=["model=arima", "model.params.order=[3,1,2]"],
        )

    assert create_forecaster(config.model).order == (3, 1, 2)
