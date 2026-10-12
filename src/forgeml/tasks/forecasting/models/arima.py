"""Explicit ARIMA orders using statsmodels, including seasonal and exogenous terms."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

from .base import BaseForecaster
from .inputs import (
    check_history,
    check_horizon,
    check_training_exog,
    prepare_future_exog,
)


class ARIMAForecaster(BaseForecaster):
    """Keep library details behind the shared fit/predict interface."""

    def __init__(
        self,
        order: Sequence[int] = (1, 0, 0),
        seasonal_order: Sequence[int] = (0, 0, 0, 0),
        trend: str | None = None,
    ) -> None:
        # Statsmodels already validates order combinations; do not duplicate that logic.
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend

        self.results: ARIMAResultsWrapper | None = None
        self._exog_columns: pd.Index | None = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        check_history(history)
        check_training_exog(history, exog)

        # Clear an old fit before refitting, so a failed fit cannot reuse stale results.
        self.results = None
        self._exog_columns = None

        model = ARIMA(
            endog=history.astype(float),
            exog=exog.astype(float) if exog is not None else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )

        self.results = model.fit()
        self._exog_columns = exog.columns.copy() if exog is not None else None

    def predict(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Fit the model before forecasting.")

        check_horizon(horizon)
        features = prepare_future_exog(future_exog, self._exog_columns, horizon)

        forecast = self.results.forecast(steps=horizon, exog=features)

        return np.asarray(forecast, dtype=float)

    def describe_fit(self) -> dict:
        """Small, JSON-friendly fit details for the experiment report."""
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": float(self.results.aic),
            "converged": bool(self.results.mle_retvals.get("converged", True)),
        }
