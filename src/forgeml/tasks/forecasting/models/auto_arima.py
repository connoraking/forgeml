"""Automatic ARIMA order selection through pmdarima."""

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima import ARIMA

from .base import BaseForecaster
from .inputs import (
    check_history,
    check_horizon,
    check_training_exog,
    prepare_future_exog,
)


class AutoARIMAForecaster(BaseForecaster):
    """Search orders during fit using only the supplied training history.

    ``search`` holds pmdarima options such as max_p, max_q, seasonal, m, and
    information_criterion. Keeping those options in YAML makes experiments clear.
    """

    def __init__(self, search: dict | None = None) -> None:
        self.search = {
            "start_p": 0,
            "start_q": 0,
            "max_p": 3,
            "max_q": 3,
            "seasonal": False,
            "stepwise": True,
            "suppress_warnings": True,
            "error_action": "warn",
            **(search or {}),
        }

        self.results: ARIMA | None = None
        self._exog_columns: pd.Index | None = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        check_history(history)
        check_training_exog(history, exog)

        self.results = None
        self._exog_columns = None

        # CV creates a new instance for each fold, so order selection never sees
        # that fold's validation target or the experiment's final holdout.
        self.results = auto_arima(
            y=history.to_numpy(dtype=float),
            X=exog.to_numpy(dtype=float) if exog is not None else None,
            **self.search,
        )

        self._exog_columns = exog.columns.copy() if exog is not None else None

    def predict(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Fit the model before forecasting.")

        check_horizon(horizon)
        features = prepare_future_exog(future_exog, self._exog_columns, horizon)

        return np.asarray(
            self.results.predict(n_periods=horizon, X=features), dtype=float
        )

    def describe_fit(self) -> dict:
        """Record the orders selected by the search, not just its search bounds."""
        return {
            "order": self.results.order,
            "seasonal_order": self.results.seasonal_order,
            "aic": float(self.results.aic()),
            "converged": bool(
                self.results.arima_res_.mle_retvals.get("converged", True)
            ),
        }
