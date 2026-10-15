"""ARIMA forecasting with fixed orders or automatic parameter selection."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima import ARIMA as SelectedARIMA
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from .base import BaseForecaster
from .inputs import (
    check_history,
    check_horizon,
    check_training_exog,
    prepare_future_exog,
)


class ARIMAForecaster(BaseForecaster):
    """Fit the same model family using either explicit orders or an order search.

    ``selection`` chooses how parameters are obtained. Fixed selection uses
    ``order``, ``seasonal_order``, and ``trend``; auto selection uses ``search``
    options for pmdarima, including seasonal search bounds and period ``m``.
    """

    def __init__(
        self,
        order: Sequence[int] = (1, 0, 0),
        seasonal_order: Sequence[int] = (0, 0, 0, 0),
        trend: str | None = None,
        selection: str = "fixed",
        search: dict | None = None,
    ) -> None:
        if selection not in {"fixed", "auto"}:
            raise ValueError("ARIMA selection must be 'fixed' or 'auto'.")

        # Statsmodels already validates order combinations; do not duplicate that logic.
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.selection = selection
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

        self.results: ARIMAResultsWrapper | SARIMAXResultsWrapper | None = None
        self._selected_model: SelectedARIMA | None = None
        self._exog_columns: pd.Index | None = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        check_history(history)
        check_training_exog(history, exog)

        # Clear an old fit before refitting, so a failed fit cannot reuse stale results.
        self.results = None
        self._selected_model = None
        self._exog_columns = None

        if self.selection == "auto":
            # Selection sees only this fit's training history. The pipeline creates
            # a fresh ARIMA instance for every CV fold and for the final holdout.
            self._selected_model = auto_arima(
                y=history.to_numpy(dtype=float),
                X=exog.to_numpy(dtype=float) if exog is not None else None,
                **self.search,
            )
            self.results = self._selected_model.arima_res_
        else:
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

        # Preserve the selected estimator's prediction handling, including any
        # differencing settings supplied through the automatic search options.
        if self._selected_model is not None:
            forecast = self._selected_model.predict(n_periods=horizon, X=features)
        else:
            forecast = self.results.forecast(steps=horizon, exog=features)

        return np.asarray(forecast, dtype=float)

    def describe_fit(self) -> dict:
        """Small, JSON-friendly fit details for the experiment report."""
        order = self.order
        seasonal_order = self.seasonal_order
        if self._selected_model is not None:
            order = self._selected_model.order
            seasonal_order = self._selected_model.seasonal_order

        return {
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": float(self.results.aic),
            "converged": bool(self.results.mle_retvals.get("converged", True)),
        }
