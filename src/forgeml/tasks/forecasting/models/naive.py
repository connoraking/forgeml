"""A readable last-value baseline for comparison with fitted models."""

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .inputs import check_history, check_horizon


class NaiveForecaster(BaseForecaster):
    """Repeat the final historical value. Optional features are unused."""

    def __init__(self) -> None:
        self.last_observed: float | None = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        check_history(history)

        self.last_observed = float(history.iloc[-1])

    def predict(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        if self.last_observed is None:
            raise RuntimeError("Fit the model before forecasting.")

        check_horizon(horizon)

        return np.full(horizon, self.last_observed, dtype=float)
