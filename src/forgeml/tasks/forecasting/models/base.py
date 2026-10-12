"""The only contract the pipeline needs from a forecasting model."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Forecast one target series, optionally using features known in advance."""

    @abstractmethod
    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Learn from historical observations. This method does not evaluate."""

    @abstractmethod
    def predict(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Return exactly one predicted value for each requested future step."""

    def describe_fit(self) -> dict:
        """Optional fit diagnostics to save with results; models can extend this."""
        return {}
