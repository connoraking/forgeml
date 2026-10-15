"""The prepared-data contract for the forecasting task."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastData:
    """One chronological target series and optional aligned future-known features."""

    target: pd.Series
    exog: pd.DataFrame | None = None


class BaseForecastDataLoader(ABC):
    """Load forecasting observations, without fitting transforms on future targets.

    Other ML tasks should define their own data contracts and loaders. This
    contract describes univariate forecasting, rather than images or class labels.
    """

    @abstractmethod
    def load(self) -> ForecastData:
        """Return raw prepared observations; chronological splitting happens later."""
