"""The prepared-data contract for the forecasting task."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastData:
    """One chronological target series and optional aligned future-known features."""

    target: pd.Series
    exog: pd.DataFrame | None = None

    def fingerprint(self, include_exog: bool = False) -> str:
        """Identify the prepared values used by a fit, including their dates.

        Cache reuse must notice edits to data even when its filename stays the
        same. A model using regressors also depends on their values and columns.
        """
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(self.target, index=True).to_numpy().tobytes()
        )
        if include_exog and self.exog is not None:
            digest.update(json.dumps(list(self.exog.columns)).encode())
            digest.update(
                pd.util.hash_pandas_object(self.exog, index=True).to_numpy().tobytes()
            )
        return digest.hexdigest()


class BaseForecastDataLoader(ABC):
    """Load forecasting observations, without fitting transforms on future targets.

    Other ML tasks should define their own data contracts and loaders. This
    contract describes univariate forecasting, rather than images or class labels.
    """

    @abstractmethod
    def load(self) -> ForecastData:
        """Return raw prepared observations; chronological splitting happens later."""
