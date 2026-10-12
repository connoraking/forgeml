"""The few input checks that protect every forecaster from misleading results."""

import numpy as np
import pandas as pd


def check_history(history: pd.Series) -> None:
    """An empty or nonfinite target cannot produce a meaningful fitted model."""
    if history.empty or not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("History must be non-empty and contain finite numeric values.")


def check_horizon(horizon: int) -> None:
    """Keep invalid forecast requests away from the underlying model libraries."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Forecast horizon must be a positive integer.")


def check_training_exog(history: pd.Series, exog: pd.DataFrame | None) -> None:
    """Features must describe the same observations as the historical target."""
    if exog is None:
        return

    if not exog.index.equals(history.index) or exog.columns.has_duplicates:
        raise ValueError("Exog needs unique columns and an index matching history.")

    if exog.shape[1] == 0 or not np.isfinite(exog.to_numpy(dtype=float)).all():
        raise ValueError("Exog must contain finite numeric features.")


def prepare_future_exog(
    future_exog: pd.DataFrame | None,
    columns: pd.Index | None,
    horizon: int,
) -> pd.DataFrame | None:
    """Match forecast features to training features, including their column order."""
    if columns is None:
        return None

    if future_exog is None or len(future_exog) != horizon:
        raise ValueError(f"Future exog is required and must have {horizon} rows.")

    if future_exog.columns.has_duplicates or set(future_exog.columns) != set(columns):
        raise ValueError("Future exog columns must match the training features.")

    # Reordered CSV columns are harmless; align them instead of rejecting them.
    aligned = future_exog.loc[:, columns].astype(float)

    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Future exog must contain finite numeric values.")

    return aligned
