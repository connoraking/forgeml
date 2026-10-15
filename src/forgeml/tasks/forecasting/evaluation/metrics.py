"""Forecast metrics, with explicit behavior for zero-valued sales."""

from collections.abc import Sequence

import numpy as np


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAE is expressed in the same units as the target."""
    return float(np.mean(np.abs(actual - predicted)))


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE gives larger errors more weight than MAE."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def mean_absolute_percentage_error(
    actual: np.ndarray, predicted: np.ndarray
) -> float | None:
    """MAPE in percent. Zero actuals are excluded; all-zero actuals return None."""
    nonzero = actual != 0

    if not nonzero.any():
        return None

    relative_error = np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])

    return float(100 * np.mean(relative_error))


def symmetric_mean_absolute_percentage_error(
    actual: np.ndarray, predicted: np.ndarray
) -> float:
    """SMAPE in percent, from 0 to 200. An actual/predicted zero pair scores zero."""
    denominator = np.abs(actual) + np.abs(predicted)
    relative_error = np.zeros_like(denominator, dtype=float)

    np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=relative_error,
        where=denominator != 0,
    )

    return float(100 * np.mean(relative_error))


def weighted_absolute_percentage_error(
    actual: np.ndarray, predicted: np.ndarray
) -> float | None:
    """WAPE in percent. Undefined when all actuals are zero."""
    total = np.sum(np.abs(actual))

    if total == 0:
        return None

    return float(100 * np.sum(np.abs(actual - predicted)) / total)


METRICS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mape": mean_absolute_percentage_error,
    "smape": symmetric_mean_absolute_percentage_error,
    "wape": weighted_absolute_percentage_error,
}


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float | None]:
    """Check forecast alignment once, then calculate the requested metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.shape != actual.shape or actual.size == 0:
        raise ValueError("Actual and forecast must be matching non-empty 1D arrays.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and forecast must contain finite values.")

    unknown = set(metrics) - METRICS.keys()

    if unknown:
        raise ValueError(
            f"Unknown metrics: {sorted(unknown)}. Choose from {list(METRICS)}."
        )

    return {name: METRICS[name](actual, predicted) for name in metrics}
