"""Numerical diagnostics for time series, independent of plotting and Hydra."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def describe_series(target: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Describe the prepared target and flag extremes without removing observations."""
    statistics = {
        key: float(value) if np.isfinite(value) else None
        for key, value in target.describe().items()
    }
    statistics.update(
        zeros=int(target.eq(0).sum()),
        negatives=int(target.lt(0).sum()),
        missing=int(target.isna().sum()),
    )

    # IQR flags describe the marginal distribution. Seasonal peaks may be flagged
    # too, so they are saved for inspection and never used to clean the series.
    lower_quartile, upper_quartile = target.quantile([0.25, 0.75])
    spread = upper_quartile - lower_quartile
    lower = lower_quartile - 1.5 * spread
    upper = upper_quartile + 1.5 * spread
    outliers = target.loc[target.lt(lower) | target.gt(upper)].to_frame("observed")
    statistics["iqr_flags"] = len(outliers)
    statistics["iqr_lower"] = float(lower)
    statistics["iqr_upper"] = float(upper)

    return statistics, outliers


def decompose_series(
    target: pd.Series, period: int | None, robust: bool
) -> tuple[pd.DataFrame | None, str | None]:
    """Fit additive STL when a regular series contains enough seasonal history."""
    if period is None:
        return None, "STL disabled: set eda.seasonal_period to a cycle length."

    frequency = target.index.freqstr
    if frequency is None and len(target) >= 3:
        frequency = pd.infer_freq(target.index)

    if frequency is None:
        return None, "STL skipped: dates do not establish a regular frequency."

    # Requiring two cycles is our conservative EDA policy, not an STL API limit.
    if len(target) < 2 * period:
        return None, f"STL skipped: need {2 * period} observations for two cycles."

    fitted = STL(target, period=period, robust=robust).fit()
    components = pd.DataFrame(
        {
            "observed": target,
            "trend": fitted.trend,
            "seasonal": fitted.seasonal,
            "residual": fitted.resid,
        },
        index=target.index,
    )

    return components, None


def autocorrelations(
    series: dict[str, pd.Series], max_lags: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Calculate coefficients and 95% confidence intervals for each diagnostic series.

    ACF can use more lags than PACF. Keep the actual limits in the CSV rather than
    requesting an invalid PACF on a short differenced series.
    """
    rows = []
    notes = {}

    for name, values in series.items():
        if len(values) < 4 or values.nunique() < 2:
            notes[name] = "ACF/PACF skipped: fewer than four nonconstant observations."
            continue

        for kind in ("acf", "pacf"):
            try:
                if kind == "acf":
                    result = acf(
                        values,
                        nlags=min(max_lags, len(values) - 1),
                        alpha=0.05,
                        result_object=True,
                    )
                    coefficients = result.acf
                else:
                    result = pacf(
                        values,
                        nlags=min(max_lags, len(values) // 2 - 1),
                        method="ywm",
                        alpha=0.05,
                        result_object=True,
                    )
                    coefficients = result.pacf
            except (ValueError, np.linalg.LinAlgError) as error:
                notes[name] = notes.get(name, "") + f"{kind.upper()} skipped: {error}; "
                continue

            intervals = result.confint
            if not np.isfinite(coefficients).all() or not np.isfinite(intervals).all():
                notes[name] = notes.get(name, "") + f"{kind.upper()} was nonfinite; "
                continue

            for lag, coefficient in enumerate(coefficients):
                rows.append(
                    {
                        "series": name,
                        "kind": kind,
                        "lag": lag,
                        "coefficient": float(coefficient),
                        "lower_95": float(intervals[lag, 0]),
                        "upper_95": float(intervals[lag, 1]),
                    }
                )

    columns = ["series", "kind", "lag", "coefficient", "lower_95", "upper_95"]

    return pd.DataFrame(rows, columns=columns), notes


def stationarity(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Report ADF and KPSS with their null hypotheses and any numerical limitations.

    These are diagnostics around a constant, not automatic model selection. ADF's
    null is a unit root; KPSS's null is level stationarity. Their p-values therefore
    have opposite interpretations, which stay explicit in the saved report.
    """
    rows = []

    for name, values in series.items():
        for test in ("adf", "kpss"):
            row = {
                "series": name,
                "test": test,
                "null_hypothesis": "unit root" if test == "adf" else "level stationary",
                "statistic": None,
                "p_value": None,
                "lags": None,
                "observations": len(values),
                "critical_1pct": None,
                "critical_5pct": None,
                "critical_10pct": None,
                "reject_null_at_5pct": None,
                "status": "skipped",
                "note": "",
            }

            if len(values) < 8 or values.nunique() < 2:
                row["note"] = "Need at least eight nonconstant observations."
                rows.append(row)
                continue

            # A numerical failure of an optional diagnostic should not discard
            # the series plot or the other reports. Input/file errors still raise.
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if test == "adf":
                        result = adfuller(values, autolag="AIC", result_object=True)
                    else:
                        result = kpss(
                            values, regression="c", nlags="auto", result_object=True
                        )

                if np.isfinite([result.statistic, result.pvalue]).all():
                    row.update(
                        statistic=float(result.statistic),
                        p_value=float(result.pvalue),
                        lags=int(result.lags),
                        critical_1pct=float(result.critical_values["1%"]),
                        critical_5pct=float(result.critical_values["5%"]),
                        critical_10pct=float(result.critical_values["10%"]),
                        reject_null_at_5pct=bool(result.pvalue < 0.05),
                        status="completed",
                        note="; ".join(str(item.message) for item in caught),
                    )
                else:
                    row["note"] = (
                        "Diagnostic returned a nonfinite statistic or p-value."
                    )

            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
                row["note"] = str(error)

            rows.append(row)

    return pd.DataFrame(rows)


def seasonal_profile(target: pd.Series, period: int) -> pd.DataFrame:
    """Overlay configured cycles starting at the series' first observation.

    These positions are relative to the first date, not calendar week numbers.
    Incomplete last cycles are kept so their observed positions remain visible.
    """
    positions = np.arange(len(target))

    return pd.DataFrame(
        {
            "date": target.index,
            "cycle": positions // period + 1,
            "position": positions % period + 1,
            "observed": target.to_numpy(),
        }
    )
