"""Run exploratory analysis using the forecasting experiment's prepared data."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..dataloaders import load_data
from .diagnostics import (
    autocorrelations,
    decompose_series,
    describe_series,
    seasonal_profile,
    stationarity,
)
from .plots import (
    plot_correlations,
    plot_differences,
    plot_distribution,
    plot_rolling,
    plot_seasonality,
    plot_series,
    plot_stl,
)

# Only these files belong to this pipeline. Refresh them without deleting other
# files a user may have placed in the output directory.
_TABLES = (
    "series.csv",
    "summary.csv",
    "outliers.csv",
    "rolling.csv",
    "differences.csv",
    "autocorrelation.csv",
    "stationarity.csv",
    "stl.csv",
    "seasonality.csv",
)
_PLOTS = (
    "series.png",
    "rolling.png",
    "distribution.png",
    "differences.png",
    "acf.png",
    "pacf.png",
    "stl.png",
    "seasonality.png",
)


def run_eda(config: DictConfig) -> dict:
    """Explore the selected target without fitting a forecasting model.

    EDA uses the same loader, filters, and aggregation as training. Its default
    scope is the full prepared history; optional dates restrict the explored span.
    Model manifests and CV settings do not affect these descriptive diagnostics.
    """
    options = OmegaConf.to_container(config.eda, resolve=True)
    period = options["seasonal_period"]
    window = options["rolling_window"]
    max_lags = options["max_lags"]
    difference_lag = options["difference_lag"]

    # Invalid analysis settings are genuine configuration errors. Short history
    # and constant data, on the other hand, are handled by individual diagnostics.
    counts = (window, max_lags, difference_lag)
    if (
        any(type(value) is not int for value in counts)
        or window < 2
        or max_lags < 1
        or difference_lag < 1
        or (period is not None and (type(period) is not int or period < 2))
    ):
        raise ValueError(
            "EDA needs integer rolling_window >= 2, max_lags/difference_lag >= 1, "
            "and seasonal_period >= 2 or null."
        )

    directory = Path(options["directory"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    # Refresh the existing EDA output rather than inventing runs or attempt names.
    # In particular, a skipped STL must not leave an earlier decomposition CSV.
    for filename in (*_TABLES, "summary.json", "config.yaml"):
        (directory / filename).unlink(missing_ok=True)
    for filename in _PLOTS:
        (directory / "plots" / filename).unlink(missing_ok=True)

    OmegaConf.save(config, directory / "config.yaml", resolve=True)
    target = load_data(config.data).target
    target = target.loc[options.get("start_date") : options.get("end_date")]

    if target.empty:
        raise ValueError("The selected EDA date range contains no observations.")

    statistics, outliers = describe_series(target)
    components, stl_note = decompose_series(target, period, options["stl_robust"])
    notes = {}

    if stl_note:
        notes["stl"] = stl_note
    if len(target) < window:
        notes["rolling"] = "History is shorter than the configured rolling window."

    # These differences are diagnostic views. They never alter the training data
    # or prescribe a model's differencing order.
    differences = {"difference": target.diff(difference_lag).dropna()}
    difference_lags = {"difference": difference_lag}
    if period is not None:
        differences["seasonal_difference"] = target.diff(period).dropna()
        difference_lags["seasonal_difference"] = period

    diagnostic_series = {"original": target, **differences}
    if components is not None:
        diagnostic_series["stl_residual"] = components.residual

    correlations, correlation_notes = autocorrelations(diagnostic_series, max_lags)
    notes.update(correlation_notes)
    tests = stationarity(diagnostic_series)
    rolling = pd.DataFrame(
        {"mean": target.rolling(window).mean(), "std": target.rolling(window).std()}
    )
    cycles = seasonal_profile(target, period) if period is not None else None

    # CSVs retain the plotted numbers, including empty headers for skipped
    # diagnostics. Leading unavailable rolling/difference values are blank cells.
    target.rename("observed").rename_axis("date").to_csv(directory / "series.csv")
    outliers.rename_axis("date").to_csv(directory / "outliers.csv")
    rolling.rename_axis("date").to_csv(directory / "rolling.csv")
    pd.DataFrame(differences, index=target.index).rename_axis("date").to_csv(
        directory / "differences.csv"
    )
    correlations.to_csv(directory / "autocorrelation.csv", index=False)
    tests.to_csv(directory / "stationarity.csv", index=False)

    if components is not None:
        components.rename_axis("date").to_csv(directory / "stl.csv")
    if cycles is not None:
        cycles.to_csv(directory / "seasonality.csv", index=False)

    title = f"{config.experiment.name} · {target.name}"
    filters = config.data.get("filters", {})
    if filters:
        title += " · " + ", ".join(f"{key}={value}" for key, value in filters.items())

    plots = directory / "plots"
    plot_series(target, outliers, title, plots)
    plot_rolling(target, rolling, window, plots)
    plot_distribution(target, plots)
    plot_differences(differences, difference_lags, plots)
    for kind in ("acf", "pacf"):
        plot_correlations(
            correlations, diagnostic_series, correlation_notes, kind, plots
        )
    plot_stl(components, period, stl_note, plots)
    plot_seasonality(target, cycles, plots)

    frequency = target.index.freqstr
    if frequency is None and len(target) >= 3:
        frequency = pd.infer_freq(target.index)

    metadata = {
        "experiment": config.experiment.name,
        "target": str(target.name),
        "points": len(target),
        "start": str(target.index[0]),
        "end": str(target.index[-1]),
        "frequency": frequency,
        "seasonal_period": period,
        "rolling_window": window,
        "difference_lag": difference_lag,
    }
    report = {
        **metadata,
        "generated_at": datetime.now(UTC).isoformat(),
        "statistics": statistics,
        "notes": notes,
        "plots": [f"plots/{filename}" for filename in _PLOTS],
    }
    pd.DataFrame([{**metadata, **statistics}]).to_csv(
        directory / "summary.csv", index=False
    )
    (directory / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    print(
        f"EDA: {len(target)} observations, {target.index[0]} through {target.index[-1]}"
    )
    print(f"Plots and reports: {directory}")
    for note in notes.values():
        print(note)

    return report
