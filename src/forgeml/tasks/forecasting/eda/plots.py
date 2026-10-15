"""Readable standalone forecasting EDA plots, rendered without a GUI."""

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

_BLUE = "#2563eb"
_ORANGE = "#ea580c"
_LABELS = {
    "original": "Original series",
    "difference": "Differenced series",
    "seasonal_difference": "Seasonally differenced series",
    "stl_residual": "STL residuals",
}


def _style(axis, ylabel: str = "") -> None:
    """Use plain values and light grids so large sales numbers stay readable."""
    axis.set_ylabel(ylabel)
    maximum = max(abs(value) for value in axis.get_ylim())
    pattern = "{x:,.0f}" if maximum >= 100 else "{x:,.2f}"
    axis.yaxis.set_major_formatter(StrMethodFormatter(pattern))
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _time_axis(axis) -> None:
    locator = AutoDateLocator(minticks=4, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=150, facecolor="white")
    figure.clear()

    return path


def plot_series(
    target: pd.Series, outliers: pd.DataFrame, title: str, directory: Path
) -> Path:
    """Show every prepared observation and flag distribution extremes in context."""
    figure = Figure(figsize=(12, 4.5), layout="constrained")
    axis = figure.subplots()
    axis.plot(target.index, target, color=_BLUE, linewidth=1.4, label="Observed")

    if not outliers.empty:
        axis.scatter(
            outliers.index,
            outliers.observed,
            s=32,
            color=_ORANGE,
            zorder=3,
            label="Beyond 1.5 × IQR",
        )

    axis.set_title(f"{title} · full selected series", loc="left")
    axis.legend(frameon=False)
    _style(axis, str(target.name))
    _time_axis(axis)

    return _save(figure, directory / "series.png")


def plot_rolling(
    target: pd.Series, rolling: pd.DataFrame, window: int, directory: Path
) -> Path:
    """Show trailing mean and variability without using centered future values."""
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 1, sharex=True)
    axes[0].plot(target.index, target, color="#94a3b8", label="Observed", linewidth=1)
    axes[0].plot(rolling.index, rolling["mean"], color=_BLUE, label="Trailing mean")
    axes[0].fill_between(
        rolling.index,
        rolling["mean"] - rolling["std"],
        rolling["mean"] + rolling["std"],
        color=_BLUE,
        alpha=0.12,
        label="Mean ± one standard deviation",
    )
    axes[0].set_title(f"Trailing statistics · {window} observations", loc="left")
    axes[0].legend(frameon=False, ncols=2)
    axes[1].plot(rolling.index, rolling["std"], color=_ORANGE)
    axes[1].set_title("Rolling standard deviation", loc="left")

    for axis in axes:
        _style(axis, str(target.name))
        _time_axis(axis)

    return _save(figure, directory / "rolling.png")


def plot_distribution(target: pd.Series, directory: Path) -> Path:
    figure = Figure(figsize=(11, 4), layout="constrained")
    axes = figure.subplots(1, 2, width_ratios=[3, 1])
    axes[0].hist(target, bins="auto", color=_BLUE, alpha=0.85, edgecolor="white")
    axes[0].set_title("Target distribution", loc="left")
    axes[0].set_xlabel(str(target.name))
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    _style(axes[0], "Count")
    axes[1].boxplot(target, tick_labels=[str(target.name)])
    axes[1].set_title("Median, quartiles and extremes", loc="left", fontsize=10)
    _style(axes[1], str(target.name))

    return _save(figure, directory / "distribution.png")


def plot_differences(
    differences: dict[str, pd.Series], lags: dict[str, int], directory: Path
) -> Path:
    figure = Figure(figsize=(12, 3.5 * len(differences)), layout="constrained")
    axes = figure.subplots(len(differences), 1, squeeze=False)[:, 0]

    for axis, (name, values) in zip(axes, differences.items(), strict=True):
        axis.plot(values.index, values, color=_BLUE, linewidth=1.3)
        axis.axhline(0, color="#64748b", linewidth=0.7)
        axis.set_title(f"{_LABELS[name]} · lag {lags[name]}", loc="left")
        if values.empty:
            axis.text(0.5, 0.5, "Not enough observations", transform=axis.transAxes)
        _style(axis, "Change in target")
        _time_axis(axis)

    return _save(figure, directory / "differences.png")


def plot_correlations(
    coefficients: pd.DataFrame,
    series: dict[str, pd.Series],
    notes: dict[str, str],
    kind: str,
    directory: Path,
) -> Path:
    """Plot saved coefficients with confidence bands centered on zero.

    The CSV stores confidence intervals around each estimate. For the familiar
    correlation plot, subtract that estimate before drawing the reference band.
    """
    figure = Figure(figsize=(12, 3 * len(series)), layout="constrained")
    axes = figure.subplots(len(series), 1, squeeze=False)[:, 0]

    for axis, name in zip(axes, series, strict=True):
        rows = coefficients.loc[
            (coefficients.series == name) & (coefficients.kind == kind)
        ]
        axis.set_title(f"{kind.upper()} · {_LABELS[name]}", loc="left")

        if rows.empty:
            axis.text(
                0.5,
                0.5,
                notes.get(name, "No coefficients available"),
                ha="center",
                transform=axis.transAxes,
            )
        else:
            # Lag zero always equals one; omitting it makes useful lags clearer.
            rows = rows.loc[rows.lag > 0]
            axis.vlines(rows.lag, 0, rows.coefficient, color=_BLUE, linewidth=1.2)
            axis.scatter(rows.lag, rows.coefficient, s=12, color=_BLUE)
            axis.fill_between(
                rows.lag,
                rows.lower_95 - rows.coefficient,
                rows.upper_95 - rows.coefficient,
                color=_BLUE,
                alpha=0.12,
                label="95% reference band",
            )
            axis.legend(frameon=False, loc="upper right")

        axis.axhline(0, color="#64748b", linewidth=0.7)
        axis.set_ylim(-1.05, 1.05)
        axis.set_xlabel("Lag (observations)")
        _style(axis, "Correlation")

    return _save(figure, directory / f"{kind}.png")


def plot_stl(
    components: pd.DataFrame | None,
    period: int | None,
    note: str | None,
    directory: Path,
) -> Path:
    """Keep a named STL plot even when decomposition is unavailable, with its reason."""
    if components is None:
        figure = Figure(figsize=(12, 3), layout="constrained")
        axis = figure.subplots()
        axis.set_title("STL decomposition", loc="left")
        axis.text(0.5, 0.5, note, ha="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        figure = Figure(figsize=(12, 10), layout="constrained")
        axes = figure.subplots(4, 1, sharex=True)
        figure.suptitle(f"Additive STL decomposition · period {period}")

        for axis, column in zip(axes, components, strict=True):
            axis.plot(components.index, components[column], color=_BLUE, linewidth=1.2)
            axis.set_title(column.capitalize(), loc="left", fontsize=11)
            _style(axis, "Target units")
            _time_axis(axis)

    return _save(figure, directory / "stl.png")


def plot_seasonality(
    target: pd.Series, cycles: pd.DataFrame | None, directory: Path
) -> Path:
    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(2, 1)
    months = sorted(set(target.index.month))
    axes[0].boxplot(
        [target.loc[target.index.month == month].to_numpy() for month in months],
        tick_labels=[pd.Timestamp(2000, month, 1).strftime("%b") for month in months],
    )
    axes[0].set_title("Calendar month distributions · observed target", loc="left")

    if cycles is None:
        axes[1].text(
            0.5,
            0.5,
            "Cycle overlay disabled: no seasonal period configured",
            ha="center",
            transform=axes[1].transAxes,
        )
    else:
        for cycle, rows in cycles.groupby("cycle"):
            axes[1].plot(
                rows.position, rows.observed, alpha=0.45, label=f"Cycle {cycle}"
            )
        means = cycles.groupby("position")["observed"].mean()
        axes[1].plot(means.index, means, color="#0f172a", linewidth=2.2, label="Mean")
        axes[1].legend(frameon=False, ncols=3, fontsize=9)

    axes[1].set_title("Configured cycles · aligned to first observation", loc="left")
    axes[1].set_xlabel("Position in cycle (observations)")

    for axis in axes:
        _style(axis, str(target.name))

    return _save(figure, directory / "seasonality.png")
