"""Saved forecast curves and metric comparisons, rendered without a GUI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

_PERCENT_METRICS = {"mape", "smape", "wape"}


def _save_figure(figure: Figure, path: Path) -> Path:
    """Write a standalone PNG while keeping Matplotlib's GUI state out of the app."""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=150, facecolor="white")
    figure.clear()

    return path


def plot_run_forecasts(
    directory: Path, metadata: dict, refresh: bool = False
) -> list[Path]:
    """Show each evaluation's history and forecasts on one shared timeline.

    CV folds remain separate forecast lines, including overlapping horizons.
    The observed series is clipped at this evaluation's last test date, so the
    CV chart excludes a later final holdout. Colored cutoffs show how each
    expanding training window grows from the beginning of the visible history.
    """
    source = directory / "predictions.csv"

    if not source.exists():
        return []

    predictions = pd.read_csv(source, parse_dates=["date"])
    series = pd.read_csv(directory / "series.csv", parse_dates=["date"])
    evaluation = json.loads((directory / "metrics.json").read_text())
    inputs = [
        source,
        directory / "series.csv",
        directory / "metrics.json",
        directory / "run.json",
    ]
    updated_at = max(path.stat().st_mtime for path in inputs)
    paths = []

    for name, rows in predictions.groupby("evaluation"):
        path = directory / "plots" / f"{name}.png"
        paths.append(path)

        if not refresh and path.exists() and path.stat().st_mtime >= updated_at:
            continue

        windows = [
            window for window in evaluation["windows"] if window["evaluation"] == name
        ]
        start = min(pd.Timestamp(window["train_start"]) for window in windows)
        end = max(pd.Timestamp(window["test_end"]) for window in windows)
        observed = series.loc[series.date.between(start, end)].sort_values("date")

        figure = Figure(figsize=(12, 5), layout="constrained")
        axis = figure.subplots()
        axis.plot(
            observed.date,
            observed.actual,
            color="#475569",
            linewidth=1.6,
            label="Observed target",
        )

        first_cutoff = min(pd.Timestamp(window["train_end"]) for window in windows)
        training_label = (
            "Initial training history" if name == "cv" else "Training history"
        )
        axis.axvspan(
            start, first_cutoff, color="#94a3b8", alpha=0.10, label=training_label
        )

        for index, window in enumerate(sorted(windows, key=lambda item: item["fold"])):
            forecast = rows.loc[rows.fold == window["fold"]].sort_values("date")
            color = colormaps["tab10"](index % 10)
            cutoff = pd.Timestamp(window["train_end"])
            test_start = pd.Timestamp(window["test_start"])
            test_end = pd.Timestamp(window["test_end"])
            label = (
                f"Fold {window['fold']} forecast"
                if name == "cv"
                else "Holdout forecast"
            )

            # Training ends at the colored dotted cutoff. Forecasts start at the
            # following test observation; targets after the cutoff never fit this fold.
            axis.axvline(cutoff, color=color, linestyle=":", linewidth=1.3)
            axis.axvspan(test_start, test_end, color=color, alpha=0.07)
            axis.plot(
                forecast.date,
                forecast.predicted,
                color=color,
                linestyle="--",
                marker=".",
                label=label,
            )
            axis.text(
                cutoff,
                0.98 - 0.07 * (index % 3),
                f"Fold {window['fold']}" if name == "cv" else "Training cutoff",
                transform=axis.get_xaxis_transform(),
                color=color,
                fontsize=9,
                ha="left",
                va="top",
            )

        title = "Cross-validation: all folds" if name == "cv" else "Holdout forecast"
        state = " · partial run" if metadata["status"] != "completed" else ""
        axis.set_title(f"{metadata['model']} · {title}{state}", loc="left")
        axis.set_xlabel("Date")
        axis.set_ylabel(metadata["data_config"]["target_column"])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncols=2, fontsize=9, loc="upper left")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        if observed.actual.abs().max() >= 100:
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        dates = AutoDateLocator(minticks=5, maxticks=9)
        axis.xaxis.set_major_locator(dates)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(dates))
        figure.supxlabel(
            "Each forecast trains on the history up to its matching colored cutoff.",
            fontsize=9,
        )

        _save_figure(figure, path)

    return paths


def plot_metric_comparison(rows: pd.DataFrame, metric: str, directory: Path) -> Path:
    """Plot each run's score, separating incompatible comparison groups.

    Each panel contains matching target data and evaluation windows. Each configured
    variant has one bar; undefined percentage scores are labeled explicitly.
    """
    path = directory / "plots" / f"{metric}.png"
    panels = list(rows.groupby(["evaluation", "comparison_id"], sort=True))

    if not panels:
        figure = Figure(figsize=(10, 3), layout="constrained")
        axis = figure.subplots()
        axis.set_title(f"{metric.upper()} by model", loc="left")
        axis.text(
            0.5,
            0.5,
            "No completed results",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()

        return _save_figure(figure, path)

    # Allow enough vertical space for every run, rather than squeezing long labels.
    heights = [max(2.4, 1.6 + 0.38 * len(panel)) for _, panel in panels]
    label_length = rows.run_name.astype(str).str.len().max()
    width = max(11, min(20, 6 + label_length * 0.08))
    figure = Figure(figsize=(width, sum(heights) + 0.6), layout="constrained")
    axes = figure.subplots(len(panels), 1, squeeze=False, height_ratios=heights)[:, 0]
    figure.suptitle(f"{metric.upper()} by model · lower is better", fontsize=14)

    model_names = sorted(rows.model.unique())
    palette = colormaps["tab10"]
    colors = {name: palette(index % 10) for index, name in enumerate(model_names)}

    for axis, ((evaluation, comparison_id), panel) in zip(axes, panels, strict=True):
        panel = panel.sort_values([metric, "model", "run_id"], na_position="last")
        values = panel[metric].to_numpy(dtype=float)
        labels = panel.run_name.tolist()
        finite = np.isfinite(values)
        widths = np.where(finite, values, 0.0)

        bars = axis.barh(labels, widths, color=[colors[name] for name in panel.model])
        score_labels = [
            f"{value:,.2f}" if valid else "Undefined"
            for value, valid in zip(values, finite, strict=True)
        ]
        axis.bar_label(bars, labels=score_labels, padding=5, fontsize=9)
        axis.invert_yaxis()

        maximum = float(widths.max()) if widths.size else 0.0
        axis.set_xlim(0, max(maximum * 1.35, 1.0))
        axis.set_title(
            f"{evaluation.upper()} · matching target/windows · {comparison_id}",
            loc="left",
            fontsize=10,
        )
        unit = "%" if metric in _PERCENT_METRICS else "target units"
        if metric == "mse":
            unit = "target units²"

        axis.set_xlabel(f"{metric.upper()} ({unit})")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        if maximum >= 100:
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="y", labelsize=9)

    return _save_figure(figure, path)
