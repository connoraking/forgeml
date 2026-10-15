"""Forecast timelines and model comparisons, with observed training history."""

from collections import defaultdict
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
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=150, facecolor="white")
    figure.clear()
    return path


def model_label(record: dict) -> str:
    """Short chart labels; full fitted orders also live in result paths and CSVs."""
    metadata = record["metadata"]
    name = metadata["model_type"]
    if name == "arima" and metadata.get("selection") == "auto":
        label = "Auto-ARIMA"
    elif name == "arima":
        fitted = record["evaluation"]["windows"][0]["model_fit"]
        label = f"ARIMA {tuple(fitted['order'])} × {tuple(fitted['seasonal_order'])}"
    else:
        label = name.replace("_", " ").title()
    if metadata["use_exog"]:
        label += " + exogenous"
    if metadata["status"] != "completed":
        label += " (partial)"
    return label


def _windows(record: dict, scope: str | None) -> list[dict]:
    return [
        window
        for window in record["evaluation"]["windows"]
        if scope is None or window["evaluation"] == scope
    ]


def _style(axis, target: str) -> None:
    axis.set_ylabel(target)
    axis.set_xlabel("Date")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    locator = AutoDateLocator(minticks=4, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _history(axis, record: dict, windows: list[dict]) -> None:
    """Clip split plots at their last evaluation date; combined views show both."""
    start = min(pd.Timestamp(window["train_start"]) for window in windows)
    end = max(pd.Timestamp(window["test_end"]) for window in windows)
    observed = record["series"]
    observed = observed.loc[observed.date.between(start, end)]
    axis.plot(observed.date, observed.actual, color="#64748b", lw=1.2, label="Observed")

    first_cutoff = min(pd.Timestamp(window["train_end"]) for window in windows)
    axis.axvspan(start, first_cutoff, color="#94a3b8", alpha=0.08)
    for window in windows:
        cutoff = pd.Timestamp(window["train_end"])
        holdout = window["evaluation"] == "holdout"
        color = "#f59e0b" if holdout else "#3b82f6"
        axis.axvspan(
            pd.Timestamp(window["test_start"]),
            pd.Timestamp(window["test_end"]),
            color=color,
            alpha=0.08,
        )
        axis.axvline(cutoff, color=color, linestyle=":", lw=0.9)
        text = "Holdout" if holdout else f"CV {window['fold']}"
        axis.text(
            cutoff,
            0.98 - 0.065 * (window["fold"] % 3),
            text,
            transform=axis.get_xaxis_transform(),
            color=color,
            fontsize=8,
            va="top",
        )
    _style(axis, record["metadata"]["data_config"]["target_column"])


def _forecast_lines(axis, record: dict, windows: list[dict], color=None) -> None:
    """Draw each forecast separately, never connecting across fold boundaries."""
    predictions = record["predictions"]
    for index, window in enumerate(windows):
        rows = predictions.loc[
            (predictions.evaluation == window["evaluation"])
            & (predictions.fold == window["fold"])
        ].sort_values("date")
        holdout = window["evaluation"] == "holdout"
        if color is None:
            line_color = colormaps["tab10"](index % 10)
            label = "Holdout" if holdout else f"CV fold {window['fold']}"
            fitted = window["model_fit"]
            if "order" in fitted:
                label += (
                    f" · {tuple(fitted['order'])} × {tuple(fitted['seasonal_order'])}"
                )
        else:
            line_color = color
            label = model_label(record) if index == 0 else None
        axis.plot(
            rows.date,
            rows.predicted,
            color=line_color,
            linestyle="-" if holdout else "--",
            marker=".",
            markersize=3,
            linewidth=1.6,
            label=label,
        )


def plot_run_forecasts(record: dict, directory: Path, scope: str) -> list[Path]:
    """Each model gets its split view and its CV-plus-holdout timeline."""
    paths = []
    for selected_scope, filename in (
        (scope, "forecast.png"),
        (None, "cv_and_holdout.png"),
    ):
        windows = _windows(record, selected_scope)
        if not windows:
            continue
        figure = Figure(figsize=(13, 5.5), layout="constrained")
        axis = figure.subplots()
        _history(axis, record, windows)
        _forecast_lines(axis, record, windows)
        view = scope.upper() if selected_scope else "CV and holdout"
        axis.set_title(f"{model_label(record)} · {view}", loc="left")
        axis.legend(frameon=False, fontsize=8, loc="upper left")
        figure.supxlabel(
            "Dashed: CV. Solid: holdout. Dotted lines: training cutoffs.",
            fontsize=9,
        )
        paths.append(_save_figure(figure, directory / "plots" / filename))
    return paths


def plot_forecast_comparison(
    records: list[dict],
    directory: Path,
    scope: str | None = None,
    stacked: bool = False,
) -> Path:
    """Compare every model overlaid or stacked, keeping incompatible data apart."""
    groups = defaultdict(list)
    for record in records:
        windows = _windows(record, scope)
        if not windows:
            continue
        boundaries = tuple(
            (
                w["evaluation"],
                w["fold"],
                w["train_start"],
                w["train_end"],
                w["test_start"],
                w["test_end"],
            )
            for w in windows
        )
        groups[(record["metadata"]["target_fingerprint"], boundaries)].append(record)

    panels = []
    for members in groups.values():
        members.sort(key=model_label)
        panels.extend([[record] for record in members] if stacked else [members])

    filename = "forecast_stacked.png" if stacked else "forecast_overlay.png"
    if not panels:
        figure = Figure(figsize=(11, 3), layout="constrained")
        axis = figure.subplots()
        axis.text(
            0.5, 0.5, "No forecasts available", ha="center", transform=axis.transAxes
        )
        axis.set_axis_off()
        return _save_figure(figure, directory / filename)

    height = 4.1 if stacked else 5.8
    figure = Figure(figsize=(14, height * len(panels)), layout="constrained")
    axes = figure.subplots(len(panels), 1, squeeze=False)[:, 0]
    view = scope.upper() if scope else "CV and holdout"
    figure.suptitle(f"All models · {view} · {'stacked' if stacked else 'overlaid'}")
    palette = colormaps["tab10"]
    for axis, members in zip(axes, panels, strict=True):
        reference = members[0]
        _history(axis, reference, _windows(reference, scope))
        for index, record in enumerate(members):
            _forecast_lines(
                axis,
                record,
                _windows(record, scope),
                color=None if stacked else palette(index % 10),
            )
        title = (
            model_label(reference)
            if stacked
            else "Matching target and evaluation windows"
        )
        axis.set_title(title, loc="left", fontsize=11)
        axis.legend(
            frameon=False, fontsize=8, loc="upper left", ncols=1 if stacked else 2
        )
    figure.supxlabel(
        "Observed history in gray. Dashed forecasts: CV. Solid forecasts: holdout.",
        fontsize=9,
    )
    return _save_figure(figure, directory / filename)


def plot_metric_comparison(rows: pd.DataFrame, metric: str, directory: Path) -> Path:
    """One evaluation per file, with panels for differing targets or split plans."""
    panels = list(rows.groupby("comparison_id", sort=True))
    figure = Figure(
        figsize=(13, sum(max(2.8, 1.5 + 0.5 * len(frame)) for _, frame in panels) or 3),
        layout="constrained",
    )
    if not panels:
        axis = figure.subplots()
        axis.text(
            0.5, 0.5, "No completed results", ha="center", transform=axis.transAxes
        )
        axis.set_axis_off()
    else:
        axes = figure.subplots(len(panels), 1, squeeze=False)[:, 0]
        for axis, (_, frame) in zip(axes, panels, strict=True):
            frame = frame.sort_values(metric, na_position="last")
            values = frame[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            widths = np.where(finite, values, 0.0)
            bars = axis.barh(frame.model, widths, color="#2563eb")
            value_format = ".4f" if metric == "rmsle" else ",.2f"
            labels = [
                f"{value:{value_format}}" if valid else "Undefined"
                for value, valid in zip(values, finite, strict=True)
            ]
            axis.bar_label(bars, labels=labels, padding=5, fontsize=9)
            axis.invert_yaxis()
            maximum = float(widths.max()) if len(widths) else 0.0
            minimum_range = 1e-6 if metric == "rmsle" else 1
            axis.set_xlim(0, max(minimum_range, maximum * 1.3))
            unit = "%" if metric in _PERCENT_METRICS else "target units"
            if metric == "mse":
                unit = "target units²"
            elif metric == "rmsle":
                unit = "unitless"
            axis.set_xlabel(f"{metric.upper()} ({unit})")
            tick_format = "{x:.3f}" if metric == "rmsle" else "{x:,.0f}"
            axis.xaxis.set_major_formatter(StrMethodFormatter(tick_format))
            axis.grid(axis="x", alpha=0.15)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(
                "Matching target and evaluation windows", loc="left", fontsize=10
            )
    figure.suptitle(f"{directory.name.upper()} · {metric.upper()} · lower is better")
    return _save_figure(figure, directory / "plots" / f"{metric}.png")
