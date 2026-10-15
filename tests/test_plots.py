"""Plot correctness: forecast alignment and separated metric comparison panels."""

import numpy as np
import pandas as pd
from matplotlib.image import imread

from forgeml.tasks.forecasting.evaluation import plots


def test_cv_plot_combines_folds_with_their_training_history(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-03", periods=9, freq="W-FRI")
    series = pd.DataFrame({"date": dates, "actual": np.arange(9, dtype=float)})
    series.to_csv(tmp_path / "series.csv", index=False)
    predictions = pd.DataFrame(
        {
            "evaluation": ["cv"] * 4 + ["holdout"] * 2,
            "fold": [1, 1, 2, 2, 1, 1],
            "date": [dates[4], dates[3], dates[5], dates[6], dates[7], dates[8]],
            "actual": [4, 3, 5, 6, 7, 8],
            "predicted": [2.0, 2.0, 4.0, 4.0, 6.0, 6.0],
        }
    )
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    metadata = {
        "model": "naive",
        "model_type": "naive",
        "use_exog": False,
        "parameters": {},
        "status": "completed",
        "data_config": {"target_column": "sales"},
    }
    windows = [
        {
            "evaluation": "cv",
            "fold": 1,
            "train_start": str(dates[0]),
            "train_end": str(dates[2]),
            "test_start": str(dates[3]),
            "test_end": str(dates[4]),
        },
        {
            "evaluation": "cv",
            "fold": 2,
            "train_start": str(dates[0]),
            "train_end": str(dates[4]),
            "test_start": str(dates[5]),
            "test_end": str(dates[6]),
        },
        {
            "evaluation": "holdout",
            "fold": 1,
            "train_start": str(dates[0]),
            "train_end": str(dates[6]),
            "test_start": str(dates[7]),
            "test_end": str(dates[8]),
        },
    ]
    for window in windows:
        window["model_fit"] = {}

    observed = {}
    original_save = plots._save_figure

    def capture(figure, path):
        observed[path.stem] = {
            line.get_label(): line.get_ydata().copy() for line in figure.axes[0].lines
        }
        return original_save(figure, path)

    monkeypatch.setattr(plots, "_save_figure", capture)
    paths = plots.plot_run_forecasts(
        {
            "metadata": metadata,
            "evaluation": {"windows": windows},
            "series": series,
            "predictions": predictions,
        },
        tmp_path,
        "cv",
    )

    assert {path.name for path in paths} == {"forecast.png", "cv_and_holdout.png"}
    # Initial training observations are visible, and the CV chart stops before holdout.
    np.testing.assert_array_equal(observed["forecast"]["Observed"], np.arange(7))
    np.testing.assert_array_equal(observed["forecast"]["CV fold 1"], [2.0, 2.0])
    np.testing.assert_array_equal(observed["forecast"]["CV fold 2"], [4.0, 4.0])
    np.testing.assert_array_equal(observed["cv_and_holdout"]["Observed"], np.arange(9))
    np.testing.assert_array_equal(observed["cv_and_holdout"]["Holdout"], [6.0, 6.0])

    for path in paths:
        assert imread(path).shape[:2] == (825, 1950)


def test_metric_plot_separates_groups_and_labels_undefined_scores(
    tmp_path, monkeypatch
):
    rows = pd.DataFrame(
        {
            "evaluation": ["holdout", "holdout", "holdout"],
            "comparison_id": ["first", "first", "other"],
            "model": ["naive", "arima_standard", "naive"],
            "run_id": ["run-a", "run-b", "run-c"],
            "run_name": ["baseline", "arima", "other_baseline"],
            "mape": [2.5, None, 5.0],
        }
    )
    observed = []
    original_save = plots._save_figure

    def capture(figure, path):
        for axis in figure.axes:
            observed.append(
                {
                    "title": axis.get_title(loc="left"),
                    "bars": len(axis.patches),
                    "labels": [text.get_text() for text in axis.texts],
                }
            )
        return original_save(figure, path)

    monkeypatch.setattr(plots, "_save_figure", capture)
    path = plots.plot_metric_comparison(rows, "mape", tmp_path)

    assert path == tmp_path / "plots/mape.png"
    assert imread(path).size > 0
    assert len(observed) == 2
    assert observed[0]["bars"] == 2
    assert observed[0]["labels"] == ["2.50", "Undefined"]
    assert "Matching target" in observed[0]["title"]
    assert observed[1]["bars"] == 1
    assert "Matching target" in observed[1]["title"]


def test_empty_metric_report_produces_a_clear_placeholder(tmp_path, monkeypatch):
    rows = pd.DataFrame(columns=["evaluation", "comparison_id", "rmse"])
    messages = []
    original_save = plots._save_figure

    def capture(figure, path):
        messages.extend(text.get_text() for text in figure.axes[0].texts)
        return original_save(figure, path)

    monkeypatch.setattr(plots, "_save_figure", capture)
    path = plots.plot_metric_comparison(rows, "rmse", tmp_path)

    assert messages == ["No completed results"]
    assert imread(path).shape[:2] == (450, 1950)
