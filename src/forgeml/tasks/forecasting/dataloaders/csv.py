"""CSV preparation for a forecasting dataset, independent of its download source."""

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseForecastDataLoader, ForecastData


class CSVForecastDataLoader(BaseForecastDataLoader):
    """Read a CSV, filter rows, aggregate targets, and establish a regular time grid."""

    def __init__(self, options: dict) -> None:
        self.options = options

    def load(self) -> ForecastData:
        options = self.options
        path = Path(options["path"]).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Set data.path to your CSV."
            )

        frame = pd.read_csv(path)

        # Filters select the series for this experiment from a larger dataset.
        for column, value in options.get("filters", {}).items():
            frame = frame.loc[frame[column] == value]

        date_column = options["date_column"]
        target_column = options["target_column"]
        feature_columns = list(options.get("known_future_columns", []))

        if target_column in feature_columns:
            raise ValueError("The target cannot also be a future feature.")

        columns = [target_column, *feature_columns]
        frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
        frame = frame.set_index(date_column)[columns].apply(pd.to_numeric)

        if frame.index.hasnans or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                "The CSV contains missing dates or nonfinite numeric values."
            )

        # Aggregation rules belong to the experiment config, such as summing sales
        # across departments. The loader contains no dataset-specific column names.
        aggregation = options.get("aggregate")

        if aggregation:
            frame = frame.groupby(level=0).agg(aggregation)[columns]

        frame = frame.sort_index()

        if frame.empty or frame.index.has_duplicates or frame.index.hasnans:
            raise ValueError(
                "Data must have valid, unique dates after filtering/aggregation."
            )

        # A missing week must not silently become 'one step' next to the previous week.
        frequency = options.get("frequency")

        if frequency:
            regular = frame.asfreq(frequency)

            if not frame.index.isin(regular.index).all():
                raise ValueError("Some dates do not match data.frequency.")

            frame = regular

        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                "Data has missing periods or nonfinite values; clean the CSV first."
            )

        target = frame[target_column].astype(float)
        exog = frame[feature_columns].astype(float) if feature_columns else None

        return ForecastData(target=target, exog=exog)
