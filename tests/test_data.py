"""Data acquisition, loader selection, and task dispatch extension boundaries."""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from forgeml.data import sources
from forgeml.tasks import registry
from forgeml.tasks.forecasting.dataloaders import (
    CSVForecastDataLoader,
    create_data_loader,
)


def test_local_source_needs_no_download(tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("A local source must not contact Kaggle.")

    monkeypatch.setattr(sources.kagglehub, "dataset_download", unexpected_download)
    config = OmegaConf.create({"provider": "local", "directory": str(tmp_path)})

    assert sources.prepare_data(config) == tmp_path


def test_kaggle_source_is_configured_and_preserves_selected_files(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    (cache / "nested").mkdir(parents=True)
    (cache / "nested/sales.csv").write_text("date,value\n2020-01-01,1\n")
    (cache / "unused.csv").write_text("unused\n")
    handles = []

    def download(handle):
        handles.append(handle)
        return str(cache)

    monkeypatch.setattr(sources.kagglehub, "dataset_download", download)
    destination = tmp_path / "dataset"
    config = OmegaConf.create(
        {
            "provider": "kaggle",
            "handle": "owner/another-dataset/versions/3",
            "directory": str(destination),
            "files": ["nested/sales.csv"],
        }
    )
    assert sources.prepare_data(config) == destination
    assert handles == ["owner/another-dataset/versions/3"]
    assert (destination / "nested/sales.csv").read_bytes() == (
        cache / "nested/sales.csv"
    ).read_bytes()
    assert not (destination / "unused.csv").exists()


def test_csv_loader_uses_its_contract_without_hydra_in_implementation(tmp_path):
    path = tmp_path / "values.csv"
    pd.DataFrame(
        {"timestamp": ["2020-01-01", "2020-01-02"], "value": [3.0, 4.0]}
    ).to_csv(path, index=False)
    config = OmegaConf.create(
        {
            "loader": "csv",
            "path": str(path),
            "date_column": "timestamp",
            "target_column": "value",
        }
    )
    loader = create_data_loader(config)

    assert isinstance(loader, CSVForecastDataLoader)
    assert isinstance(loader.options, dict)
    assert loader.load().target.tolist() == [3.0, 4.0]

    config.loader = "missing"
    with pytest.raises(ValueError, match="loader"):
        create_data_loader(config)


def test_task_dispatch_does_not_impose_forecasting_data_on_another_task(monkeypatch):
    calls = []

    def another_task(config, launch_metadata=None):
        calls.append((config.data.image_directory, launch_metadata))
        return {"status": "completed"}

    monkeypatch.setitem(registry._TASKS, "example_task", another_task)
    config = OmegaConf.create(
        {"task": "example_task", "data": {"image_directory": "images"}}
    )

    assert registry.run_task(config, {"job_number": 1}) == {"status": "completed"}
    assert calls == [("images", {"job_number": 1})]


def test_report_dispatch_uses_the_selected_task(monkeypatch):
    calls = []

    def reporter(config, refresh_plots=False):
        calls.append((config.task, refresh_plots))

    monkeypatch.setitem(registry._REPORTERS, "example_task", reporter)
    config = OmegaConf.create({"task": "example_task"})
    registry.consolidate_task_results(config, refresh_plots=True)

    assert calls == [("example_task", True)]
