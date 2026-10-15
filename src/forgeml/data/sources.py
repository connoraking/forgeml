"""Materialize raw datasets independently of ML task and model preparation."""

import json
import shutil
from pathlib import Path

import kagglehub
from omegaconf import DictConfig, OmegaConf


def _local(options: dict) -> Path:
    """A local source is already available; decoding belongs to the task's loader."""
    directory = Path(options["directory"]).expanduser().resolve()

    if not directory.is_dir():
        raise FileNotFoundError(f"Local data directory not found: {directory}")

    return directory


def _kaggle(options: dict) -> Path:
    """Download the configured Kaggle handle and preserve requested file paths."""
    cached = Path(kagglehub.dataset_download(options["handle"]))
    destination = Path(options["directory"]).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # A manifest may select only the files it needs. Otherwise copy the whole source.
    files = options.get("files")
    sources = [cached / name for name in files] if files else cached.rglob("*")

    for source in sources:
        if source.is_file():
            target = destination / source.relative_to(cached)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif files:
            raise FileNotFoundError(f"Configured source file not found: {source}")

    provenance = {"provider": "kaggle", "handle": options["handle"]}
    (destination / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")

    return destination


_SOURCES = {"local": _local, "kaggle": _kaggle}


def prepare_data(config: DictConfig) -> Path:
    """Materialize a source declared in YAML; training never downloads implicitly."""
    options = OmegaConf.to_container(config, resolve=True)
    provider = options["provider"]

    if provider not in _SOURCES:
        raise ValueError(f"Unknown data source provider: {provider!r}")

    return _SOURCES[provider](options)
