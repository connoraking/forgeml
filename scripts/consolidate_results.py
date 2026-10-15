"""Rebuild metric reports and plots through the configured task's reporter."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from forgeml.tasks.registry import consolidate_task_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment's main YAML config.",
    )
    parser.add_argument("--refresh-plots", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    consolidate_task_results(config, refresh_plots=args.refresh_plots)
    print(f"Metric reports and plots saved to {config.results.directory}")


if __name__ == "__main__":
    main()
