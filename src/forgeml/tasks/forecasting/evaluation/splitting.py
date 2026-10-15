"""Chronological holdout and expanding-window cross-validation plans."""

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class EvaluationWindow:
    """Position boundaries: training is [0, train_end), testing is [start, end)."""

    evaluation: str
    fold: int
    train_end: int
    test_start: int
    test_end: int


def plan_evaluation(size: int, config: DictConfig) -> list[EvaluationWindow]:
    """Reserve the holdout first, then plan CV inside the remaining history."""
    windows = []
    development_end = size
    holdout = config.holdout

    if holdout.enabled:
        test_size = holdout.test_size
        count = int(size * test_size) if isinstance(test_size, float) else test_size

        if not 1 <= count <= size - 2:
            raise ValueError("Holdout must leave at least two training observations.")

        development_end = size - count

    cv = config.cv

    if cv.enabled:
        # Work backwards so the last validation window ends exactly at development_end.
        # Earlier folds are evaluated first, each with more historical training data.
        step = cv.get("step")

        if step is None:
            step = cv.horizon

        first_train_end = development_end - cv.horizon - (cv.n_splits - 1) * step

        if (
            cv.n_splits < 1
            or cv.horizon < 1
            or step < 1
            or first_train_end < max(2, cv.min_train_size)
        ):
            raise ValueError(
                "Not enough history for the requested CV windows/settings."
            )

        for fold in range(cv.n_splits):
            train_end = first_train_end + fold * step
            windows.append(
                EvaluationWindow(
                    "cv", fold + 1, train_end, train_end, train_end + cv.horizon
                )
            )

    if holdout.enabled:
        windows.append(
            EvaluationWindow("holdout", 1, development_end, development_end, size)
        )

    if not windows:
        raise ValueError("Enable holdout, CV, or both in evaluation config.")

    return windows
