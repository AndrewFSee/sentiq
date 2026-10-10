"""Purged + embargoed time-series cross-validation, and the frozen holdout guard.

Samples carry a decision session ``s`` and a label-end session ``e`` (exit of the forward
return). Test folds are contiguous blocks of decision sessions [a, b].

* walk_forward (default): train = samples with e < a (label fully known before the test block
  starts -- this *is* the purge). Only past data trains each fold.
* blocked (Lopez de Prado): train = samples with e < a, or s > b + embargo. Training after the
  test block is allowed but labels overlapping the test block are purged, and an embargo gap
  after it limits leakage through serial correlation.

Holdout: every sample whose decision or label end falls on/after ``holdout_start`` is removed
before any split; the CLI only evaluates on it with an explicit ``--use-holdout`` flag (then
training uses only samples whose labels end before the holdout starts).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from newsfeed.config import CVConfig


@dataclass(frozen=True)
class Fold:
    train: np.ndarray  # row indices
    test: np.ndarray


class HoldoutError(RuntimeError):
    pass


def pre_holdout_mask(
    decision: np.ndarray, label_end: np.ndarray, holdout_start: date
) -> np.ndarray:
    """True for samples entirely before the holdout (decision and label end)."""
    hs = np.datetime64(holdout_start)
    return (decision < hs) & (label_end < hs)


def purged_splits(s: np.ndarray, e: np.ndarray, cfg: CVConfig) -> list[Fold]:
    """``s``/``e``: decision and label-end session indices per sample (ints)."""
    sessions = np.unique(s)
    if cfg.mode == "walk_forward":
        start = np.searchsorted(sessions, sessions[0] + cfg.min_train_sessions)
        eligible = sessions[start:]
    else:
        eligible = sessions
    if len(eligible) < cfg.n_folds:
        return []
    blocks = np.array_split(eligible, cfg.n_folds)
    folds = []
    for blk in blocks:
        a, b = int(blk[0]), int(blk[-1])
        test = np.flatnonzero((s >= a) & (s <= b))
        if cfg.mode == "walk_forward":
            train = np.flatnonzero(e < a)
        else:
            train = np.flatnonzero((e < a) | (s > b + cfg.embargo_sessions))
        if len(train) and len(test):
            folds.append(Fold(train, test))
    return folds
