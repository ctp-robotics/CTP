from __future__ import annotations

import numpy as np


def build_episode_split_mask(
    num_episodes: int,
    *,
    split: str,
    val_ratio: float,
    seed: int,
    behavior_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Split episodes reproducibly, stratifying by behavior when labels exist."""
    if num_episodes <= 1:
        return np.full(num_episodes, split == "train", dtype=bool)

    rng = np.random.default_rng(seed)
    if behavior_ids is None:
        groups = [np.arange(num_episodes, dtype=np.int64)]
    else:
        groups = [np.flatnonzero(behavior_ids == value) for value in np.unique(behavior_ids)]

    val_mask = np.zeros(num_episodes, dtype=bool)
    for indices in groups:
        # A singleton must remain in train; it cannot be represented in both splits.
        if len(indices) < 2:
            continue
        n_val = max(1, int(round(len(indices) * float(val_ratio))))
        n_val = min(n_val, len(indices) - 1)
        val_mask[rng.choice(indices, size=n_val, replace=False)] = True

    if not val_mask.any():
        raise ValueError(
            "Cannot create a validation split: every behavior has fewer than two episodes."
        )
    return ~val_mask if split == "train" else val_mask
