from __future__ import annotations
import os
from typing import Dict, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from datasets.split_utils import build_episode_split_mask
from utils.normalizer import FieldNormalizer, MultiFieldNormalizer


class ForceEpisodeDataset(Dataset):
    """Episode-level force, position, and tactile trajectories for pretraining."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        force_key: str = "finger_force_30hz",
        state_key: str = "state",
        tactile_left_key: str = "gripper1_tactile_30hz",
        tactile_right_key: str = "gripper2_tactile_30hz",
        val_ratio: float | None = 0.1,
        split_seed: int | None = None,
        max_episode_steps: int | None = None,
        preload_to_ram: bool = False,
    ):
        self.root_dir = str(root_dir)
        self.split = str(split)
        self.force_key = str(force_key)
        self.state_key = str(state_key)
        self.tactile_left_key = str(tactile_left_key)
        self.tactile_right_key = str(tactile_right_key)
        self.val_ratio = val_ratio
        self.split_seed = 42 if split_seed is None else int(split_seed)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.preload_to_ram = bool(preload_to_ram)
        self.zarr_path, self.physical_split = self._resolve_zarr_path(
            self.root_dir, self.split
        )
        self.zarr_root = zarr.open_group(self.zarr_path, mode="r")
        self.data_group = self.zarr_root["data"]
        self.meta_group = self.zarr_root["meta"]
        self._validate_required_keys()
        self.episode_ends = np.asarray(
            self.meta_group["episode_ends"][:], dtype=np.int64
        )
        self.episode_starts = np.concatenate(
            [np.array([0], dtype=np.int64), self.episode_ends[:-1]]
        )
        self.behavior_ids = self._load_behavior_ids()
        self.episode_mask = self._build_episode_mask(len(self.episode_ends))
        self.episode_indices = np.flatnonzero(self.episode_mask).astype(np.int64)
        if len(self.episode_indices) == 0:
            raise ValueError(
                f"No episodes selected for split={self.split} in {self.root_dir}"
            )
        self.ram_data: Dict[str, np.ndarray] = {}
        if self.preload_to_ram:
            keys = [self.force_key, self.state_key]
            keys.extend([self.tactile_left_key, self.tactile_right_key])
            for key in keys:
                self.ram_data[key] = np.asarray(
                    self.data_group[key][:], dtype=np.float32
                )

    @staticmethod
    def _resolve_zarr_path(root_dir: str, split: str) -> tuple[str, bool]:
        if root_dir.endswith(".zarr") and os.path.isdir(root_dir):
            return (root_dir, True)
        split_path = os.path.join(root_dir, split, "replay_buffer.zarr")
        if os.path.isdir(split_path):
            return (split_path, True)
        fallback_candidates: list[str] = []
        if split == "val":
            fallback_candidates.append(
                os.path.join(root_dir, "test", "replay_buffer.zarr")
            )
        if split == "test":
            fallback_candidates.append(
                os.path.join(root_dir, "val", "replay_buffer.zarr")
            )
        for path in fallback_candidates:
            if os.path.isdir(path):
                return (path, True)
        shared_path = os.path.join(root_dir, "replay_buffer.zarr")
        if os.path.isdir(shared_path):
            return (shared_path, False)
        tried = [split_path, *fallback_candidates, shared_path]
        raise FileNotFoundError(
            f"Cannot find replay_buffer.zarr from root_dir={root_dir}, split={split}. Tried: {tried}"
        )

    def _validate_required_keys(self) -> None:
        for key in (self.force_key, self.state_key):
            if key not in self.data_group:
                raise KeyError(f"Missing key in zarr data group: {key}")
        for key in (self.tactile_left_key, self.tactile_right_key):
            if key not in self.data_group:
                raise KeyError(f"Missing tactile key in zarr data group: {key}")
        if "episode_ends" not in self.meta_group:
            raise KeyError("Missing key in zarr meta group: episode_ends")

    def _load_behavior_ids(self) -> np.ndarray | None:
        if "behavior_id" not in self.meta_group:
            return None
        behavior_ids = np.asarray(self.meta_group["behavior_id"][:], dtype=np.int64)
        if behavior_ids.shape != (len(self.episode_ends),):
            raise ValueError(
                "meta/behavior_id must have one entry per episode: "
                f"got {behavior_ids.shape}, episodes={len(self.episode_ends)}"
            )
        return behavior_ids

    def _build_episode_mask(self, num_episodes: int) -> np.ndarray:
        if self.physical_split or self.val_ratio is None or self.val_ratio <= 0.0:
            return np.ones(num_episodes, dtype=bool)
        return build_episode_split_mask(
            num_episodes,
            split=self.split,
            val_ratio=float(self.val_ratio),
            seed=self.split_seed,
            behavior_ids=self.behavior_ids,
        )

    def _read_array(self, key: str, slc) -> np.ndarray:
        if key in self.ram_data:
            return np.asarray(self.ram_data[key][slc], dtype=np.float32)
        return np.asarray(self.data_group[key][slc], dtype=np.float32)

    @staticmethod
    def _reshape_tactile(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[1] == 700:
            arr = arr.reshape(arr.shape[0], 35, 20, arr.shape[-1])
        if arr.ndim != 4:
            raise ValueError(f"Unsupported tactile shape: {arr.shape}")
        if arr.shape[1:3] != (35, 20):
            raise ValueError(
                f"Expected native tactile spatial shape 35x20, got {arr.shape}"
            )
        return np.asarray(arr, dtype=np.float32)

    def _read_tactile(self, slc) -> np.ndarray:
        left = self._reshape_tactile(self._read_array(self.tactile_left_key, slc))
        right = self._reshape_tactile(self._read_array(self.tactile_right_key, slc))
        if left.shape[:3] != right.shape[:3]:
            raise ValueError(
                f"Left/right tactile mismatch: {left.shape} vs {right.shape}"
            )
        return np.concatenate([left, right], axis=-1)

    def __len__(self) -> int:
        return int(len(self.episode_indices))

    @property
    def num_episodes_total(self) -> int:
        return int(len(self.episode_ends))

    def episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        ep_idx = int(ep_idx)
        return (int(self.episode_starts[ep_idx]), int(self.episode_ends[ep_idx]))

    def read_episode(self, ep_idx: int) -> Dict[str, torch.Tensor]:
        start, end = self.episode_bounds(ep_idx)
        if self.max_episode_steps is not None:
            end = min(end, start + self.max_episode_steps)
        state = self._read_array(self.state_key, slice(start, end))
        force = self._read_array(self.force_key, slice(start, end))
        if state.shape[0] != force.shape[0]:
            raise ValueError(
                f"state/force length mismatch: {state.shape} vs {force.shape}"
            )
        if state.shape[-1] < 3:
            raise ValueError(
                f"state must contain xyz in first 3 dims, got {state.shape[-1]}"
            )
        pos = state[:, :3] - state[:1, :3]
        out = {
            "pos": torch.from_numpy(pos.astype(np.float32, copy=False)),
            "force": torch.from_numpy(force.astype(np.float32, copy=False)),
            "episode_index": torch.tensor(ep_idx, dtype=torch.long),
            "length": torch.tensor(state.shape[0], dtype=torch.long),
        }
        tactile = self._read_tactile(slice(start, end))
        if tactile.shape[0] != state.shape[0]:
            raise ValueError(
                f"state/tactile length mismatch: {state.shape} vs {tactile.shape}"
            )
        out["tactile"] = torch.from_numpy(tactile)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.read_episode(int(self.episode_indices[int(idx)]))

    @staticmethod
    def _subsample_rows(arr: np.ndarray, max_rows: int | None) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
        if max_rows is None or max_rows <= 0 or x.shape[0] <= int(max_rows):
            return x
        idx = np.linspace(0, x.shape[0] - 1, num=int(max_rows), dtype=np.int64)
        return x[idx]

    def sample_normalizer_arrays(
        self, max_rows: int | None = None
    ) -> Dict[str, np.ndarray]:
        per_episode = None
        if max_rows is not None and max_rows > 0:
            per_episode = max(8, int(max_rows) // max(1, len(self.episode_indices)))
        buckets: Dict[str, list[np.ndarray]] = {"force": [], "mode_pos": []}
        buckets["tactile"] = []
        for ep_idx in self.episode_indices:
            start, end = self.episode_bounds(int(ep_idx))
            if self.max_episode_steps is not None:
                end = min(end, start + self.max_episode_steps)
            state = self._read_array(self.state_key, slice(start, end))
            force = self._read_array(self.force_key, slice(start, end))
            pos = state[:, :3] - state[:1, :3]
            buckets["force"].append(self._subsample_rows(force, per_episode))
            buckets["mode_pos"].append(self._subsample_rows(pos, per_episode))
            tactile = self._read_tactile(slice(start, end))
            buckets["tactile"].append(self._subsample_rows(tactile, per_episode))
        out = {}
        for key, parts in buckets.items():
            merged = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            out[key] = self._subsample_rows(merged, max_rows)
        return out

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        arrays = self.sample_normalizer_arrays(max_rows=max_rows)
        normalizer = MultiFieldNormalizer()
        for key, arr in arrays.items():
            normalizer[key] = FieldNormalizer.from_data_limits(arr)
        return normalizer


class ConcatForceEpisodeDataset(Dataset):

    def __init__(self, datasets: Sequence):
        if not datasets:
            raise ValueError("ConcatForceEpisodeDataset requires at least one dataset")
        self.datasets = list(datasets)
        lengths = [len(ds) for ds in self.datasets]
        self._cum = np.cumsum([0, *lengths]).astype(np.int64)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ds_idx = int(np.searchsorted(self._cum, idx, side="right") - 1)
        item = self.datasets[ds_idx][idx - int(self._cum[ds_idx])]
        return item

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        max_rows = int(max_rows) if max_rows is not None else 500000
        per_root = max(256, max_rows // max(1, len(self.datasets)))
        buckets: Dict[str, list[np.ndarray]] = {}
        for ds in self.datasets:
            arrays = ds.sample_normalizer_arrays(max_rows=per_root)
            for key, arr in arrays.items():
                buckets.setdefault(key, []).append(np.asarray(arr, dtype=np.float32))
        normalizer = MultiFieldNormalizer()
        for key, parts in buckets.items():
            merged = np.concatenate(parts, axis=0)
            if merged.shape[0] > max_rows:
                idx = np.linspace(0, merged.shape[0] - 1, num=max_rows, dtype=np.int64)
                merged = merged[idx]
            normalizer[key] = FieldNormalizer.from_data_limits(merged)
            print(f"  field={key} samples={merged.shape[0]} dim={merged.shape[-1]}")
        return normalizer


def build_force_episode_dataset(split: str = "train", **kwargs) -> Dataset:
    root_dir = kwargs.pop("root_dir")
    roots = [root_dir] if isinstance(root_dir, (str, os.PathLike)) else list(root_dir)
    roots = [str(r) for r in roots]
    if not roots:
        raise ValueError("root_dir is empty")
    datasets = [
        ForceEpisodeDataset(root_dir=root, split=split, **kwargs) for root in roots
    ]
    if len(datasets) == 1:
        return datasets[0]
    print(
        f"[build_force_episode_dataset] concat {len(datasets)} roots for split={split}:"
    )
    for ds in datasets:
        print(f"  - {ds.root_dir}: {len(ds)} episodes")
    return ConcatForceEpisodeDataset(datasets)


def collate_force_episodes(
    batch: list[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Pad variable-length FPT trajectories and mark valid source frames."""
    lengths = torch.stack([item["length"] for item in batch])
    max_len = int(lengths.max())
    result = {}
    for key in ("pos", "force", "tactile"):
        values = torch.zeros(
            len(batch), max_len, *batch[0][key].shape[1:], dtype=torch.float32
        )
        for i, item in enumerate(batch):
            values[i, : int(item["length"])] = item[key]
        result[key] = values
    result["length"] = lengths
    result["episode_index"] = torch.stack([item["episode_index"] for item in batch])
    result["mask"] = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return result
