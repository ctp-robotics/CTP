from __future__ import annotations
import os
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr
from utils.normalizer import (
    FieldNormalizer,
    MultiFieldNormalizer,
    fit_limited_normalizer,
)
from utils.action import absolute_actions_to_relative_actions
from datasets.split_utils import build_episode_split_mask

FORCE_NORM_MIN_RANGE = 0.001
FORCE_NORM_MAX_ABS_SCALE = 100.0


class ZarrDataset(Dataset):
    """Policy dataset: image / state history + action chunk + future force labels.

    Each sample:
      obs.image        [T_img, V, 3, H, W]  (or obs.image_backbone_feat from cache)
      obs.state        [T, Ds]
      obs.force        [T_force, Df]  (recent online force history, optional)
      obs.tactile      [T_tac, 35, 20, 6]   (left ch 0:3 / right ch 3:6, optional)
      action           [Ta, Da]
      future_force     [Tf, Df]  (at configurable offset from action chunk start, optional)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        window_size: int = 32,
        stride: int = 1,
        action_dim: int = 10,
        image_size: int = 224,
        n_image_steps: int | None = None,
        action_window_size: int | None = None,
        image_as_uint8: bool = True,
        preload_to_ram: bool = False,
        latent_cache_root_dir: str | None = None,
        force_vq_cache_root_dir: str | None = None,
        force_vq_prompt_tokens: int = 16,
        image_keys: Sequence[str] = ("global_rgb_cam", "left_cam1"),
        force_key: str = "finger_force_30hz",
        state_key: str = "state",
        action_key: str = "action",
        tactile_left_key: str = "gripper1_tactile_30hz",
        tactile_right_key: str = "gripper2_tactile_30hz",
        action_representation: str = "absolute",
        future_force_steps: int | None = None,
        future_force_offset: int = 0,
        force_steps: int | None = None,
        tactile_steps: int | None = None,
        val_ratio: float | None = 0.1,
        split_seed: int | None = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.val_ratio = val_ratio
        self.split_seed = 42 if split_seed is None else int(split_seed)
        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.action_dim = int(action_dim)
        self.image_size = int(image_size)
        self.n_image_steps = (
            self.window_size if n_image_steps is None else max(0, int(n_image_steps))
        )
        self.action_window_size = (
            self.window_size
            if action_window_size is None
            else max(1, int(action_window_size))
        )
        self.future_force_steps = int(future_force_steps) if future_force_steps else 0
        self.future_force_offset = int(future_force_offset)
        if self.future_force_offset < 0:
            raise ValueError(
                f"future_force_offset must be non-negative, got {future_force_offset}"
            )
        if self.future_force_steps > self.action_window_size:
            raise ValueError(
                f"future_force_steps must be <= action_window_size so the future force window stays inside the episode, got future_force_steps={self.future_force_steps}, action_window_size={self.action_window_size}"
            )
        self.force_steps = int(force_steps) if force_steps else 0
        self.tactile_steps = int(tactile_steps) if tactile_steps else 0
        self.image_as_uint8 = bool(image_as_uint8)
        self.preload_to_ram = bool(preload_to_ram)
        self.latent_cache_root_dir = latent_cache_root_dir
        self.force_vq_cache_root_dir = force_vq_cache_root_dir
        self.force_vq_prompt_tokens = int(force_vq_prompt_tokens)
        self.image_keys = list(image_keys)
        self.force_key = force_key
        self.state_key = state_key
        self.action_key = action_key
        self.tactile_left_key = tactile_left_key
        self.tactile_right_key = tactile_right_key
        self.action_representation = self._normalize_action_representation(
            action_representation
        )
        self.zarr_path, self.physical_split = self._resolve_zarr_path(root_dir, split)
        self.zarr_root = zarr.open_group(self.zarr_path, mode="r")
        self.data_group = self.zarr_root["data"]
        self.meta_group = self.zarr_root["meta"]
        self.latent_cache_zarr = None
        self.latent_cache_group = None
        self.cached_image_backbone_feat = None
        self.force_vq_cache_zarr = None
        self.force_vq_padding_mask = None
        self.force_vq_sample_indices = None
        self.force_vq_valid_episode_mask = None
        self.force_vq_ref_episode_indices: np.ndarray | None = None
        self.force_vq_ref_episode_indices_by_behavior: dict[int, np.ndarray] = {}
        self.force_vq_reference_pos: np.ndarray | None = None
        self.force_vq_reference_force: np.ndarray | None = None
        self.force_vq_reference_tactile: np.ndarray | None = None
        self.force_vq_reference_phase: np.ndarray | None = None
        self.ram_data: Dict[str, np.ndarray] = {}
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
        self.windows = self._build_windows()
        if len(self.windows) == 0:
            raise ValueError(
                f"No valid strict anchor windows. window_size={self.window_size}, n_image_steps={self.n_image_steps}, action_window_size={self.action_window_size}, stride={self.stride}"
            )
        self._maybe_open_latent_cache()
        self._maybe_open_force_vq_cache()
        self._maybe_build_trainable_force_vq_references()
        self._maybe_preload_arrays()

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

    def _build_episode_mask(self, num_episodes: int) -> np.ndarray:
        if self.physical_split or self.val_ratio is None or self.val_ratio <= 0.0:
            return np.ones(num_episodes, dtype=bool)
        mask = build_episode_split_mask(
            num_episodes,
            split=self.split,
            val_ratio=float(self.val_ratio),
            seed=self.split_seed,
            behavior_ids=self.behavior_ids,
        )
        n_selected = int(mask.sum())
        print(
            f"[ZarrDataset] episode-level split={self.split}: {n_selected}/{num_episodes} episodes (val_ratio={self.val_ratio}, seed={self.split_seed}, zarr={self.zarr_path})"
        )
        if self.behavior_ids is not None:
            counts = {
                int(value): int((mask & (self.behavior_ids == value)).sum())
                for value in np.unique(self.behavior_ids)
            }
            print(f"[ZarrDataset] split behavior counts: {counts}")
        if n_selected == 0:
            raise ValueError(
                f"No episodes selected for split={self.split}. num_episodes={num_episodes}, val_ratio={self.val_ratio}"
            )
        return mask

    @staticmethod
    def _normalize_action_representation(value: str) -> str:
        rep = str(value).strip().lower()
        if rep in {"relative", "relative_action"}:
            rep = "chunk_relative"
        if rep not in {"absolute", "chunk_relative"}:
            raise ValueError(
                f"Unsupported action_representation={value}. Choose from ['absolute', 'chunk_relative']."
            )
        return rep

    def _validate_required_keys(self) -> None:
        required = [self.state_key, self.action_key]
        if self.n_image_steps > 0:
            required.extend(self.image_keys)
        if self.future_force_steps > 0 or self.force_steps > 0:
            required.append(self.force_key)
        if self.tactile_steps > 0:
            required.extend([self.tactile_left_key, self.tactile_right_key])
        for key in set(required):
            if key not in self.data_group:
                raise KeyError(f"Missing key in zarr data group: {key}")
        if "episode_ends" not in self.meta_group:
            raise KeyError("Missing key in zarr meta group: episode_ends")

    def _load_behavior_ids(self) -> np.ndarray | None:
        if "behavior_id" not in self.meta_group:
            return None
        behavior_ids = np.asarray(self.meta_group["behavior_id"][:], dtype=np.int64)
        if behavior_ids.shape != (len(self.episode_ends),):
            raise ValueError(
                f"meta/behavior_id must have one entry per episode: got {behavior_ids.shape}, episodes={len(self.episode_ends)}"
            )
        return behavior_ids

    def _resolve_latent_cache_path(self) -> str | None:
        if self.latent_cache_root_dir:
            path = os.path.join(
                self.latent_cache_root_dir, self.split, "policy_latent_cache.zarr"
            )
            if os.path.isdir(path):
                return path
        return None

    def _resolve_force_vq_cache_path(self) -> str | None:
        if not self.force_vq_cache_root_dir:
            return None
        root = str(self.force_vq_cache_root_dir)
        if root.endswith(".zarr"):
            if os.path.isdir(root):
                return root
            raise FileNotFoundError(f"contact encoder prompt cache not found: {root}")
        path = os.path.join(root, self.split, "force_vq_prompt_cache.zarr")
        if os.path.isdir(path):
            return path
        raise FileNotFoundError(f"contact encoder prompt cache not found: {path}")

    def _maybe_open_latent_cache(self) -> None:
        if self.n_image_steps <= 0:
            return
        cache_path = self._resolve_latent_cache_path()
        if cache_path is None:
            return
        try:
            self.latent_cache_zarr = zarr.open_group(cache_path, mode="r")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open latent cache at {cache_path}. Rebuild cache or set data.latent_cache_root_dir: null."
            ) from e
        if "data" not in self.latent_cache_zarr or "meta" not in self.latent_cache_zarr:
            raise KeyError(f"Invalid latent cache zarr structure: {cache_path}")
        self.latent_cache_group = self.latent_cache_zarr["data"]
        cache_meta = self.latent_cache_zarr["meta"]
        if "image_backbone_feat" not in self.latent_cache_group:
            raise KeyError(
                "Missing key in latent cache data group: image_backbone_feat"
            )
        for key in (
            "window_anchor_times",
            "window_episode_ends",
            "window_episode_indices",
        ):
            if key not in cache_meta:
                raise KeyError(
                    f"Missing key in latent cache meta group: {key}. Please rebuild cache using the strict anchor precompute script."
                )
        cache_anchor = np.asarray(cache_meta["window_anchor_times"][:], dtype=np.int64)
        cache_ep_end = np.asarray(cache_meta["window_episode_ends"][:], dtype=np.int64)
        cache_ep_idx = np.asarray(
            cache_meta["window_episode_indices"][:], dtype=np.int64
        )
        if not len(cache_anchor) == len(cache_ep_end) == len(cache_ep_idx):
            raise ValueError(
                f"Latent cache meta length mismatch: anchor={len(cache_anchor)}, ep_end={len(cache_ep_end)}, ep_idx={len(cache_ep_idx)}"
            )
        cache_attrs = getattr(self.latent_cache_zarr, "attrs", {})
        expected_attrs = {
            "window_size": self.window_size,
            "action_window_size": self.action_window_size,
            "stride": self.stride,
        }
        for key, expected in expected_attrs.items():
            actual = cache_attrs.get(key)
            if actual is not None and int(actual) != int(expected):
                raise ValueError(
                    f"Latent cache attr mismatch for {key}: cache={actual}, dataset={expected}"
                )
        window_arr = np.asarray(self.windows, dtype=np.int64)
        if len(cache_anchor) != len(window_arr):
            raise ValueError(
                f"Latent cache window count mismatch: cache={len(cache_anchor)} dataset={len(window_arr)}"
            )
        if not (
            np.array_equal(cache_anchor, window_arr[:, 0])
            and np.array_equal(cache_ep_end, window_arr[:, 1])
            and np.array_equal(cache_ep_idx, window_arr[:, 2])
        ):
            raise ValueError(
                "Latent cache windows do not match dataset strict anchor windows. Please rebuild cache for the current data config."
            )
        self.cached_image_backbone_feat = self.latent_cache_group["image_backbone_feat"]
        if self.cached_image_backbone_feat.shape[0] != len(self.windows):
            raise ValueError(
                f"Latent cache image feature row count mismatch: cache_rows={self.cached_image_backbone_feat.shape[0]}, windows={len(self.windows)}"
            )

    def _maybe_open_force_vq_cache(self) -> None:
        cache_path = self._resolve_force_vq_cache_path()
        if cache_path is None:
            return
        try:
            self.force_vq_cache_zarr = zarr.open_group(cache_path, mode="r")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open contact encoder prompt cache at {cache_path}. Rebuild cache or set data.force_vq_cache_root_dir: null."
            ) from e
        if (
            "data" not in self.force_vq_cache_zarr
            or "meta" not in self.force_vq_cache_zarr
        ):
            raise KeyError(
                f"Invalid contact encoder prompt cache structure: {cache_path}"
            )
        data_group = self.force_vq_cache_zarr["data"]
        meta_group = self.force_vq_cache_zarr["meta"]
        sample_indices = np.asarray(data_group["sample_indices"][:], dtype=np.int64)
        valid = np.asarray(meta_group["valid_episode_mask"][:], dtype=bool)
        if (
            sample_indices.ndim != 2
            or sample_indices.shape[0] != self.num_episodes
            or valid.shape != (self.num_episodes,)
        ):
            raise ValueError(
                "Reference cache episode count does not match the dataset."
            )
        missing_selected = np.flatnonzero(self.episode_mask & ~valid)
        if len(missing_selected) > 0:
            raise ValueError(
                f"contact encoder cache is missing {len(missing_selected)} selected episodes for split={self.split}: first_missing={missing_selected[:8].tolist()}"
            )
        self.force_vq_sample_indices = sample_indices
        self.force_vq_padding_mask = np.zeros(
            (self.num_episodes, self.force_vq_prompt_tokens), dtype=bool
        )
        self.force_vq_valid_episode_mask = valid
        self.force_vq_ref_episode_indices = np.flatnonzero(
            self.episode_mask & valid
        ).astype(np.int64)
        self.force_vq_ref_episode_indices_by_behavior = {}
        prompt_source = "same_root_other_episode"
        if self.behavior_ids is not None:
            prompt_source = "same_behavior"
            for behavior_id in np.unique(
                self.behavior_ids[self.force_vq_ref_episode_indices]
            ):
                behavior_id = int(behavior_id)
                refs = self.force_vq_ref_episode_indices[
                    self.behavior_ids[self.force_vq_ref_episode_indices] == behavior_id
                ]
                self.force_vq_ref_episode_indices_by_behavior[behavior_id] = (
                    refs.astype(np.int64)
                )
        if len(self.force_vq_ref_episode_indices) < 2:
            print(
                "[ZarrDataset] warning: only one valid selected contact encoder episode; reference prompt will fall back to current episode."
            )
        print(
            f"[ZarrDataset] loaded reference cache: {cache_path}, source={prompt_source}"
        )

    def _select_force_vq_reference_episode(self, ep_idx: int) -> int:
        """Select a reference prompt according to the configured episode-level policy."""
        ep_idx = int(ep_idx)
        if self.behavior_ids is not None:
            behavior_id = int(self.behavior_ids[ep_idx])
            refs = self.force_vq_ref_episode_indices_by_behavior.get(behavior_id)
        else:
            refs = self.force_vq_ref_episode_indices
        if refs is None or len(refs) < 2:
            return ep_idx
        if self.split == "train":
            candidates = refs[refs != ep_idx]
            if len(candidates) == 0:
                return ep_idx
            return int(candidates[np.random.randint(0, len(candidates))])
        pos = int(np.searchsorted(refs, ep_idx, side="right"))
        ref_ep_idx = int(refs[pos % len(refs)])
        if ref_ep_idx == ep_idx and len(refs) > 1:
            ref_ep_idx = int(refs[(pos + 1) % len(refs)])
        return ref_ep_idx

    def _maybe_build_trainable_force_vq_references(self) -> None:
        """Load cached sample indices as FPT inputs for the trainable reference encoder."""
        if self.force_vq_cache_zarr is None:
            return
        if self.force_vq_sample_indices is None:
            raise ValueError(
                "trainable continuous contact encoder requires cache data/sample_indices."
            )
        baseline_mode = str(
            self.force_vq_cache_zarr.attrs.get("force_baseline_mode", "none")
        ).lower()
        if baseline_mode != "none":
            raise ValueError(
                f"trainable continuous contact encoder currently requires force_baseline_mode='none'; cache uses {baseline_mode!r}."
            )
        sample_count = int(self.force_vq_sample_indices.shape[1])
        state_dim = int(self.data_group[self.state_key].shape[-1])
        force_dim = int(self.data_group[self.force_key].shape[-1])
        pos = np.zeros((self.num_episodes, sample_count, 3), dtype=np.float32)
        force = np.zeros((self.num_episodes, sample_count, force_dim), dtype=np.float32)
        phase = np.zeros((self.num_episodes, sample_count), dtype=np.float32)
        tactile = None
        for ep_idx in self.force_vq_ref_episode_indices:
            ep_start, ep_end = self.episode_bounds(int(ep_idx))
            state = self._read_array(
                self.state_key, slice(ep_start, ep_end), dtype=np.float32
            )
            episode_force = self._read_array(
                self.force_key, slice(ep_start, ep_end), dtype=np.float32
            )
            if (
                state.shape[0] < 1
                or state.shape[0] != episode_force.shape[0]
                or state_dim < 3
            ):
                raise ValueError(f"invalid contact encoder reference episode {ep_idx}")
            index = np.asarray(self.force_vq_sample_indices[ep_idx], dtype=np.int64)
            if (index < 0).any() or (index >= state.shape[0]).any():
                raise ValueError(
                    f"invalid contact encoder sample indices for episode {ep_idx}"
                )
            pos[ep_idx] = state[index, :3] - state[:1, :3]
            force[ep_idx] = episode_force[index, :force_dim]
            phase[ep_idx] = index.astype(np.float32) / float(max(state.shape[0] - 1, 1))
            tactile_episode = self._read_tactile_indices(ep_start + index)
            if tactile is None:
                tactile = np.zeros(
                    (self.num_episodes, sample_count, *tactile_episode.shape[1:]),
                    dtype=np.float32,
                )
            tactile[ep_idx] = tactile_episode
        if tactile is None:
            raise ValueError(
                "trainable continuous contact encoder could not read tactile references"
            )
        self.force_vq_reference_pos = pos
        self.force_vq_reference_force = force
        self.force_vq_reference_tactile = tactile
        self.force_vq_reference_phase = phase
        print(
            f"[ZarrDataset] built trainable continuous contact encoder references: pos={pos.shape} force={force.shape} tactile={(None if tactile is None else tactile.shape)}"
        )

    def _maybe_preload_arrays(self) -> None:
        if not self.preload_to_ram:
            return
        keys = [self.state_key, self.action_key]
        if self.future_force_steps > 0 or self.force_steps > 0:
            keys.append(self.force_key)
        if self.tactile_steps > 0:
            keys.extend([self.tactile_left_key, self.tactile_right_key])
        if self.n_image_steps > 0 and self.cached_image_backbone_feat is None:
            keys.extend(self.image_keys)
        total_gb = 0.0
        print("[ZarrDataset] preloading zarr keys into RAM...")
        for key in dict.fromkeys(keys):
            arr = np.asarray(self.data_group[key][:])
            self.ram_data[key] = arr
            arr_gb = arr.nbytes / 1024**3
            total_gb += arr_gb
            print(
                f"[ZarrDataset] loaded {key}: shape={arr.shape}, dtype={arr.dtype}, size={arr_gb:.3f} GB"
            )
        if self.cached_image_backbone_feat is not None:
            arr = np.asarray(self.latent_cache_group["image_backbone_feat"][:])
            total_gb += arr.nbytes / 1024**3
            print(
                f"[ZarrDataset] loaded latent cache image_backbone_feat: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes / 1024 ** 3:.3f} GB"
            )
            self.cached_image_backbone_feat = arr
        print(f"[ZarrDataset] total RAM preload: {total_gb:.3f} GB")

    def _build_windows(self) -> List[Tuple[int, int, int]]:
        windows: List[Tuple[int, int, int]] = []
        cond_len = max(
            self.window_size, self.n_image_steps, self.force_steps, self.tactile_steps
        )
        for ep_idx, (ep_start, ep_end) in enumerate(
            zip(self.episode_starts, self.episode_ends)
        ):
            if not self.episode_mask[ep_idx]:
                continue
            ep_start = int(ep_start)
            ep_end = int(ep_end)
            first_t = ep_start + cond_len - 1
            last_t = min(
                ep_end - self.action_window_size,
                ep_end - self.future_force_offset - self.future_force_steps,
            )
            if last_t < first_t:
                continue
            for t in range(first_t, last_t + 1, self.stride):
                windows.append((t, ep_end, ep_idx))
        return windows

    @staticmethod
    def _reshape_tactile(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[1] == 700:
            arr = arr.reshape(arr.shape[0], 35, 20, arr.shape[-1])
        if arr.ndim != 4:
            raise ValueError(f"Unsupported tactile shape: {arr.shape}")
        return arr

    def _read_tactile_window(self, obs_end: int) -> np.ndarray:
        tac_start = int(obs_end - self.tactile_steps)
        left = self._reshape_tactile(
            self._read_array(
                self.tactile_left_key, slice(tac_start, obs_end), dtype=np.float32
            )
        )
        right = self._reshape_tactile(
            self._read_array(
                self.tactile_right_key, slice(tac_start, obs_end), dtype=np.float32
            )
        )
        if left.shape[0] != self.tactile_steps or right.shape[0] != self.tactile_steps:
            raise ValueError(
                f"Tactile history length mismatch: left={left.shape[0]}, right={right.shape[0]}, expected={self.tactile_steps}"
            )
        return np.concatenate([left, right], axis=-1)

    def _read_force_window(self, obs_end: int) -> np.ndarray:
        force_start = int(obs_end - self.force_steps)
        force = self._read_array(
            self.force_key, slice(force_start, obs_end), dtype=np.float32
        )
        if force.shape[0] != self.force_steps:
            raise ValueError(
                f"Force history length mismatch: force={force.shape[0]}, expected={self.force_steps}"
            )
        return force

    def _read_tactile_indices(self, idx: np.ndarray) -> np.ndarray:
        left = self._reshape_tactile(
            self._read_array(self.tactile_left_key, idx, dtype=np.float32)
        )
        right = self._reshape_tactile(
            self._read_array(self.tactile_right_key, idx, dtype=np.float32)
        )
        return np.concatenate([left, right], axis=-1)

    def _read_array(self, key: str, slc=None, dtype=None) -> np.ndarray:
        if key in self.ram_data:
            base = self.ram_data[key]
            out = base if slc is None else base[slc]
        else:
            base = self.data_group[key]
            out = base[:] if slc is None else base[slc]
        if dtype is not None:
            out = np.asarray(out, dtype=dtype)
        else:
            out = np.asarray(out)
        return out

    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        arr = np.asarray(img)
        single_frame = arr.ndim == 3
        if single_frame:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        arr = arr[..., ::-1].copy()
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            xf = F.interpolate(
                x.float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.image_as_uint8:
                out = xf.round().clamp_(0.0, 255.0).to(torch.uint8)
            else:
                out = xf.div_(255.0).mul_(2.0).sub_(1.0)
            return out[0] if single_frame else out
        if self.image_as_uint8:
            return x[0] if single_frame else x
        out = x.float().div_(255.0).mul_(2.0).sub_(1.0)
        return out[0] if single_frame else out

    def _transform_action(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self.action_representation == "absolute":
            return action
        if state.shape[-1] < self.action_dim:
            raise ValueError(
                f"chunk_relative action representation requires state_dim >= action_dim, got state_dim={state.shape[-1]}, action_dim={self.action_dim}"
            )
        base_absolute_action = np.asarray(
            state[-1, : self.action_dim], dtype=np.float32
        )
        return absolute_actions_to_relative_actions(
            action, base_absolute_action=base_absolute_action
        )

    def _obs_window_indices(self, anchor_t: int, ep_idx: int) -> tuple[int, int]:
        obs_start = int(anchor_t - self.window_size + 1)
        obs_end = int(anchor_t + 1)
        ep_start = int(self.episode_starts[ep_idx])
        ep_end = int(self.episode_ends[ep_idx])
        if obs_start < ep_start or obs_end > ep_end:
            raise ValueError(
                f"Observation window out of episode bounds: window=({obs_start},{obs_end}), ep_bounds=({ep_start},{ep_end})"
            )
        return (obs_start, obs_end)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def num_episodes(self) -> int:
        return int(len(self.episode_ends))

    def episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        ep_idx = int(ep_idx)
        return (int(self.episode_starts[ep_idx]), int(self.episode_ends[ep_idx]))

    def get_episode_action_raw(self, ep_idx: int) -> np.ndarray:
        ep_start, ep_end = self.episode_bounds(ep_idx)
        return self._read_array(
            self.action_key, slice(ep_start, ep_end), dtype=np.float32
        )[..., : self.action_dim]

    def get_episode_images_raw(self, ep_idx: int) -> np.ndarray:
        ep_start, ep_end = self.episode_bounds(ep_idx)
        return np.stack(
            [self._read_array(key, slice(ep_start, ep_end)) for key in self.image_keys],
            axis=1,
        )

    @staticmethod
    def _subsample_rows(arr: np.ndarray, max_rows: int | None) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if max_rows is None or max_rows <= 0:
            return x.reshape(-1, x.shape[-1])
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] <= max_rows:
            return flat
        idx = np.linspace(0, flat.shape[0] - 1, num=max_rows, dtype=np.int64)
        return flat[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_t, episode_end, ep_idx = self.windows[idx]
        obs_start, obs_end = self._obs_window_indices(anchor_t, ep_idx)
        state_current = self._read_array(
            self.state_key, slice(obs_start, obs_end), dtype=np.float32
        )
        if state_current.shape[0] != self.window_size:
            raise ValueError(
                f"Condition history length mismatch: state={state_current.shape[0]}, expected={self.window_size}"
            )
        action_start = int(anchor_t)
        action_end = int(anchor_t + self.action_window_size)
        if action_end > int(episode_end):
            raise ValueError(
                f"Action horizon exceeds episode end: action=({action_start},{action_end}), episode_end={episode_end}"
            )
        action = self._read_array(
            self.action_key, slice(action_start, action_end), dtype=np.float32
        )[..., : self.action_dim]
        if action.shape[0] != self.action_window_size:
            raise ValueError(
                f"Action horizon mismatch: got {action.shape[0]}, expected {self.action_window_size}"
            )
        action = self._transform_action(action, state_current)
        state = state_current
        obs: Dict[str, torch.Tensor] = {"state": torch.from_numpy(state)}
        if self.force_steps > 0:
            obs["force"] = torch.from_numpy(self._read_force_window(obs_end))
        if self.tactile_steps > 0:
            obs["tactile"] = torch.from_numpy(self._read_tactile_window(obs_end))
        if self.n_image_steps > 0:
            if self.cached_image_backbone_feat is not None:
                image_backbone_feat = np.asarray(
                    self.cached_image_backbone_feat[idx], dtype=np.float32
                )
                if image_backbone_feat.ndim == 2:
                    image_backbone_feat = image_backbone_feat[None, :, :]
                obs["image_backbone_feat"] = torch.from_numpy(image_backbone_feat)
            else:
                image_start = int(anchor_t - self.n_image_steps + 1)
                image_end = int(anchor_t + 1)
                ep_start = int(self.episode_starts[ep_idx])
                if image_start < ep_start:
                    raise ValueError(
                        f"Image history start out of episode bounds: image_start={image_start}, ep_start={ep_start}"
                    )
                images = []
                for key in self.image_keys:
                    image = self._read_array(key, slice(image_start, image_end))
                    if image.shape[0] != self.n_image_steps:
                        raise ValueError(
                            f"Image history length mismatch for {key}: got {image.shape[0]}, expected {self.n_image_steps}"
                        )
                    images.append(self._process_image(image))
                obs["image"] = torch.stack(images, dim=1)
        out = {"obs": obs, "action": torch.from_numpy(action)}
        if self.behavior_ids is not None:
            out["behavior_id"] = torch.tensor(
                int(self.behavior_ids[ep_idx]), dtype=torch.long
            )
        if self.force_vq_sample_indices is not None:
            if self.force_vq_valid_episode_mask is not None and (
                not bool(self.force_vq_valid_episode_mask[ep_idx])
            ):
                raise ValueError(
                    f"contact encoder prompt cache has invalid episode index: {ep_idx}"
                )
            ref_ep_idx = self._select_force_vq_reference_episode(ep_idx)
            out["force_vq_padding_mask"] = torch.from_numpy(
                self.force_vq_padding_mask[ref_ep_idx]
            )
            out["force_vq_source_episode_index"] = torch.tensor(
                int(ref_ep_idx), dtype=torch.long
            )
            out["episode_index"] = torch.tensor(int(ep_idx), dtype=torch.long)
        if self.future_force_steps > 0:
            future_start = int(anchor_t + self.future_force_offset)
            future_end = int(future_start + self.future_force_steps)
            future_force = self._read_array(
                self.force_key, slice(future_start, future_end), dtype=np.float32
            )
            if future_force.shape[0] != self.future_force_steps:
                raise ValueError(
                    f"Future force horizon mismatch: got {future_force.shape[0]}, expected {self.future_force_steps}"
                )
            out["future_force"] = torch.from_numpy(future_force)
        return out

    def sample_normalizer_arrays(
        self, max_rows: int | None = None
    ) -> Dict[str, np.ndarray]:
        """Collect raw arrays used to fit field normalizers (no fitting)."""
        out: Dict[str, np.ndarray] = {}
        raw_action = self._read_array(self.action_key, dtype=np.float32)[
            ..., : self.action_dim
        ]
        state = self._read_array(self.state_key, dtype=np.float32)
        selected_idx = self._selected_frame_indices(max_rows=max_rows)
        if self.action_representation == "absolute":
            action = raw_action[selected_idx]
        else:
            n_win = len(self.windows)
            take = n_win if max_rows is None else min(n_win, max(256, int(max_rows)))
            step = max(1, n_win // take)
            chunks = []
            for i in range(0, n_win, step):
                anchor_t, episode_end, ep_idx = self.windows[i]
                action_start = int(anchor_t)
                action_end = int(anchor_t + self.action_window_size)
                if action_end > int(episode_end):
                    continue
                obs_start, obs_end = self._obs_window_indices(anchor_t, ep_idx)
                state_hist = state[obs_start:obs_end]
                action_chunk = raw_action[action_start:action_end]
                if (
                    state_hist.shape[0] != self.window_size
                    or action_chunk.shape[0] != self.action_window_size
                ):
                    continue
                chunks.append(self._transform_action(action_chunk, state_hist))
                if len(chunks) >= take:
                    break
            if not chunks:
                raise ValueError(
                    f"No relative action chunks for normalizer in {self.root_dir}"
                )
            action = np.asarray(chunks, dtype=np.float32)
        out["action"] = self._subsample_rows(action, max_rows)
        out["state"] = self._subsample_rows(state[selected_idx], max_rows)
        if self.future_force_steps > 0 or self.force_steps > 0:
            force = self._read_array(self.force_key, dtype=np.float32)
            out["force"] = self._subsample_rows(force[selected_idx], max_rows)
        if self.tactile_steps > 0:
            tactile = self._sample_tactile_frames(max_frames=min(1024, int(True)))
            out["tactile"] = tactile.reshape(-1, tactile.shape[-1])
        return out

    def _selected_frame_indices(self, max_rows: int | None = None) -> np.ndarray:
        parts = [
            np.arange(int(start), int(end), dtype=np.int64)
            for start, end, selected in zip(
                self.episode_starts, self.episode_ends, self.episode_mask
            )
            if selected
        ]
        if not parts:
            raise ValueError(
                f"No selected frames for split={self.split} in {self.root_dir}"
            )
        indices = np.concatenate(parts)
        if max_rows is not None and max_rows > 0 and (len(indices) > int(max_rows)):
            take = np.linspace(0, len(indices) - 1, num=int(max_rows), dtype=np.int64)
            indices = indices[take]
        return indices

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        arrays = self.sample_normalizer_arrays(max_rows=max_rows)
        normalizer = MultiFieldNormalizer()
        for key, arr in arrays.items():
            if key == "force":
                normalizer[key], _ = fit_limited_normalizer(
                    arr,
                    min_range=FORCE_NORM_MIN_RANGE,
                    max_abs_scale=FORCE_NORM_MAX_ABS_SCALE,
                )
            else:
                normalizer[key] = FieldNormalizer.from_data_limits(arr)
        return normalizer

    def _sample_tactile_frames(self, max_frames: int = 1024) -> np.ndarray:
        """Evenly sample tactile frames (both hands, channel-concatenated) for normalizer stats."""
        hands = []
        indices = self._selected_frame_indices(max_rows=max_frames)
        for key in (self.tactile_left_key, self.tactile_right_key):
            base = self.ram_data.get(key, self.data_group[key])
            hands.append(np.asarray(base[indices], dtype=np.float32))
        return np.concatenate(hands, axis=-1)


class ConcatZarrDataset(Dataset):
    """Concatenate multiple ZarrDataset roots (equal window sampling by concatenation)."""

    def __init__(self, datasets: Sequence):
        if not datasets:
            raise ValueError("ConcatZarrDataset requires at least one dataset")
        self.datasets = list(datasets)
        lengths = [len(ds) for ds in self.datasets]
        self._cum = np.cumsum([0, *lengths]).astype(np.int64)
        offsets = []
        offset = 0
        for ds in self.datasets:
            offsets.append(offset)
            refs = getattr(ds, "force_vq_reference_pos", None)
            if refs is not None:
                offset += int(refs.shape[0])
        self._force_vq_ref_offsets = offsets

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
        offset = int(self._force_vq_ref_offsets[ds_idx])
        if offset and "force_vq_source_episode_index" in item:
            item["force_vq_source_episode_index"] = (
                item["force_vq_source_episode_index"] + offset
            )
        return item

    @property
    def action_representation(self) -> str:
        return self.datasets[0].action_representation

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        max_rows = int(max_rows) if max_rows is not None else 500000
        per_root = max(256, max_rows // max(1, len(self.datasets)))
        print(
            f"[ConcatZarrDataset] fitting normalizer on {len(self.datasets)} roots (~{per_root} rows/root, cap={max_rows})"
        )
        buckets: Dict[str, list] = {}
        for ds in self.datasets:
            arrays = ds.sample_normalizer_arrays(max_rows=per_root)
            for k, arr in arrays.items():
                buckets.setdefault(k, []).append(np.asarray(arr, dtype=np.float32))
        normalizer = MultiFieldNormalizer()
        for k, parts in buckets.items():
            merged = np.concatenate([p.reshape(-1, p.shape[-1]) for p in parts], axis=0)
            if merged.shape[0] > max_rows:
                idx = np.linspace(0, merged.shape[0] - 1, num=max_rows, dtype=np.int64)
                merged = merged[idx]
            if k == "force":
                normalizer[k], _ = fit_limited_normalizer(
                    merged,
                    min_range=FORCE_NORM_MIN_RANGE,
                    max_abs_scale=FORCE_NORM_MAX_ABS_SCALE,
                )
            else:
                normalizer[k] = FieldNormalizer.from_data_limits(merged)
            print(f"  field={k} samples={merged.shape[0]} dim={merged.shape[-1]}")
        return normalizer


def collect_force_vq_reference_tables(
    dataset, *, include_phase: bool = False
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
    | tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]
):
    datasets = (
        list(dataset.datasets) if isinstance(dataset, ConcatZarrDataset) else [dataset]
    )
    pos_parts, force_parts, tactile_parts, phase_parts = ([], [], [], [])
    for ds in datasets:
        if getattr(ds, "force_vq_reference_pos", None) is None:
            continue
        pos_parts.append(np.asarray(ds.force_vq_reference_pos))
        force_parts.append(np.asarray(ds.force_vq_reference_force))
        if include_phase:
            phase = getattr(ds, "force_vq_reference_phase", None)
            if phase is None:
                raise ValueError(
                    "trainable continuous contact encoder requires reference phase tables."
                )
            phase_parts.append(np.asarray(phase))
        if getattr(ds, "force_vq_reference_tactile", None) is not None:
            tactile_parts.append(np.asarray(ds.force_vq_reference_tactile))
    if not pos_parts:
        raise ValueError(
            "trainable continuous contact encoder reference tables are empty."
        )
    if tactile_parts and len(tactile_parts) != len(pos_parts):
        raise ValueError(
            "contact encoder reference tactile missing on some concatenated datasets."
        )
    tactile = np.concatenate(tactile_parts, axis=0) if tactile_parts else None
    result = (
        np.concatenate(pos_parts, axis=0),
        np.concatenate(force_parts, axis=0),
        tactile,
    )
    if include_phase:
        return (*result, np.concatenate(phase_parts, axis=0))
    return result


def build_zarr_dataset(split: str = "train", **kwargs) -> Dataset:
    """Build one ZarrDataset, or ConcatZarrDataset when root_dir is a list."""
    kwargs.pop("base_path", None)
    root_dir = kwargs.pop("root_dir")
    cache_root = kwargs.pop("latent_cache_root_dir", None)
    force_vq_cache_root = kwargs.pop("force_vq_cache_root_dir", None)
    roots = [root_dir] if isinstance(root_dir, (str, os.PathLike)) else list(root_dir)
    roots = [str(r) for r in roots]
    if not roots:
        raise ValueError("root_dir is empty")
    cache_list: list[str | None] | None = None
    if isinstance(cache_root, (list, tuple)):
        cache_list = [None if c is None else str(c) for c in cache_root]
        if len(cache_list) != len(roots):
            raise ValueError(
                f"latent_cache_root_dir list length {len(cache_list)} != root_dir length {len(roots)}"
            )
    per_root_cache = False
    if isinstance(cache_root, str) and cache_root.lower() in {"auto", "true", "1"}:
        per_root_cache = True
        cache_root = "auto"
    force_vq_cache_list: list[str | None] | None = None
    if isinstance(force_vq_cache_root, (list, tuple)):
        force_vq_cache_list = [
            None if c is None else str(c) for c in force_vq_cache_root
        ]
        if len(force_vq_cache_list) != len(roots):
            raise ValueError(
                f"force_vq_cache_root_dir list length {len(force_vq_cache_list)} != root_dir length {len(roots)}"
            )
    per_root_force_vq_cache = False
    if isinstance(force_vq_cache_root, str) and force_vq_cache_root.lower() in {
        "auto",
        "true",
        "1",
    }:
        per_root_force_vq_cache = True
        force_vq_cache_root = "auto"
    datasets: List[ZarrDataset] = []
    for i, root in enumerate(roots):
        kw = dict(kwargs)
        kw["root_dir"] = root
        if cache_list is not None:
            kw["latent_cache_root_dir"] = cache_list[i]
        elif cache_root is None:
            kw["latent_cache_root_dir"] = None
        elif per_root_cache or len(roots) > 1:
            kw["latent_cache_root_dir"] = root
        else:
            kw["latent_cache_root_dir"] = str(cache_root)
        if force_vq_cache_list is not None:
            kw["force_vq_cache_root_dir"] = force_vq_cache_list[i]
        elif force_vq_cache_root is None:
            kw["force_vq_cache_root_dir"] = None
        elif per_root_force_vq_cache or len(roots) > 1:
            kw["force_vq_cache_root_dir"] = root
        else:
            kw["force_vq_cache_root_dir"] = str(force_vq_cache_root)
        datasets.append(ZarrDataset(split=split, **kw))
    if len(datasets) == 1:
        return datasets[0]
    print(f"[build_zarr_dataset] concat {len(datasets)} roots for split={split}:")
    for ds in datasets:
        print(f"  - {ds.root_dir}: {len(ds)} windows")
    return ConcatZarrDataset(datasets)


def policy_dataset_kwargs(cfg: dict) -> dict:
    """Build aligned observation histories and one-step-ahead force targets."""
    data = dict(cfg["data"])
    data.pop("force_vq_prompt_representation", None)
    if data.get("split_seed") is None:
        data["split_seed"] = int(cfg.get("seed", 42))
    policy = cfg["model"]["policy"]
    data["force_steps"] = data["tactile_steps"] = int(data["window_size"])
    data["future_force_steps"] = int(policy["force_consistency"]["horizon"])
    data["future_force_offset"] = 1
    data["force_vq_prompt_tokens"] = int(
        policy["mode_prompt"]["force_vq"].get("num_tokens", 16)
    )
    return data
