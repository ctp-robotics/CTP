from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class FieldNormalizer:
    scale: torch.Tensor
    offset: torch.Tensor

    @classmethod
    def from_data_limits(
        cls,
        data: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        eps: float = 1e-6,
    ) -> "FieldNormalizer":
        x = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        x_range = np.maximum(x_max - x_min, eps)
        scale = (output_max - output_min) / x_range
        offset = output_min - scale * x_min
        return cls(
            scale=torch.from_numpy(scale.astype(np.float32)),
            offset=torch.from_numpy(offset.astype(np.float32)),
        )

    def to(self, device: torch.device):
        self.scale = self.scale.to(device)
        self.offset = self.offset.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(x.device, x.dtype)
        offset = self.offset.to(x.device, x.dtype)
        return x * scale + offset

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(x.device, x.dtype)
        offset = self.offset.to(x.device, x.dtype)
        return (x - offset) / scale

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "scale": self.scale.detach().cpu(),
            "offset": self.offset.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "FieldNormalizer":
        return cls(scale=state["scale"], offset=state["offset"])


def fit_limited_normalizer(
    data: np.ndarray,
    *,
    min_range: float,
    max_abs_scale: float,
) -> tuple["FieldNormalizer", dict]:
    """Fit a FieldNormalizer with near-constant dimension protection.

    Dense dimensions whose observed range is below ``min_range`` (e.g. always-zero
    channels) would otherwise get an exploding scale of ``2/eps`` that overflows
    fp16 training. Their scale is bounded by ``max_abs_scale`` instead.
    """
    x = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_range = x_max - x_min
    effective_range = np.maximum(x_range, float(min_range))
    scale = 2.0 / effective_range
    scale_clamped = scale > float(max_abs_scale)
    scale = np.minimum(scale, float(max_abs_scale))
    center = 0.5 * (x_min + x_max)
    offset = -scale * center
    info = {
        "range_min": float(x_range.min()),
        "range_max": float(x_range.max()),
        "scale_max": float(scale.max()),
        "near_constant_dims": np.flatnonzero(x_range < float(min_range)).astype(int).tolist(),
        "scale_clamped_dims": np.flatnonzero(scale_clamped).astype(int).tolist(),
    }
    return (
        FieldNormalizer(
            scale=torch.from_numpy(scale.astype(np.float32)),
            offset=torch.from_numpy(offset.astype(np.float32)),
        ),
        info,
    )


class MultiFieldNormalizer:
    def __init__(self):
        self.fields: Dict[str, FieldNormalizer] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def __getitem__(self, key: str) -> FieldNormalizer:
        return self.fields[key]

    def __setitem__(self, key: str, value: FieldNormalizer):
        self.fields[key] = value

    def to(self, device: torch.device):
        for field in self.fields.values():
            field.to(device)
        return self

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {k: v.state_dict() for k, v in self.fields.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]):
        self.fields = {
            k: FieldNormalizer.from_state_dict(v)
            for k, v in state.items()
            if isinstance(v, dict) and "scale" in v and "offset" in v
        }

    def normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in obs.items():
            if k in self.fields and torch.is_tensor(v):
                out[k] = self.fields[k].normalize(v)
            else:
                out[k] = v
        return out


def save_normalizer(path: str, normalizer: MultiFieldNormalizer, meta: dict | None = None) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"fields": normalizer.state_dict(), "meta": meta or {}}
    torch.save(payload, path)


def load_normalizer(path: str) -> MultiFieldNormalizer:
    normalizer, _ = load_normalizer_with_meta(path)
    return normalizer


def load_normalizer_with_meta(path: str) -> tuple[MultiFieldNormalizer, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    fields = obj["fields"] if isinstance(obj, dict) and "fields" in obj else obj
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    normalizer = MultiFieldNormalizer()
    normalizer.load_state_dict(fields)
    return normalizer, dict(meta) if isinstance(meta, dict) else {}
