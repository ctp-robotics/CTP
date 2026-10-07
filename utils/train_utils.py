from __future__ import annotations

import os
import random
import re
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch
import yaml

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")
_MISSING = object()
_SECTION_KEYS = ("runtime", "output", "data", "model", "checkpoint")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_by_path(cfg: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            if default is _MISSING:
                raise KeyError(path)
            return default
        cur = cur[part]
    return cur


def _set_by_path(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def cfg_get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    return _get_by_path(cfg, path, default)


def _expand_cfg_strings_inplace(cfg: dict) -> None:
    def expand_str(s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            key = m.group(1).strip()
            value = _get_by_path(cfg, key, _MISSING)
            if value is _MISSING:
                raise KeyError(f"Unknown config placeholder '${{{key}}}'")
            if isinstance(value, (dict, list)) or value is None:
                raise TypeError(
                    f"Placeholder '${{{key}}}' must resolve to scalar, got {type(value)}"
                )
            return str(value)

        return _PLACEHOLDER.sub(repl, s)

    def walk(x: Any):
        if isinstance(x, str) and "${" in x:
            return expand_str(x)
        if isinstance(x, dict):
            for k in list(x.keys()):
                x[k] = walk(x[k])
            return x
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = walk(x[i])
            return x
        return x

    walk(cfg)


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str, _stack: tuple[str, ...] = ()) -> dict:
    path = os.path.abspath(path)
    if path in _stack:
        chain = " -> ".join((*_stack, path))
        raise ValueError(f"Circular base_config chain: {chain}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(
            f"Config at {path} must be a YAML mapping, got {type(cfg).__name__}."
        )
    base_path = cfg.pop("base_config", None)
    if base_path is None:
        return cfg
    if not isinstance(base_path, str):
        raise TypeError(f"base_config in {path} must be a path string.")
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(path), base_path)
    base = load_yaml(base_path, _stack=(*_stack, path))
    return _deep_merge(base, cfg)


def _validate_mapping_section(cfg: Mapping[str, Any], path: str) -> None:
    value = _get_by_path(cfg, path, _MISSING)
    if value is _MISSING:
        raise KeyError(f"Missing required config section '{path}'.")
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Config section '{path}' must be a mapping, got {type(value).__name__}."
        )


def _validate_canonical_config(cfg: Mapping[str, Any]) -> None:
    for key in ("seed", *_SECTION_KEYS):
        if key not in cfg:
            raise KeyError(f"Missing required top-level config key '{key}'.")

    for section in _SECTION_KEYS:
        _validate_mapping_section(cfg, section)

    stage = cfg.get("stage", "policy")
    if stage not in {"pretrain", "policy"}:
        raise ValueError("stage must be 'pretrain' or 'policy'.")
    training_section = "train_vq" if stage == "pretrain" else "train"
    _validate_mapping_section(cfg, f"{training_section}.optimizer")
    _validate_mapping_section(
        cfg, "model.force_vq" if stage == "pretrain" else "model.policy"
    )

    for path in (
        "runtime.device",
        "output.root_dir",
        "output.run_name",
        "data.root_dir",
        f"{training_section}.batch_size",
        f"{training_section}.val_batch_size",
        f"{training_section}.epochs",
        "checkpoint.save_every",
    ):
        _get_by_path(cfg, path)

    if stage == "policy":
        _get_by_path(cfg, "data.window_size")
        _get_by_path(cfg, "checkpoint.val_every")
        _get_by_path(cfg, "checkpoint.best_metric")


def validate_training_paths(cfg: Mapping[str, Any]) -> None:
    """Validate user-supplied paths at training time, not checkpoint load time."""
    roots = cfg["data"]["root_dir"]
    if isinstance(roots, str):
        roots = [roots]
    if (
        not isinstance(roots, (list, tuple))
        or not roots
        or any(not isinstance(root, str) or not root.strip() for root in roots)
    ):
        raise ValueError(
            "Set data.root_dir to a nonempty list of dataset paths in your config."
        )
    for root in roots:
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if cfg.get("stage") == "policy":
        checkpoint = cfg["model"]["policy"]["mode_prompt"]["force_vq"].get("ckpt")
        if not checkpoint:
            raise ValueError(
                "Set model.policy.reference_encoder.checkpoint to the pretrained contact checkpoint."
            )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Pretrained contact checkpoint does not exist: {checkpoint}"
            )


def _apply_runtime_overrides(
    cfg: dict[str, Any], overrides: Mapping[str, Any] | None
) -> None:
    if not overrides:
        return
    for path, value in overrides.items():
        if value is not None:
            _set_by_path(cfg, path, value)


def _resolve_release_config(cfg: dict[str, Any]) -> None:
    """Resolve the two public recipes to the checkpoint-compatible model fields.

    Old checkpoints already contain the resolved fields and bypass this step.
    Policy encoder architecture comes from its pretrained checkpoint; it is not
    independently configured a second time in policy.yaml.
    """
    if cfg.get("stage") == "posttrain":
        cfg["stage"] = "policy"
    model = cfg.get("model", {})
    if "contact_encoder" in model:
        if "force_vq" in model or "train_vq" in cfg:
            raise ValueError(
                "Do not mix public and resolved pretraining configuration fields."
            )
        model["force_vq"] = model.pop("contact_encoder")
        cfg["train_vq"] = cfg.pop("train")

    policy = model.get("policy", {})
    if "reference_encoder" not in policy:
        return
    if "mode_prompt" in policy or "force_consistency" in policy or "force_vq" in model:
        raise ValueError("Do not mix public and resolved policy configuration fields.")
    reference = policy.pop("reference_encoder")
    unknown = set(reference) - {"checkpoint"}
    if unknown:
        raise ValueError(f"Unknown reference_encoder fields: {sorted(unknown)}")
    policy["mode_prompt"] = {
        "force_vq": {
            "ckpt": reference.get("checkpoint"),
            "num_tokens": 16,
            "latent_dim": 128,
        },
    }
    data = cfg["data"]
    if "force_vq_cache_root_dir" in data or "force_vq_prompt_representation" in data:
        raise ValueError(
            "Use data.reference_cache_root_dir in the public policy configuration."
        )
    data["force_vq_cache_root_dir"] = data.pop("reference_cache_root_dir", "auto")
    data["action_dim"] = int(policy["action_dim"])
    policy["action_representation"] = data["action_representation"]
    force = policy.pop("force_prediction")
    unknown = set(force) - {"force_indices", "horizon", "hidden_dim", "loss_weight"}
    if unknown:
        raise ValueError(f"Unknown force_prediction fields: {sorted(unknown)}")
    policy["force_consistency"] = {
        "force_indices": force["force_indices"],
        "horizon": force["horizon"],
        "hidden_dim": force.get("hidden_dim", 256),
        "force_loss_weight": force.get("loss_weight", 0.1),
    }


def _synchronize_temporal_config(cfg: dict[str, Any]) -> None:
    """Resolve duplicated temporal settings and reject silent target truncation."""
    data = cfg.get("data")
    policy = (cfg.get("model") or {}).get("policy")
    if not isinstance(data, dict) or not isinstance(policy, dict):
        return

    if "window_size" in data:
        state_steps = int(data["window_size"])
        configured_steps = policy.get("curr_steps")
        if configured_steps is not None and int(configured_steps) != state_steps:
            raise ValueError(
                "data.window_size and model.policy.curr_steps must match: "
                f"got {state_steps} and {configured_steps}. "
                "State history has one source of truth: data.window_size."
            )
        policy["curr_steps"] = state_steps

    if "action_window_size" in data:
        target_horizon = int(data["action_window_size"])
        policy_horizon = int(policy.get("action_horizon", target_horizon))
        if policy_horizon != target_horizon:
            raise ValueError(
                "data.action_window_size and model.policy.action_horizon must match: "
                f"got {target_horizon} and {policy_horizon}."
            )
        policy["action_horizon"] = target_horizon


def build_canonical_config(
    source: str | Mapping[str, Any], overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if isinstance(source, str):
        raw = load_yaml(source)
    else:
        raw = deepcopy(source)

    if not isinstance(raw, Mapping):
        raise TypeError(f"Config source must be a mapping, got {type(raw).__name__}.")

    cfg = deepcopy(dict(raw))
    _apply_runtime_overrides(cfg, overrides)
    _expand_cfg_strings_inplace(cfg)
    _resolve_release_config(cfg)
    _synchronize_temporal_config(cfg)
    _validate_canonical_config(cfg)
    return cfg


def _config_items(cfg: Any):
    if isinstance(cfg, Mapping):
        return cfg.items()
    return vars(cfg).items()


def log_hparams_to_tensorboard(writer, cfg: Any, log_dir: str) -> None:
    if writer is None:
        return

    writer.add_text("meta/log_dir", log_dir, 0)
    for k, v in _config_items(cfg):
        if isinstance(v, (dict, list)):
            text = yaml.safe_dump(v, sort_keys=False)
        else:
            text = str(v)
        writer.add_text(f"hparams/{k}", text, 0)


def detach_scalar_dict(d: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                out[k] = v.detach().item()
            else:
                out[k] = v.detach().float().mean().item()
        else:
            out[k] = float(v)
    return out
