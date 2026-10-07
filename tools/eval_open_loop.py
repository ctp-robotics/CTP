from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.policy_zarr_dataset import (
    ZarrDataset,
    build_zarr_dataset,
    policy_dataset_kwargs,
)  # noqa: E402
from utils.action import relative_actions_to_absolute_tensor  # noqa: E402
from utils.checkpoint_util import load_policy_from_checkpoint  # noqa: E402
from utils.tensor import move_to_device  # noqa: E402
from trainers.policy_trainer import set_policy_reference_dataset, _PROMPT_BATCH_KEYS
from utils.train_utils import cfg_get  # noqa: E402


def _dataset_kwargs_from_cfg(
    cfg: dict, data_root_override: Optional[str]
) -> dict[str, Any]:
    data = policy_dataset_kwargs(cfg)
    if data_root_override:
        data["root_dir"] = str(data_root_override)
        data["force_vq_cache_root_dir"] = "auto"
        if data.get("latent_cache_root_dir"):
            data["latent_cache_root_dir"] = "auto"
    return data


def _to_absolute_actions(
    action_seq: torch.Tensor,
    obs_batch: dict[str, torch.Tensor],
    action_representation: str,
) -> torch.Tensor:
    if str(action_representation).lower() == "absolute":
        return action_seq
    if "state" not in obs_batch:
        raise KeyError("chunk_relative conversion requires obs['state'].")

    action_dim = action_seq.shape[-1]
    state = obs_batch["state"]
    if state.shape[-1] < action_dim:
        raise ValueError(
            f"state_dim={state.shape[-1]} smaller than action_dim={action_dim}"
        )

    base = state[:, -1, :action_dim].unsqueeze(1).expand(-1, action_seq.shape[1], -1)
    return relative_actions_to_absolute_tensor(action_seq, base)


def _summarize_metrics(metrics: dict[str, float], prefix: str) -> None:
    parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
    print(f"[{prefix}] " + ", ".join(parts))


@torch.no_grad()
def evaluate_open_loop(
    policy: torch.nn.Module,
    ds: ZarrDataset,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
    num_inference_steps: int,
    solver: str,
) -> dict[str, float]:
    policy.eval()
    action_representation = str(
        getattr(ds, "action_representation", "absolute")
    ).lower()
    policy_action_mse_sum = 0.0
    policy_action_l1_sum = 0.0
    policy_force_mse_sum = 0.0
    policy_force_l1_sum = 0.0
    force_sample_count = 0
    sample_count = 0
    policy_xyz_errors: list[np.ndarray] = []
    predicted_absolute_chunks: list[torch.Tensor] = []
    target_absolute_chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch_d = move_to_device(batch, device)
        obs = dict(batch_d["obs"])
        for key in _PROMPT_BATCH_KEYS:
            if key in batch_d:
                obs[key] = batch_d[key]
        out = policy.predict_action(
            obs,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )

        pred_model = out.get("action_model", out["action"])
        gt_model = batch_d["action"]
        t_model = min(pred_model.shape[1], gt_model.shape[1])
        d_model = min(pred_model.shape[2], gt_model.shape[2])
        if t_model > 0 and d_model > 0:
            pred_model_slice = pred_model[:, :t_model, :d_model]
            gt_model_slice = gt_model[:, :t_model, :d_model]
            bs = gt_model_slice.shape[0]
            policy_action_mse_sum += (
                float(
                    F.mse_loss(
                        pred_model_slice, gt_model_slice, reduction="mean"
                    ).item()
                )
                * bs
            )
            policy_action_l1_sum += (
                float(torch.mean(torch.abs(pred_model_slice - gt_model_slice)).item())
                * bs
            )
            sample_count += bs

        if "force_pred_model" in out and "future_force" in batch_d:
            force_indices = out["force_pred_indices"].to(batch_d["future_force"].device)
            pred_force = out["force_pred_model"]
            gt_force = batch_d["future_force"].index_select(-1, force_indices)
            force_t = min(pred_force.shape[1], gt_force.shape[1])
            if force_t > 0:
                pred_force = pred_force[:, :force_t]
                gt_force = gt_force[:, :force_t]
                bs = gt_force.shape[0]
                policy_force_mse_sum += (
                    float(F.mse_loss(pred_force, gt_force).item()) * bs
                )
                policy_force_l1_sum += (
                    float(F.l1_loss(pred_force, gt_force).item()) * bs
                )
                force_sample_count += bs

        pred_abs = out["action"]
        gt_abs = _to_absolute_actions(
            batch_d["action"], batch_d["obs"], action_representation
        )
        t = min(pred_abs.shape[1], gt_abs.shape[1])
        d = min(pred_abs.shape[2], gt_abs.shape[2], 3)
        if t <= 0 or d <= 0:
            continue

        pred_xyz = pred_abs[:, :t, :d]
        gt_xyz = gt_abs[:, :t, :d]
        policy_err = torch.mean(torch.abs(pred_xyz - gt_xyz), dim=(1, 2))
        policy_xyz_errors.append(policy_err.detach().cpu().numpy())
        # Adjacent dataset windows have an overlapping action horizon. Keep
        # absolute chunks so we can quantify replanning consistency below.
        predicted_absolute_chunks.append(pred_xyz.detach().cpu())
        target_absolute_chunks.append(gt_xyz.detach().cpu())

    metrics = {
        "action_mse": policy_action_mse_sum / max(sample_count, 1),
        "action_l1": policy_action_l1_sum / max(sample_count, 1),
        "policy_chunk_mae_xyz": 0.0,
    }
    if force_sample_count > 0:
        metrics["force_mse"] = policy_force_mse_sum / force_sample_count
        metrics["force_l1"] = policy_force_l1_sum / force_sample_count
    if policy_xyz_errors:
        policy_err_all = np.concatenate(policy_xyz_errors, axis=0).astype(
            np.float64, copy=False
        )
        metrics["policy_chunk_mae_xyz"] = float(np.mean(policy_err_all))
    if predicted_absolute_chunks:
        predicted_all = torch.cat(predicted_absolute_chunks, dim=0)
        target_all = torch.cat(target_absolute_chunks, dim=0)
        if predicted_all.shape[0] >= 2 and predicted_all.shape[1] >= 2:
            # The same physical future instant is predicted by [i, 1:] and
            # [i+1, :-1]. The target quantity reports the small residual from
            # episode boundaries / indexing; the excess is replanning jitter.
            pred_overlap = torch.mean(
                torch.abs(predicted_all[:-1, 1:] - predicted_all[1:, :-1])
            ).item()
            target_overlap = torch.mean(
                torch.abs(target_all[:-1, 1:] - target_all[1:, :-1])
            ).item()
            metrics["replan_overlap_mae_xyz"] = float(pred_overlap)
            metrics["target_overlap_mae_xyz"] = float(target_overlap)
            metrics["excess_replan_jitter_xyz"] = float(
                max(0.0, pred_overlap - target_overlap)
            )
    return metrics


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-loop evaluation for a trained flow-matching policy."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Policy checkpoint path"
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="Optional dataset root override"
    )
    parser.add_argument(
        "--split", type=str, default="both", choices=["train", "val", "both"]
    )
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument(
        "--solver", type=str, default="euler", choices=["euler", "heun"]
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional loader batch size override",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override; use 0 for deterministic prompt sampling.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed for reproducible flow-matching evaluation.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = str(args.checkpoint).strip()
    policy, cfg = load_policy_from_checkpoint(ckpt_path, device)
    data_kwargs = _dataset_kwargs_from_cfg(cfg, args.data_root)

    batch_size = int(
        args.batch_size
        or cfg_get(cfg, "train.val_batch_size", cfg_get(cfg, "train.batch_size", 32))
    )
    num_workers = int(
        cfg_get(cfg, "train.num_workers", 0)
        if args.num_workers is None
        else args.num_workers
    )
    if num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg_get(cfg, "train.persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(cfg_get(cfg, "train.prefetch_factor", 4))

    max_batches = max(1, int(args.max_batches))
    common_kwargs = {
        "max_batches": max_batches,
        "num_inference_steps": int(args.num_inference_steps),
        "solver": str(args.solver),
    }

    print(f"[info] checkpoint: {ckpt_path}")
    print(f"[info] device: {device}")
    if args.seed is not None:
        print(f"[info] sampling_seed: {args.seed}")

    if args.split in ("train", "both"):
        train_ds = build_zarr_dataset(split="train", **data_kwargs)
        set_policy_reference_dataset(policy, train_ds)
        train_loader = DataLoader(train_ds, **loader_kwargs)
        print(f"[info] train_windows={len(train_ds)}")
        metrics = evaluate_open_loop(
            policy, train_ds, train_loader, device, **common_kwargs
        )
        _summarize_metrics(metrics, "open_loop/train")

    if args.split in ("val", "both"):
        val_ds = build_zarr_dataset(split="val", **data_kwargs)
        set_policy_reference_dataset(policy, val_ds)
        val_loader = DataLoader(val_ds, **loader_kwargs)
        print(f"[info] val_windows={len(val_ds)}")
        metrics = evaluate_open_loop(
            policy, val_ds, val_loader, device, **common_kwargs
        )
        _summarize_metrics(metrics, "open_loop/val")


if __name__ == "__main__":
    main()
