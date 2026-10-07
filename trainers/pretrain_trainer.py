from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.force_episode_dataset import (
    build_force_episode_dataset,
    collate_force_episodes,
)
from models.policy.contact_autoencoder import ContactAutoencoder
from utils.normalizer import (
    FieldNormalizer,
    MultiFieldNormalizer,
    fit_limited_normalizer,
)
from utils.tensor import move_to_device
from utils.train_utils import cfg_get, detach_scalar_dict, set_seed


def _dataset_kwargs(cfg: dict) -> dict:
    data = cfg["data"]
    keys = (
        "root_dir",
        "force_key",
        "state_key",
        "tactile_left_key",
        "tactile_right_key",
        "val_ratio",
        "preload_to_ram",
    )
    result = {key: data[key] for key in keys if key in data}
    result["split_seed"] = (
        cfg.get("seed", 42) if data.get("split_seed") is None else data["split_seed"]
    )
    return result


def build_vq_model(cfg: dict, device: torch.device) -> ContactAutoencoder:
    return ContactAutoencoder.from_cfg(cfg).to(device)


def _normalize_batch(batch: dict, normalizer):
    force = batch["force"]
    pos = batch["pos"]
    tactile = batch.get("tactile")
    if "force" in normalizer:
        force = normalizer["force"].normalize(force)
    if "mode_pos" in normalizer:
        pos = normalizer["mode_pos"].normalize(pos)
    if tactile is not None and "tactile" in normalizer:
        tactile = normalizer["tactile"].normalize(tactile)
    return pos, force, tactile


def _perplexity(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """Compute codebook usage perplexity across the batch."""
    flat = indices.reshape(-1).to(dtype=torch.long)
    valid = flat[(flat >= 0) & (flat < int(codebook_size))]
    if valid.numel() == 0:
        return flat.new_zeros(())
    hist = torch.bincount(valid, minlength=int(codebook_size)).float()
    prob = hist / hist.sum().clamp_min(1.0)
    return torch.exp(-(prob * (prob + 1e-10).log()).sum())


def _normalizer_cfg(cfg: dict) -> dict:
    norm_cfg = dict(cfg.get("train_vq", {}).get("normalizer") or {})
    return {
        "force_min_range": float(norm_cfg.get("force_min_range", 1.0e-3)),
        "max_abs_scale": float(norm_cfg.get("max_abs_scale", 100.0)),
    }


def _sample_normalizer_arrays(dataset, max_rows: int | None) -> dict[str, np.ndarray]:
    if hasattr(dataset, "datasets"):
        max_rows = int(max_rows) if max_rows is not None else 500000
        per_root = max(256, max_rows // max(1, len(dataset.datasets)))
        buckets: dict[str, list[np.ndarray]] = {}
        for ds in dataset.datasets:
            arrays = ds.sample_normalizer_arrays(max_rows=per_root)
            for key, arr in arrays.items():
                buckets.setdefault(key, []).append(np.asarray(arr, dtype=np.float32))
        out = {}
        for key, parts in buckets.items():
            merged = np.concatenate(parts, axis=0)
            if merged.shape[0] > max_rows:
                idx = np.linspace(0, merged.shape[0] - 1, num=max_rows, dtype=np.int64)
                merged = merged[idx]
            out[key] = merged.astype(np.float32, copy=False)
        return out
    if hasattr(dataset, "sample_normalizer_arrays"):
        return dataset.sample_normalizer_arrays(max_rows=max_rows)
    raise TypeError(
        f"Unsupported dataset type for contact encoder normalizer: {type(dataset).__name__}"
    )


def _fit_force_vq_normalizer(
    dataset, max_rows: int | None, norm_cfg: dict
) -> tuple[MultiFieldNormalizer, dict]:
    arrays = _sample_normalizer_arrays(dataset, max_rows=max_rows)
    normalizer = MultiFieldNormalizer()
    meta = {
        "source": "force_vq_train_fit",
        "fields": sorted(arrays.keys()),
        "force_min_range": float(norm_cfg["force_min_range"]),
        "force_max_abs_scale": float(norm_cfg["max_abs_scale"]),
    }
    for key, arr in arrays.items():
        if key == "force":
            normalizer[key], force_info = fit_limited_normalizer(
                arr,
                min_range=float(norm_cfg["force_min_range"]),
                max_abs_scale=float(norm_cfg["max_abs_scale"]),
            )
            meta["force_stats"] = force_info
        else:
            normalizer[key] = FieldNormalizer.from_data_limits(arr)
        print(f"  field={key} samples={arr.shape[0]} dim={arr.shape[-1]}")
    return normalizer, meta


def _reconstruction_metrics(out: dict) -> dict:
    metrics = {
        "loss": out["loss"],
        "recon_loss": out["recon_loss"],
        "vq_loss": out["vq_loss"],
    }
    for key in (
        "recon_force_loss",
        "recon_pos_loss",
        "recon_tactile_loss",
        "masked_modality_recon_loss",
    ):
        if key in out:
            metrics[key] = out[key]
    return metrics


def train_one_epoch(model, loader, normalizer, optimizer, device, grad_clip=None):
    model.train()
    metric_sum = defaultdict(float)
    count = 0
    pbar = tqdm(loader, desc="Pretrain", leave=False)
    for batch in pbar:
        batch = move_to_device(batch, device)
        pos, force, tactile = _normalize_batch(batch, normalizer)
        out = model(
            pos=pos,
            force=force,
            tactile=tactile,
            mask=batch["mask"],
        )
        loss = out["loss"]
        out["loss"] = loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        bs = force.shape[0]
        count += bs
        metrics = detach_scalar_dict(_reconstruction_metrics(out))
        for key, value in metrics.items():
            metric_sum[key] += float(value) * bs
        metric_sum["perplexity"] += (
            float(_perplexity(out["indices"], model.codebook_size)) * bs
        )
        pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")
    return {k: v / max(count, 1) for k, v in metric_sum.items()}


@torch.no_grad()
def validate_one_epoch(model, loader, normalizer, device):
    model.eval()
    metric_sum = defaultdict(float)
    count = 0
    for batch in tqdm(loader, desc="Pretrain-Val", leave=False):
        batch = move_to_device(batch, device)
        pos, force, tactile = _normalize_batch(batch, normalizer)
        out = model(
            pos=pos,
            force=force,
            tactile=tactile,
            mask=batch["mask"],
        )
        if model.contact_fusion and model.masked_modality_probability > 0.0:
            masked_out = model(
                pos=pos,
                force=force,
                tactile=tactile,
                mask=batch["mask"],
                mask_modalities=True,
            )
            out["masked_modality_recon_loss"] = masked_out["masked_modality_recon_loss"]
        bs = force.shape[0]
        count += bs
        metrics = detach_scalar_dict(_reconstruction_metrics(out))
        for key, value in metrics.items():
            metric_sum[key] += float(value) * bs
        metric_sum["perplexity"] += (
            float(_perplexity(out["indices"], model.codebook_size)) * bs
        )
    return {k: v / max(count, 1) for k, v in metric_sum.items()}


def main(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(
        cfg_get(cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    train_cfg = cfg["train_vq"] if "train_vq" in cfg else cfg["train"]
    data_kwargs = _dataset_kwargs(cfg)

    train_ds = build_force_episode_dataset(split="train", **data_kwargs)
    val_ds = build_force_episode_dataset(split="val", **data_kwargs)

    max_rows = cfg.get("train", {}).get("normalizer_max_rows")
    norm_cfg = _normalizer_cfg(cfg)
    normalizer, _ = _fit_force_vq_normalizer(
        train_ds, max_rows=max_rows, norm_cfg=norm_cfg
    )
    normalizer.to(device)

    num_workers = int(train_cfg.get("num_workers", 4))
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_force_episodes,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            train_cfg.get("persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(
            train_cfg.get("val_batch_size", train_cfg.get("batch_size", 64))
        ),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = build_vq_model(cfg, device)
    optimizer_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    output_cfg = cfg["output"]
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_cfg.get("root_dir", "outputs"), run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(
        os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    writer = SummaryWriter(log_dir=run_dir)

    epochs = int(train_cfg.get("epochs", 100))
    val_every = max(1, int(cfg.get("checkpoint", {}).get("val_every", 5)))
    save_every = max(1, int(cfg.get("checkpoint", {}).get("save_every", 20)))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    best_val = float("inf")

    print(
        f"contact encoder train episodes={len(train_ds)}, val={len(val_ds)}, device={device}"
    )
    for epoch in range(1, epochs + 1):
        train_avg = train_one_epoch(
            model, train_loader, normalizer, optimizer, device, grad_clip=grad_clip
        )
        recon_parts = [f"recon={train_avg['recon_loss']:.6f}"]
        for key, label in (
            ("recon_force_loss", "force"),
            ("recon_pos_loss", "pos"),
            ("recon_tactile_loss", "tactile"),
        ):
            if key in train_avg:
                recon_parts.append(f"{label}={train_avg[key]:.6f}")
        msg = (
            f"[contact encoder Epoch {epoch:03d}] train_loss={train_avg['loss']:.6f} "
            f"{' '.join(recon_parts)} ppl={train_avg['perplexity']:.2f}"
        )

        val_avg = None
        if epoch % val_every == 0 or epoch == epochs:
            val_avg = validate_one_epoch(model, val_loader, normalizer, device)
            msg += (
                f", val_loss={val_avg['loss']:.6f} val_ppl={val_avg['perplexity']:.2f}"
            )
        print(msg)

        for k, v in train_avg.items():
            writer.add_scalar(f"contact encoder/train_{k}", v, epoch)
        if val_avg is not None:
            for k, v in val_avg.items():
                writer.add_scalar(f"contact encoder/val_{k}", v, epoch)

        state = {
            "epoch": epoch,
            "force_vq_state_dict": model.state_dict(),
            "force_normalizer_state_dict": normalizer.state_dict(),
            "config": cfg,
        }
        torch.save(state, os.path.join(ckpt_dir, "latest.pt"))
        if epoch % save_every == 0:
            torch.save(state, os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))
        if val_avg is not None and val_avg["loss"] < best_val:
            best_val = val_avg["loss"]
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> best contact encoder checkpoint val_loss={best_val:.6f}")
        writer.flush()

    writer.close()
    print(f"contact encoder training finished: {run_dir}")
