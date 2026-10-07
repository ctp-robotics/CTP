from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.ddp import (
    barrier,
    cleanup,
    init_process_group,
    is_distributed,
    is_main,
    launched_by_torchrun,
    rank,
    reduce_mean,
    unwrap,
    world_size,
    wrap_policy,
)
from utils.train_utils import (
    build_canonical_config,
    cfg_get,
    detach_scalar_dict,
    log_hparams_to_tensorboard,
    set_seed,
)
from datasets.policy_zarr_dataset import (
    build_zarr_dataset,
    collect_force_vq_reference_tables,
    policy_dataset_kwargs,
)
from models.policy.contact_policy import ContactPolicy
from models.policy.contact_autoencoder import bundle_force_vq_checkpoint, load_force_vq
from utils.tensor import move_to_device

_PROMPT_BATCH_KEYS = (
    "force_vq_prompt",
    "force_vq_padding_mask",
    "force_vq_reference_pos",
    "force_vq_reference_force",
    "force_vq_reference_tactile",
    "force_vq_reference_phase",
    "force_vq_source_episode_index",
)


def _obs_with_prompts(batch: dict) -> dict:
    """Attach reference inputs for inference; future labels remain outside obs."""
    obs = dict(batch["obs"])
    obs.update({key: batch[key] for key in _PROMPT_BATCH_KEYS if key in batch})
    return obs


def get_autocast_context(device: torch.device, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda")
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def build_dataloaders(cfg: dict):
    train_cfg = cfg["train"]
    dataset_kwargs = policy_dataset_kwargs(cfg)
    num_workers = int(train_cfg.get("num_workers", 8))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))

    train_ds = build_zarr_dataset(split="train", **dataset_kwargs)
    val_ds = build_zarr_dataset(split="val", **dataset_kwargs)

    train_sampler = None
    val_sampler = None
    if is_distributed():
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=True,
            drop_last=True,
            seed=int(dataset_kwargs.get("split_seed", cfg_get(cfg, "seed", 42))),
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=False,
            drop_last=False,
        )

    train_loader_kwargs = {
        "batch_size": train_cfg.get("batch_size", 32),
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": num_workers,
        "drop_last": True,
        "pin_memory": pin_memory,
    }
    val_loader_kwargs = {
        "batch_size": train_cfg.get("val_batch_size", train_cfg.get("batch_size", 32)),
        "shuffle": False,
        "sampler": val_sampler,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)
    return train_ds, val_ds, train_loader, val_loader


def _cache_is_current(
    cache_path: str,
    ckpt_path: str,
    dataset,
) -> bool:
    """Return whether a cache matches its checkpoint, source data, and split."""
    if not os.path.isdir(cache_path):
        return False
    st = os.stat(ckpt_path)
    try:
        cache = zarr.open_group(cache_path, mode="r")
        attrs = cache.attrs
        if (
            attrs.get("source_checkpoint") != os.path.abspath(ckpt_path)
            or attrs.get("source_checkpoint_mtime") != int(st.st_mtime)
            or attrs.get("source_zarr_path") != dataset.zarr_path
        ):
            return False

        data = cache["data"]
        meta = cache["meta"]
        n_episodes = dataset.num_episodes_total
        if (
            data["sample_indices"].shape[0] != n_episodes
            or meta["valid_episode_mask"].shape[0] != n_episodes
        ):
            return False
        if not np.array_equal(
            np.asarray(meta["episode_ends"][:]), dataset.episode_ends
        ):
            return False
        if not np.array_equal(
            np.asarray(meta["selected_episode_indices"][:]), dataset.episode_indices
        ):
            return False
        valid = np.asarray(meta["valid_episode_mask"][:], dtype=bool)
        return bool(valid[dataset.episode_indices].all())
    except Exception:
        return False


def _ensure_force_vq_caches(cfg: dict, device: torch.device) -> None:
    """Rebuild per-root contact encoder prompt caches when stale or missing."""
    mode_prompt = cfg["model"]["policy"].get("mode_prompt") or {}
    ckpt = mode_prompt["force_vq"].get("ckpt")
    if not ckpt:
        raise ValueError(
            "Set model.policy.reference_encoder.checkpoint before policy training."
        )
    ckpt_path = os.path.abspath(str(ckpt))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"contact encoder checkpoint not found: {ckpt}")

    from datasets.force_episode_dataset import ForceEpisodeDataset
    from tools.precompute_reference_cache import (
        _remove_path,
        load_force_vq_checkpoint,
        precompute_prompts,
        resolve_output_path,
    )
    from trainers.pretrain_trainer import _dataset_kwargs

    model, normalizer, vq_cfg = load_force_vq_checkpoint(ckpt_path, device)
    data_kwargs = _dataset_kwargs(vq_cfg)
    # Cache selection must match the policy dataset, including logical splits.
    data_kwargs["val_ratio"] = cfg["data"].get("val_ratio", 0.1)
    split_seed = cfg["data"].get("split_seed")
    data_kwargs["split_seed"] = int(
        cfg.get("seed", 42) if split_seed is None else split_seed
    )
    for key in ("force_key", "state_key", "tactile_left_key", "tactile_right_key"):
        if cfg.get("data", {}).get(key) is not None:
            data_kwargs[key] = cfg["data"][key]
    roots = cfg["data"]["root_dir"]
    roots = [roots] if isinstance(roots, (str, os.PathLike)) else list(roots)
    # Lists assign an independent cache directory to each dataset root.
    cache_roots = cfg.get("data", {}).get("force_vq_cache_root_dir")
    if isinstance(cache_roots, (list, tuple)):
        if len(cache_roots) != len(roots):
            raise ValueError(
                "data.force_vq_cache_root_dir list length must match data.root_dir "
                f"({len(cache_roots)} != {len(roots)})."
            )
        cache_roots = [str(path) for path in cache_roots]
    elif (
        len(roots) == 1
        and cache_roots
        and str(cache_roots).lower() not in {"auto", "true", "1"}
    ):
        cache_roots = [str(cache_roots)]
    else:
        # Multiple data roots need distinct stores to avoid episode-ID collisions.
        cache_roots = [str(root) for root in roots]
    if any(path.endswith(".zarr") for path in cache_roots):
        raise ValueError(
            "Training requires a reference cache directory containing separate train/val stores; "
            "set data.reference_cache_root_dir to the parent directory, not a .zarr store."
        )
    for split in ("train", "val"):
        for root, cache_root in zip(roots, cache_roots):
            per_root = dict(data_kwargs)
            per_root["root_dir"] = root
            dataset = ForceEpisodeDataset(split=split, **per_root)
            output_path = resolve_output_path(str(cache_root), split, None)
            if _cache_is_current(output_path, ckpt_path, dataset):
                print(f"[force-vq-cache] up to date: {output_path}")
                continue
            if os.path.lexists(output_path):
                _remove_path(output_path)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            print(f"[force-vq-cache] computing {split}: {output_path}")
            precompute_prompts(
                model,
                normalizer,
                dataset,
                output_path,
                batch_size=32,
                device=device,
                source_checkpoint=ckpt_path,
            )


def build_policy(cfg: dict, device: torch.device, train_dataset=None) -> ContactPolicy:
    policy_cfg = cfg["model"]["policy"]
    if getattr(
        train_dataset, "cached_image_backbone_feat", None
    ) is not None and not policy_cfg.get("freeze_image_encoder", True):
        raise ValueError("Cached image features require a frozen image encoder.")
    policy = ContactPolicy.from_cfg(cfg).to(device)
    if train_dataset is not None:
        policy.set_normalizer(
            train_dataset.get_normalizer(
                max_rows=cfg["train"].get("normalizer_max_rows")
            )
        )
        policy.normalizer.to(device)
    source = policy_cfg["mode_prompt"]["force_vq"]["ckpt"]
    encoder, normalizer, _ = load_force_vq(source, device)
    policy.set_trainable_force_vq(encoder, normalizer)
    if train_dataset is not None:
        set_policy_reference_dataset(policy, train_dataset)
    return policy


def _strip_state_dict_prefix(state_dict: dict, prefix: str) -> dict:
    if not state_dict or not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_pretrained_checkpoint(
    policy: ContactPolicy,
    checkpoint_path: str,
    device: torch.device,
    load_normalizer: bool = True,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("policy_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint missing policy_state_dict: {checkpoint_path}")

    from utils.checkpoint_util import policy_weights

    state_dict = policy_weights(state_dict)
    incompat = policy.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint is incompatible: "
            f"missing={incompat.missing_keys[:8]}, unexpected={incompat.unexpected_keys[:8]}"
        )

    if load_normalizer:
        normalizer_state = ckpt.get("normalizer_state_dict")
        if not isinstance(normalizer_state, dict) or not normalizer_state:
            raise KeyError(
                f"Checkpoint missing normalizer_state_dict: {checkpoint_path}"
            )
        normalizer_state = _strip_state_dict_prefix(normalizer_state, "module.")
        policy.normalizer.load_state_dict(normalizer_state)
        policy.normalizer.to(device)


def set_policy_reference_dataset(policy, dataset) -> None:
    """Bind episode IDs to the reference table of the active data split.

    Episode IDs are local to a dataset. Reusing a training table for validation
    can select unrelated training episodes even when the numeric IDs are valid.
    """
    raw = unwrap(policy)
    if getattr(raw, "_reference_table_dataset", None) is dataset:
        return
    include_phase = raw.trainable_force_vq.temporal_position_mode == "normalized_phase"
    table = collect_force_vq_reference_tables(dataset, include_phase=include_phase)
    pos, force, tactile = table[:3]
    phase = table[3] if include_phase else None
    raw.set_trainable_force_vq_reference_table(pos, force, tactile, phase)
    raw._reference_table_dataset = dataset


def train_one_epoch(
    policy,
    loader,
    optimizer,
    device,
    grad_clip=None,
    global_step: int = 0,
    writer=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    max_steps: int | None = None,
):
    policy.train()
    set_policy_reference_dataset(policy, loader.dataset)
    metric_sum = defaultdict(float)
    count = 0

    pbar = tqdm(loader, desc="Train", leave=False, disable=not is_main())
    steps_this_epoch = 0
    for batch in pbar:
        if max_steps is not None and steps_this_epoch >= max_steps:
            break
        global_step += 1
        steps_this_epoch += 1
        batch = move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, use_amp):
            out = policy(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict(out.get("metrics", {}))

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

        bs = batch["action"].shape[0]
        metric_sum["loss"] += float(loss.detach().item()) * bs
        for k, v in scalar_metrics.items():
            metric_sum[k] += v * bs
        count += bs

        step_lr = optimizer.param_groups[0]["lr"]
        if writer is not None:
            writer.add_scalar("Step/lr", step_lr, global_step)
            for k, v in scalar_metrics.items():
                writer.add_scalar(f"Step/{k}", v, global_step)

        postfix = {"loss": f"{loss.detach().item():.4f}", "lr": f"{step_lr:.2e}"}
        for k, v in scalar_metrics.items():
            postfix[k] = f"{v:.4f}"
        pbar.set_postfix(postfix)

    avg = {k: v / max(count, 1) for k, v in metric_sum.items()}
    avg = reduce_mean(avg, count, device)
    return avg, global_step


@torch.no_grad()
def validate_one_epoch(
    policy, loader, device, epoch: int, writer=None, use_amp: bool = False
):
    policy.eval()
    set_policy_reference_dataset(policy, loader.dataset)
    metric_sum = defaultdict(float)
    count = 0

    pbar = tqdm(loader, desc="Val", leave=False, disable=not is_main())
    for batch in pbar:
        batch = move_to_device(batch, device)
        with get_autocast_context(device, use_amp):
            out = policy(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict(out.get("metrics", {}))

        bs = batch["action"].shape[0]
        metric_sum["loss"] += float(loss.detach().item()) * bs
        for k, v in scalar_metrics.items():
            metric_sum[k] += v * bs
        count += bs
        pbar.set_postfix(loss=f"{loss.detach().item():.4f}")

    avg = {k: v / max(count, 1) for k, v in metric_sum.items()}
    avg = reduce_mean(avg, count, device)
    if writer is not None:
        for k, v in avg.items():
            if k != "loss":
                writer.add_scalar(f"Epoch/val_{k}", v, epoch)
    return avg


@torch.no_grad()
def evaluate_open_loop(
    policy,
    loader,
    device,
    epoch: int,
    split: str,
    max_batches: int,
    writer=None,
):
    policy.eval()
    set_policy_reference_dataset(policy, loader.dataset)
    mse_sum = 0.0
    l1_sum = 0.0
    force_mse_sum = 0.0
    force_l1_sum = 0.0
    force_count = 0
    count = 0

    pbar = tqdm(loader, desc=f"OpenLoop-{split}", leave=False, disable=not is_main())
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= max_batches:
            break

        batch = move_to_device(batch, device)
        result = policy.predict_action(_obs_with_prompts(batch))
        pred_action = result.get("action_model", result["action"])
        gt_action = batch["action"]

        t = min(pred_action.shape[1], gt_action.shape[1])
        d = min(pred_action.shape[2], gt_action.shape[2])
        if t <= 0 or d <= 0:
            continue

        pred = pred_action[:, :t, :d]
        gt = gt_action[:, :t, :d]

        mse = F.mse_loss(pred, gt, reduction="mean")
        l1 = torch.mean(torch.abs(pred - gt))

        bs = gt.shape[0]
        mse_sum += float(mse.item()) * bs
        l1_sum += float(l1.item()) * bs
        count += bs

        if "force_pred_model" in result and "future_force" in batch:
            force_pred = result["force_pred_model"]
            force_indices = result["force_pred_indices"].to(
                batch["future_force"].device
            )
            force_gt = batch["future_force"].index_select(-1, force_indices)
            force_t = min(force_pred.shape[1], force_gt.shape[1])
            if force_t > 0:
                force_pred = force_pred[:, :force_t]
                force_gt = force_gt[:, :force_t]
                force_mse_sum += float(F.mse_loss(force_pred, force_gt).item()) * bs
                force_l1_sum += float(F.l1_loss(force_pred, force_gt).item()) * bs
                force_count += bs

        pbar.set_postfix(mse=f"{mse.item():.4f}", l1=f"{l1.item():.4f}")

    metrics = {
        "action_mse": mse_sum / max(count, 1),
        "action_l1": l1_sum / max(count, 1),
    }
    metrics = reduce_mean(metrics, count, device)
    if force_count > 0:
        force_metrics = reduce_mean(
            {
                "force_mse": force_mse_sum / force_count,
                "force_l1": force_l1_sum / force_count,
            },
            force_count,
            device,
        )
        metrics.update(force_metrics)

    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(f"OpenLoop/{split}_{k}", v, epoch)

    return metrics


def prepare_force_vq_bundle(cfg: dict) -> dict:
    """Embed encoder architecture and normalization for standalone policy loading."""
    return bundle_force_vq_checkpoint(
        cfg["model"]["policy"]["mode_prompt"]["force_vq"]["ckpt"]
    )


def get_checkpoint_state(
    policy, _optimizer, epoch, cfg, force_vq_bundle: dict | None = None
):
    raw = unwrap(policy)
    state = {
        "epoch": int(epoch),
        "policy_state_dict": raw.state_dict(),
        "normalizer_state_dict": deepcopy(raw.normalizer.state_dict()),
        "config": cfg,
    }
    if force_vq_bundle is not None:
        state["force_vq_bundle"] = deepcopy(force_vq_bundle)
    if getattr(raw, "trainable_force_vq", None) is not None:
        state["trainable_force_vq_normalizer_state_dict"] = deepcopy(
            raw.trainable_force_vq_normalizer.state_dict()
        )
    return state


def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_dataloader(
    loader, device: torch.device, warmup_steps: int = 10, bench_steps: int = 100
):
    total_steps = warmup_steps + bench_steps
    it = iter(loader)
    timings = []
    sample_count = 0

    for step in range(total_steps):
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _sync_if_cuda(device)
        dt = time.perf_counter() - t0

        if step >= warmup_steps:
            timings.append(dt)
            if (
                isinstance(batch, dict)
                and "action" in batch
                and torch.is_tensor(batch["action"])
            ):
                sample_count += int(batch["action"].shape[0])

    mean_step = float(np.mean(timings)) if timings else 0.0
    std_step = float(np.std(timings)) if timings else 0.0
    samples_per_sec = sample_count / max(sum(timings), 1e-12)

    print("\n=== Dataloader Benchmark ===")
    print(f"warmup={warmup_steps}, steps={bench_steps}")
    print(
        f"step_time_mean={mean_step * 1000:.2f} ms, step_time_std={std_step * 1000:.2f} ms"
    )
    print(f"samples_per_sec={samples_per_sec:.2f}")
    return {
        "step_time_mean_ms": mean_step * 1000.0,
        "step_time_std_ms": std_step * 1000.0,
        "samples_per_sec": samples_per_sec,
    }


def benchmark_model_train_step(
    policy,
    loader,
    optimizer,
    device: torch.device,
    warmup_steps: int = 10,
    bench_steps: int = 100,
):
    total_steps = warmup_steps + bench_steps
    it = iter(loader)
    timings = []
    sample_count = 0

    policy.train()
    for step in range(total_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = move_to_device(batch, device)
        _sync_if_cuda(device)
        t0 = time.perf_counter()

        out = policy(batch)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        _sync_if_cuda(device)
        dt = time.perf_counter() - t0
        if step >= warmup_steps:
            timings.append(dt)
            if "action" in batch and torch.is_tensor(batch["action"]):
                sample_count += int(batch["action"].shape[0])

    mean_step = float(np.mean(timings)) if timings else 0.0
    std_step = float(np.std(timings)) if timings else 0.0
    samples_per_sec = sample_count / max(sum(timings), 1e-12)

    print("\n=== Model Train-Step Benchmark ===")
    print(f"warmup={warmup_steps}, steps={bench_steps}")
    print(
        f"step_time_mean={mean_step * 1000:.2f} ms, step_time_std={std_step * 1000:.2f} ms"
    )
    print(f"samples_per_sec={samples_per_sec:.2f}")
    return {
        "step_time_mean_ms": mean_step * 1000.0,
        "step_time_std_ms": std_step * 1000.0,
        "samples_per_sec": samples_per_sec,
    }


def build_optimizer(policy: ContactPolicy, train_cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = train_cfg["optimizer"]
    base_lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg["weight_decay"])
    encoder_lr = opt_cfg.get("encoder_lr", None)
    reference_encoder_lr = opt_cfg.get("reference_encoder_lr", None)

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    reference_encoder = getattr(policy, "trainable_force_vq", None)
    reference_params = (
        []
        if reference_encoder is None
        else [p for p in reference_encoder.parameters() if p.requires_grad]
    )
    if reference_encoder_lr is not None and not reference_params:
        raise ValueError(
            "train.optimizer.reference_encoder_lr is set, but no trainable continuous contact encoder parameters exist."
        )
    if encoder_lr is None and reference_encoder_lr is None:
        return torch.optim.AdamW(
            trainable_params, lr=base_lr, weight_decay=weight_decay
        )

    backbone = getattr(
        getattr(policy.obs_encoder, "image_encoder", None), "backbone", None
    )
    if encoder_lr is not None and backbone is None:
        raise ValueError(
            "train.optimizer.encoder_lr is set, but policy has no DINO backbone."
        )
    encoder_params = (
        []
        if backbone is None
        else [p for p in backbone.parameters() if p.requires_grad]
    )
    if encoder_lr is not None and not encoder_params:
        if is_main():
            print(
                "[optimizer] encoder_lr is set but DINO backbone has no trainable parameters; using base lr only."
            )

    special_ids = set()
    if encoder_params:
        special_ids.update(id(p) for p in encoder_params)
    if reference_encoder_lr is not None and reference_params:
        special_ids.update(id(p) for p in reference_params)
    policy_params = [p for p in trainable_params if id(p) not in special_ids]
    param_groups = [{"params": policy_params, "lr": base_lr, "name": "policy"}]
    summary = [
        f"policy_lr={base_lr:g} ({sum(p.numel() for p in policy_params)} params)"
    ]
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": float(encoder_lr), "name": "dino_backbone"}
        )
        summary.append(
            f"dino_backbone_lr={float(encoder_lr):g} ({sum(p.numel() for p in encoder_params)} params)"
        )
    if reference_encoder_lr is not None and reference_params:
        param_groups.append(
            {
                "params": reference_params,
                "lr": float(reference_encoder_lr),
                "name": "reference_fpt_encoder",
            }
        )
        summary.append(
            f"reference_fpt_encoder_lr={float(reference_encoder_lr):g} "
            f"({sum(p.numel() for p in reference_params)} params)"
        )
    if is_main():
        print("[optimizer] AdamW parameter groups: " + ", ".join(summary))
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def main(cfg: dict):
    ddp = bool(cfg_get(cfg, "runtime.ddp", False) or launched_by_torchrun())
    if ddp:
        device = init_process_group()
    else:
        device = torch.device(
            cfg_get(
                cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
    set_seed(int(cfg.get("seed", 42)) + rank())
    runtime_cfg = cfg["runtime"]
    benchmark = runtime_cfg.get("benchmark", "none")
    bench_steps = int(runtime_cfg.get("bench_steps", 100))
    bench_warmup = int(runtime_cfg.get("bench_warmup", 10))
    if ddp and benchmark != "none":
        cleanup()
        raise ValueError("--benchmark is not supported with --ddp.")

    if is_main():
        _ensure_force_vq_caches(cfg, device)
    barrier()
    force_vq_bundle = prepare_force_vq_bundle(cfg)
    if force_vq_bundle is not None:
        # Use the checkpoint architecture for both reference data preparation
        # and model construction; policy.yaml only needs its checkpoint path.
        encoder_cfg = build_canonical_config(force_vq_bundle["config"])["model"][
            "force_vq"
        ]
        cfg["model"]["force_vq"] = deepcopy(encoder_cfg)
        reference_cfg = cfg["model"]["policy"]["mode_prompt"]["force_vq"]
        reference_cfg["num_tokens"] = int(encoder_cfg.get("num_tokens", 16))
        reference_cfg["latent_dim"] = int(encoder_cfg.get("latent_dim", 128))
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg)
    if is_main():
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        if is_distributed():
            per_gpu = int(cfg["train"].get("batch_size", 32))
            print(
                f"[ddp] world_size={world_size()} device={device} "
                f"per_gpu_batch={per_gpu} global_batch={per_gpu * world_size()}"
            )

    policy = build_policy(cfg, device, train_ds)

    ckpt_cfg = cfg["checkpoint"]
    pretrained_ckpt = ckpt_cfg.get("pretrained_ckpt")
    if pretrained_ckpt:
        load_pretrained_checkpoint(
            unwrap(policy),
            checkpoint_path=str(pretrained_ckpt),
            device=device,
            load_normalizer=bool(ckpt_cfg.get("load_normalizer", True)),
        )
        if is_main():
            print(f"Loaded pretrained policy checkpoint: {pretrained_ckpt}")

    train_cfg = cfg["train"]
    optimizer = build_optimizer(unwrap(policy), train_cfg)
    policy = wrap_policy(policy, device)

    if benchmark in {"dataloader", "both"}:
        benchmark_dataloader(
            train_loader,
            device=device,
            warmup_steps=bench_warmup,
            bench_steps=bench_steps,
        )
    if benchmark in {"model", "both"}:
        benchmark_model_train_step(
            policy,
            train_loader,
            optimizer,
            device=device,
            warmup_steps=bench_warmup,
            bench_steps=bench_steps,
        )
    if benchmark != "none":
        print("\nBenchmark done. Skip full training loop.")
        return

    output_cfg = cfg["output"]
    output_root = output_cfg.get("root_dir", "outputs")
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    writer = None
    if is_main():
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(
            os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        writer = SummaryWriter(log_dir=run_dir)
        log_hparams_to_tensorboard(writer, cfg, run_dir)
        print(f"TensorBoard log dir: {run_dir}")
    barrier()

    epochs = int(train_cfg.get("epochs", 100))
    max_train_steps_cfg = train_cfg.get("max_steps", None)
    max_train_steps = (
        None
        if max_train_steps_cfg is None or int(max_train_steps_cfg) <= 0
        else int(max_train_steps_cfg)
    )
    val_every = max(1, int(ckpt_cfg.get("val_every", 1)))
    save_every = max(1, int(ckpt_cfg.get("save_every", 5)))
    grad_clip = train_cfg.get("grad_clip", None)
    open_loop_every = int(train_cfg.get("open_loop_test_every", 0))
    open_loop_max_batches = max(1, int(train_cfg.get("open_loop_test_max_batches", 20)))
    use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric_name = str(ckpt_cfg.get("best_metric", "loss"))
    best_val = float("inf")
    global_step = 0
    try:
        for epoch in range(1, epochs + 1):
            if max_train_steps is not None and global_step >= max_train_steps:
                break
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            train_avg, global_step = train_one_epoch(
                policy,
                train_loader,
                optimizer,
                device,
                grad_clip=grad_clip,
                global_step=global_step,
                writer=writer,
                scaler=(scaler if use_amp else None),
                use_amp=use_amp,
                max_steps=(
                    None if max_train_steps is None else max_train_steps - global_step
                ),
            )
            train_loss = train_avg["loss"]

            if epoch % val_every == 0 or epoch == epochs:
                val_avg = validate_one_epoch(
                    policy,
                    val_loader,
                    device,
                    epoch=epoch,
                    writer=writer,
                    use_amp=use_amp,
                )
                val_loss = val_avg["loss"]
            else:
                val_avg = None
                val_loss = None

            train_open = None
            val_open = None
            if open_loop_every > 0 and (
                epoch % open_loop_every == 0 or epoch == epochs
            ):
                train_open = evaluate_open_loop(
                    unwrap(policy),
                    train_loader,
                    device,
                    epoch,
                    split="train",
                    max_batches=open_loop_max_batches,
                    writer=writer,
                )
                val_open = evaluate_open_loop(
                    unwrap(policy),
                    val_loader,
                    device,
                    epoch,
                    split="val",
                    max_batches=open_loop_max_batches,
                    writer=writer,
                )

            curr_lr = optimizer.param_groups[0]["lr"]
            if writer is not None:
                writer.add_scalar("Epoch/lr", curr_lr, epoch)
                for k, v in train_avg.items():
                    if k != "loss":
                        writer.add_scalar(f"Epoch/train_{k}", v, epoch)

            msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.6f}"
            if train_open is not None:
                msg += f", train_open_mse={train_open['action_mse']:.6f}, train_open_l1={train_open['action_l1']:.6f}"
            if val_open is not None:
                msg += f", val_open_mse={val_open['action_mse']:.6f}, val_open_l1={val_open['action_l1']:.6f}"
            if is_main():
                print(msg)

            state = get_checkpoint_state(policy, optimizer, epoch, cfg, force_vq_bundle)
            if is_main():
                save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), state)
                if epoch % save_every == 0:
                    save_checkpoint(
                        os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), state
                    )
            current_best_metric = (
                None if val_avg is None else val_avg.get(best_metric_name)
            )
            if val_avg is not None and current_best_metric is None:
                raise KeyError(
                    f"checkpoint.best_metric={best_metric_name!r} is not a validation metric. "
                    f"Available metrics: {sorted(val_avg)}"
                )
            if current_best_metric is not None and current_best_metric < best_val:
                best_val = current_best_metric
                if is_main():
                    save_checkpoint(os.path.join(ckpt_dir, "best.pt"), state)
                    print(
                        f"  -> Updated best checkpoint: "
                        f"val_{best_metric_name}={best_val:.6f}"
                    )

            if writer is not None:
                writer.flush()
            barrier()
            if max_train_steps is not None and global_step >= max_train_steps:
                if is_main():
                    print(
                        f"Reached train.max_steps={max_train_steps}; stopping adaptation/training."
                    )
                break
    finally:
        if writer is not None:
            writer.close()
        if is_main():
            print(f"Training finished. Artifacts saved in: {run_dir}")
        cleanup()
