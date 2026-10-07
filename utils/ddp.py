"""Policy-training DDP helpers. VQ training stays single-process."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist


def visible_gpu_count() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None or vis.strip() == "":
        return int(torch.cuda.device_count())
    return len([part for part in vis.split(",") if part.strip() != ""])


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return rank() == 0


def launched_by_torchrun() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def maybe_reexec_torchrun(*, ddp: bool, stage: str) -> None:
    """If --ddp was passed to a single process, re-launch with torchrun."""
    if not ddp or launched_by_torchrun():
        return
    if stage != "policy":
        sys.exit("--ddp is only implemented for --stage policy.")
    nproc = visible_gpu_count()
    if nproc < 2:
        sys.exit(
            "--ddp needs at least 2 visible GPUs. Example: ./scripts/train.sh --gpus 0,1 --ddp --config ..."
        )
    script = str(Path(sys.argv[0]).resolve())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "--max_restarts=0",
        script,
        *sys.argv[1:],
    ]
    os.execv(sys.executable, cmd)


def init_process_group() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}")


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def wrap_policy(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not is_distributed():
        return policy
    return torch.nn.parallel.DistributedDataParallel(
        policy,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )


def reduce_mean(metrics: Mapping[str, float], count: int, device: torch.device) -> dict[str, float]:
    """Weighted mean of locally-averaged metrics across ranks."""
    if not is_distributed() or count <= 0:
        return dict(metrics)
    keys = sorted(metrics)
    payload = torch.tensor(
        [float(count), *[float(metrics[k]) * float(count) for k in keys]],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total = max(float(payload[0].item()), 1.0)
    return {k: float(payload[i + 1].item()) / total for i, k in enumerate(keys)}
