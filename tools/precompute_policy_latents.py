from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.policy_zarr_dataset import (
    ZarrDataset,
    policy_dataset_kwargs,
)  # noqa: E402
from models.policy.obs_encoder import DinoV2SmallEncoder  # noqa: E402
from utils.train_utils import build_canonical_config  # noqa: E402


def _dataset_kwargs(cfg: dict, root_dir: str) -> dict:
    data = policy_dataset_kwargs(cfg)
    data.update(
        root_dir=root_dir,
        latent_cache_root_dir=None,
        force_vq_cache_root_dir=None,
        preload_to_ram=True,
    )
    return data


def iter_roots(cfg: dict) -> list[str]:
    root_dir = cfg["data"]["root_dir"]
    if isinstance(root_dir, (str, os.PathLike)):
        return [str(root_dir)]
    return [str(r) for r in root_dir]


def resolve_output_path(root_dir: str, split: str, output_path: str | None) -> str:
    if output_path:
        return output_path.format(split=split, root=root_dir)
    return os.path.join(root_dir, split, "policy_latent_cache.zarr")


def build_image_batch(dataset: ZarrDataset, indices: list[int]) -> torch.Tensor:
    image_batch = []
    for idx in indices:
        anchor_t, _, ep_idx = dataset.windows[idx]
        image_start = int(anchor_t - dataset.n_image_steps + 1)
        image_end = int(anchor_t + 1)
        ep_start = int(dataset.episode_starts[ep_idx])
        if image_start < ep_start:
            raise ValueError(
                f"Image history starts before episode: image_start={image_start}, ep_start={ep_start}"
            )

        views = []
        for key in dataset.image_keys:
            image = dataset._read_array(key, slice(image_start, image_end))
            views.append(dataset._process_image(image).numpy())
        image_batch.append(np.stack(views, axis=1))
    return torch.from_numpy(np.stack(image_batch, axis=0))


def _create_array(group, name, **kwargs):
    create = getattr(group, "create_array", group.create_dataset)
    return create(name, **kwargs)


def write_window_metadata(meta_group, dataset: ZarrDataset) -> None:
    windows = np.asarray(dataset.windows, dtype=np.int64)
    _create_array(meta_group, "window_anchor_times", data=windows[:, 0])
    _create_array(meta_group, "window_episode_ends", data=windows[:, 1])
    _create_array(meta_group, "window_episode_indices", data=windows[:, 2])


def precompute_image_features(
    cfg: dict, dataset: ZarrDataset, output_path: str, args: argparse.Namespace
) -> None:
    policy_cfg = cfg["model"]["policy"]
    if not bool(policy_cfg.get("freeze_image_encoder", True)):
        raise ValueError(
            "Image feature caching requires policy.freeze_image_encoder=True."
        )

    device = torch.device(args.device)
    image_encoder = DinoV2SmallEncoder(
        out_dim=policy_cfg.get("cond_dim", 256),
        pretrained=policy_cfg.get("image_pretrained", True),
        freeze=True,
        model_name=policy_cfg.get(
            "dino_model_name", "vit_small_patch14_dinov2.lvd142m"
        ),
    ).to(device)
    image_encoder.eval()

    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = 2
    out_root.attrs["split"] = args.split
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["window_size"] = int(dataset.window_size)
    out_root.attrs["action_window_size"] = int(dataset.action_window_size)
    out_root.attrs["stride"] = int(dataset.stride)
    out_root.attrs["n_image_steps"] = int(dataset.n_image_steps)
    out_root.attrs["force_steps"] = int(getattr(dataset, "force_steps", 0))
    out_root.attrs["tactile_steps"] = int(getattr(dataset, "tactile_steps", 0))

    data_group = out_root.create_group("data")
    meta_group = out_root.create_group("meta")
    write_window_metadata(meta_group, dataset)

    img_arr = None
    chunk_bsz = max(1, min(args.batch_size, 1024))
    for start_idx in tqdm(
        range(0, len(dataset), args.batch_size),
        desc=f"image-cache:{args.split}",
        unit="batch",
    ):
        batch_indices = list(
            range(start_idx, min(start_idx + args.batch_size, len(dataset)))
        )
        image_batch = build_image_batch(dataset, batch_indices).to(
            device, non_blocking=True
        )

        with torch.inference_mode():
            bsz, steps, views = image_batch.shape[:3]
            image_feat = image_encoder.extract_backbone_feat(
                image_batch.reshape(bsz * steps * views, *image_batch.shape[3:])
            ).reshape(bsz, steps, views, -1)

        img = image_feat.detach().cpu().numpy().astype(np.float32, copy=False)
        if img_arr is None:
            img_arr = _create_array(
                data_group,
                "image_backbone_feat",
                shape=(len(dataset),) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
        img_arr[start_idx : start_idx + len(batch_indices)] = img

    print(f"Saved image feature cache to: {output_path} (windows={len(dataset)})")


def _remove_path(path: str) -> None:
    """Rename old cache aside, then delete in a background process (NFS-friendly)."""
    if not os.path.lexists(path):
        return
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
        return
    if not os.path.isdir(path):
        os.unlink(path)
        return

    trash = f"{path}.deleting_{os.getpid()}"
    try:
        os.rename(path, trash)
    except OSError:
        print(
            f"[precompute] rename failed, deleting in-place (may be slow on NFS): {path}"
        )
        shutil.rmtree(path)
        return

    subprocess.Popen(
        ["rm", "-rf", trash],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[precompute] renamed old cache -> {trash}; deleting in background")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute frozen image backbone feature caches."
    )
    parser.add_argument(
        "--config", type=str, default=str(ROOT / "config" / "policy.yaml")
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zarr path. Supports {split} and {root}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = build_canonical_config(args.config)
    roots = iter_roots(cfg)
    if not roots:
        raise ValueError("data.root_dir is empty")

    for root in roots:
        dataset = ZarrDataset(split=args.split, **_dataset_kwargs(cfg, root))
        output_path = resolve_output_path(root, args.split, args.output)
        print(
            f"[precompute] root={root} split={args.split} windows={len(dataset)} -> {output_path}"
        )

        if os.path.lexists(output_path):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output zarr already exists: {output_path}. Use --overwrite to rebuild."
                )
            _remove_path(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        precompute_image_features(cfg, dataset, output_path, args)


if __name__ == "__main__":
    main()
