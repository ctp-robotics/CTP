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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.force_episode_dataset import (
    ForceEpisodeDataset,
    collate_force_episodes,
)  # noqa: E402
from models.policy.contact_autoencoder import (
    ContactAutoencoder,
    load_force_vq,
)  # noqa: E402
from trainers.pretrain_trainer import _dataset_kwargs  # noqa: E402
from utils.normalizer import MultiFieldNormalizer  # noqa: E402
from utils.tensor import move_to_device  # noqa: E402


def iter_roots(cfg: dict) -> list[str]:
    root_dir = cfg["data"]["root_dir"]
    if isinstance(root_dir, (str, os.PathLike)):
        return [str(root_dir)]
    return [str(r) for r in root_dir]


def resolve_output_path(root_dir: str, split: str, output_path: str | None) -> str:
    if output_path:
        return output_path.format(split=split, root=root_dir)
    return os.path.join(root_dir, split, "force_vq_prompt_cache.zarr")


def _remove_path(path: str) -> None:
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
        shutil.rmtree(path)
        return
    subprocess.Popen(
        ["rm", "-rf", trash],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[force-vq-cache] renamed old cache -> {trash}; deleting in background")


def _create_array(group, name: str, **kwargs):
    if hasattr(group, "create_array"):
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def load_force_vq_bundle(
    bundle: dict, device: torch.device
) -> tuple[ContactAutoencoder, MultiFieldNormalizer, dict]:
    return load_force_vq(bundle, device)


def load_force_vq_checkpoint(
    checkpoint: str, device: torch.device
) -> tuple[ContactAutoencoder, MultiFieldNormalizer, dict]:
    return load_force_vq(checkpoint, device)


def _normalize_batch(batch: dict, normalizer: MultiFieldNormalizer):
    pos = batch["pos"]
    force = batch["force"]
    tactile = batch.get("tactile")
    if "mode_pos" in normalizer:
        pos = normalizer["mode_pos"].normalize(pos)
    if "force" in normalizer:
        force = normalizer["force"].normalize(force)
    if tactile is not None and "tactile" in normalizer:
        tactile = normalizer["tactile"].normalize(tactile)
    return pos, force, tactile


@torch.no_grad()
def precompute_prompts(
    model: ContactAutoencoder,
    normalizer: MultiFieldNormalizer,
    dataset: ForceEpisodeDataset,
    output_path: str,
    *,
    batch_size: int,
    device: torch.device,
    source_checkpoint: str,
) -> None:
    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = 3
    out_root.attrs["split"] = dataset.split
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    sample_index_count = model.sample_points
    out_root.attrs["sample_points"] = sample_index_count
    out_root.attrs["num_tokens"] = int(model.num_tokens)
    out_root.attrs["latent_dim"] = int(model.latent_dim)
    out_root.attrs["codebook_size"] = int(model.codebook_size)
    out_root.attrs["encoder_type"] = str(model.encoder_type)
    out_root.attrs["pooling_type"] = str(model.pooling_type)
    out_root.attrs["sampling_type"] = str(model.sampling_type)
    out_root.attrs["tokenization"] = str(model.tokenization)
    out_root.attrs["uniform_ratio"] = float(model.uniform_ratio)
    out_root.attrs["fps_feature_source"] = str(model.fps_feature_source)
    out_root.attrs["fps_smoothing_kernel"] = int(model.fps_smoothing_kernel)
    out_root.attrs["reference_modalities"] = [
        name
        for name, dim in (
            ("position", model.pos_dim),
            ("force", model.force_dim),
            ("tactile", model.tactile_dim),
        )
        if dim > 0
    ]
    st = os.stat(source_checkpoint)
    out_root.attrs["source_checkpoint"] = os.path.abspath(source_checkpoint)
    out_root.attrs["source_checkpoint_mtime"] = int(st.st_mtime)

    data_group = out_root.create_group("data")
    meta_group = out_root.create_group("meta")
    n_total = dataset.num_episodes_total
    sample_arr = _create_array(
        data_group,
        "sample_indices",
        shape=(n_total, sample_index_count),
        chunks=(1, sample_index_count),
        dtype="i8",
        fill_value=-1,
    )
    valid = np.zeros(n_total, dtype=bool)

    _create_array(meta_group, "episode_starts", data=dataset.episode_starts)
    _create_array(meta_group, "episode_ends", data=dataset.episode_ends)
    _create_array(meta_group, "selected_episode_indices", data=dataset.episode_indices)

    for start in tqdm(
        range(0, len(dataset), batch_size), desc=f"force-vq-cache:{dataset.split}"
    ):
        items = [
            dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = collate_force_episodes(items)
        batch = move_to_device(batch, device)
        pos, force, tactile = _normalize_batch(batch, normalizer)
        sequence = model.pack_sequence(pos=pos, force=force, tactile=tactile)
        _, indices = model.sample_sequence(sequence, mask=batch["mask"])
        episode_indices = batch["episode_index"].cpu().numpy().astype(np.int64)
        sample_arr[episode_indices] = indices.cpu().numpy().astype(np.int64, copy=False)
        valid[episode_indices] = True

    _create_array(meta_group, "valid_episode_mask", data=valid)
    print(
        f"Saved reference cache to: {output_path} (valid_episodes={int(valid.sum())}/{n_total})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute reference trajectory sampling indices."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Pretrained contact encoder checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zarr path. Supports {split} and {root}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, normalizer, cfg = load_force_vq_checkpoint(args.checkpoint, device)
    data_kwargs = _dataset_kwargs(cfg)
    roots = iter_roots(cfg)
    if not roots:
        raise ValueError("data.root_dir is empty")

    for root in roots:
        per_root_kwargs = dict(data_kwargs)
        per_root_kwargs["root_dir"] = root
        dataset = ForceEpisodeDataset(split=args.split, **per_root_kwargs)
        output_path = resolve_output_path(root, args.split, args.output)
        print(
            f"[force-vq-cache] root={root} split={args.split} episodes={len(dataset)} -> {output_path}"
        )

        if os.path.lexists(output_path):
            if not args.overwrite:
                raise FileExistsError(
                    f"Output zarr already exists: {output_path}. Use --overwrite to rebuild."
                )
            _remove_path(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        precompute_prompts(
            model,
            normalizer,
            dataset,
            output_path,
            batch_size=max(1, int(args.batch_size)),
            device=device,
            source_checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    main()
