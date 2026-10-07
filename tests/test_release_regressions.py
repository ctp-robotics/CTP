"""Regressions for split-specific references and reference cache reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zarr

from trainers import policy_trainer as trainer
from tools import eval_open_loop as evaluator
from utils.train_utils import build_canonical_config


class EmptyLoader(list):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset


class ReferencePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable_force_vq = SimpleNamespace(
            temporal_position_mode="normalized_phase"
        )
        self.selected = None

    def set_trainable_force_vq_reference_table(self, pos, force, tactile, phase):
        self.selected = pos


def table_for(dataset, *, include_phase):
    assert include_phase
    return dataset, None, None, None


class ReferenceRegressionTests(unittest.TestCase):
    def test_training_validation_and_open_loop_bind_the_active_split(self):
        policy = ReferencePolicy()
        training, validation = object(), object()
        train_loader, val_loader = EmptyLoader(training), EmptyLoader(validation)
        device = torch.device("cpu")
        with patch.object(
            trainer, "collect_force_vq_reference_tables", side_effect=table_for
        ):
            trainer.train_one_epoch(policy, train_loader, None, device)
            self.assertIs(policy.selected, training)
            trainer.validate_one_epoch(policy, val_loader, device, epoch=0)
            self.assertIs(policy.selected, validation)
            trainer.train_one_epoch(policy, train_loader, None, device)
            self.assertIs(policy.selected, training)
            trainer.evaluate_open_loop(policy, val_loader, device, 0, "val", 1)
            self.assertIs(policy.selected, validation)

    def test_cli_both_switches_reference_table_before_each_evaluation(self):
        policy = ReferencePolicy()
        splits = {"train": object(), "val": object()}
        observed = []

        def evaluate(model, dataset, *args, **kwargs):
            self.assertIs(model.selected, dataset)
            observed.append(dataset)
            return {}

        with patch.object(
            trainer, "collect_force_vq_reference_tables", side_effect=table_for
        ), patch.object(
            evaluator, "load_policy_from_checkpoint", return_value=(policy, {})
        ), patch.object(
            evaluator, "_dataset_kwargs_from_cfg", return_value={}
        ), patch.object(
            evaluator,
            "build_zarr_dataset",
            side_effect=lambda split, **kw: splits[split],
        ), patch.object(
            evaluator, "DataLoader", side_effect=lambda ds, **kw: EmptyLoader(ds)
        ), patch.object(
            evaluator, "evaluate_open_loop", side_effect=evaluate
        ), patch.object(
            evaluator, "_summarize_metrics"
        ), patch(
            "sys.argv",
            [
                "eval_open_loop",
                "--checkpoint",
                "unused.pt",
                "--split",
                "both",
                "--device",
                "cpu",
            ],
        ):
            # main() reports dataset lengths; a zero-length list has its own identity.
            splits = {"train": [], "val": []}
            evaluator.main()
        self.assertEqual(len(observed), 2)
        self.assertIs(observed[0], splits["train"])
        self.assertIs(observed[1], splits["val"])

    def test_zarr_cache_reuse_and_checkpoint_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "encoder.pt"
            checkpoint.write_bytes(b"checkpoint")
            cache_path = str(Path(directory) / "reference.zarr")
            cache = zarr.open_group(cache_path, mode="w")
            dataset = SimpleNamespace(
                zarr_path="source.zarr",
                num_episodes_total=2,
                episode_ends=np.array([8, 16]),
                episode_indices=np.array([1]),
            )
            cache.attrs.update(
                source_checkpoint=str(checkpoint),
                source_checkpoint_mtime=int(checkpoint.stat().st_mtime),
                source_zarr_path=dataset.zarr_path,
            )
            data, meta = cache.create_group("data"), cache.create_group("meta")
            data.create_dataset("sample_indices", shape=(2, 128), dtype="i8")
            meta.create_dataset("valid_episode_mask", data=np.array([False, True]))
            meta.create_dataset("episode_ends", data=dataset.episode_ends)
            meta.create_dataset(
                "selected_episode_indices", data=dataset.episode_indices
            )
            self.assertTrue(
                trainer._cache_is_current(cache_path, str(checkpoint), dataset)
            )
            del data["sample_indices"]
            self.assertFalse(
                trainer._cache_is_current(cache_path, str(checkpoint), dataset)
            )
            data.create_dataset("sample_indices", shape=(2, 128), dtype="i8")
            cache.attrs["source_checkpoint_mtime"] = -1
            self.assertFalse(
                trainer._cache_is_current(cache_path, str(checkpoint), dataset)
            )

    def test_pretraining_cache_and_policy_use_the_same_behavior_split(self):
        from datasets.force_episode_dataset import ForceEpisodeDataset
        from datasets.policy_zarr_dataset import ZarrDataset

        with tempfile.TemporaryDirectory() as directory:
            root = zarr.open_group(
                str(Path(directory) / "replay_buffer.zarr"), mode="w"
            )
            data, meta = root.create_group("data"), root.create_group("meta")
            meta.create_dataset("episode_ends", data=np.arange(1, 9) * 48)
            meta.create_dataset("behavior_id", data=np.repeat([0, 1], 4))
            for key, shape in {
                "state": (384, 10),
                "action": (384, 10),
                "finger_force_30hz": (384, 12),
                "gripper1_tactile_30hz": (384, 35, 20, 3),
                "gripper2_tactile_30hz": (384, 35, 20, 3),
                "global_rgb_cam": (384, 8, 8, 3),
                "left_cam1": (384, 8, 8, 3),
            }.items():
                data.create_dataset(key, shape=shape, dtype="f4")
            selected = {}
            for split in ("train", "val"):
                kwargs = dict(
                    root_dir=directory, split=split, val_ratio=0.25, split_seed=123
                )
                pre = ForceEpisodeDataset(**kwargs)
                policy = ZarrDataset(
                    **kwargs,
                    window_size=8,
                    action_window_size=32,
                    image_keys=["global_rgb_cam", "left_cam1"],
                    state_key="state",
                    action_key="action",
                )
                np.testing.assert_array_equal(
                    pre.episode_indices, policy.episode_indices
                )
                selected[split] = set(pre.episode_indices.tolist())
            self.assertEqual(len(selected["val"]), 2)
            self.assertFalse(selected["train"] & selected["val"])

    def test_cache_generation_uses_policy_split_and_custom_destination(self):
        cfg = build_canonical_config("config/policy.yaml")
        pre = build_canonical_config("config/pretrain.yaml")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "encoder.pt"
            checkpoint.touch()
            cfg["model"]["policy"]["mode_prompt"]["force_vq"]["ckpt"] = str(checkpoint)
            cfg["data"].update(
                root_dir=[directory],
                val_ratio=0.3,
                split_seed=123,
                force_vq_cache_root_dir=str(Path(directory) / "cache"),
            )
            with patch(
                "tools.precompute_reference_cache.load_force_vq_checkpoint",
                return_value=(None, None, pre),
            ), patch(
                "datasets.force_episode_dataset.ForceEpisodeDataset"
            ) as dataset, patch.object(
                trainer, "_cache_is_current", return_value=True
            ) as current:
                trainer._ensure_force_vq_caches(cfg, torch.device("cpu"))
            self.assertEqual(dataset.call_count, 2)
            for call, split in zip(dataset.call_args_list, ["train", "val"]):
                self.assertEqual(call.kwargs["split_seed"], 123)
                self.assertEqual(call.kwargs["val_ratio"], 0.3)
                self.assertEqual(call.kwargs["split"], split)
            self.assertEqual(
                current.call_args_list[0].args[0],
                str(Path(directory) / "cache/train/force_vq_prompt_cache.zarr"),
            )


if __name__ == "__main__":
    unittest.main()
