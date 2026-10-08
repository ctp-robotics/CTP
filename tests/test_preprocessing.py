"""Check sensor alignment, exported targets, and training-loader compatibility."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import zarr

from datasets.force_episode_dataset import ForceEpisodeDataset
from datasets.policy_zarr_dataset import ZarrDataset
from tools.convert_dataset import convert_dataset, process_episode, timestamps_ms


def write_episode(path: Path, *, offset=0):
    (path / "camera").mkdir(parents=True)
    times = 1_700_000_000_000 + np.arange(96, dtype=np.int64) * 33
    pose = np.zeros((len(times), 6), dtype=np.float64)
    pose[:, 0] = 100 + np.arange(len(times)) + offset
    pose[:, 1] = 200
    pose[:, 3:] = [0.1, 0.2, 0.3]
    entries = {
        "state.pkl": {
            "timestamps": times,
            "eef_pose": pose,
            "joints": np.zeros((96, 7)),
        },
        "gripper.pkl": {"timestamps": times, "gripper_pos": np.full(96, 127.5)},
        "force.pkl": {
            "raw_timestamps": times,
            "ext_timestamps": times,
            "raw": np.broadcast_to(np.arange(6), (96, 6)),
            "ext": np.broadcast_to(np.arange(6) + 10, (96, 6)),
        },
        "tactile.pkl": {
            f"tactile{number}": {
                "timestamps": times,
                "deform": np.full((96, 700, 3), number, dtype=np.float32),
                "mesh": np.zeros((96, 700, 3), dtype=np.float32),
            }
            for number in (1, 2)
        },
        "camera/timestamps.pkl": {
            camera: times * 1_000_000 for camera in ("global", "wrist_left")
        },
    }
    for name, value in entries.items():
        with (path / name).open("wb") as stream:
            pickle.dump(value, stream)
    for camera in ("global", "wrist_left"):
        video = cv2.VideoWriter(
            str(path / "camera" / f"{camera}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (32, 24),
        )
        if not video.isOpened():
            raise RuntimeError("OpenCV could not create the test video.")
        try:
            for _ in times:
                video.write(np.full((24, 32, 3), [20, 80, 160], dtype=np.uint8))
        finally:
            video.release()


class PreprocessingTests(unittest.TestCase):
    def test_export_preserves_sensor_layout_and_loads_for_both_training_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (0, 1):
                write_episode(root / "raw" / f"{index:04d}", offset=index * 50)
            output = convert_dataset(
                root / "raw",
                root / "dataset" / "train",
                behavior_labels={"0000": 0, "0001": 1},
            )
            store = zarr.open_group(str(output), mode="r")
            data, meta = store["data"], store["meta"]
            ends = meta["episode_ends"][:]
            np.testing.assert_array_equal(meta["behavior_id"][:], [0, 1])
            self.assertEqual(len(ends), 2)
            self.assertEqual(data["global_rgb_cam"].shape[1:], (224, 224, 3))
            self.assertEqual(data["gripper1_tactile_30hz"].shape[1:], (35, 20, 3))
            np.testing.assert_array_equal(
                data["finger_force_30hz"][0], [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
            )
            np.testing.assert_array_equal(data["gripper1_tactile_30hz"][0], 1)
            state, action = data["state"][:], data["action"][:]
            np.testing.assert_allclose(state[:, 1], 0.2)
            np.testing.assert_allclose(state[:, 9], 0.5)
            start = 0
            for end in ends:
                np.testing.assert_array_equal(
                    action[start : end - 1], state[start + 1 : end]
                )
                np.testing.assert_array_equal(action[end - 1], state[end - 1])
                start = end
            self.assertFalse(any("120hz" in key for key in data))
            pretrain = ForceEpisodeDataset(str(root / "dataset"), split="train")
            self.assertEqual(pretrain[0]["tactile"].shape[-3:], (35, 20, 6))
            policy = ZarrDataset(
                str(root / "dataset"),
                split="train",
                window_size=8,
                n_image_steps=1,
                action_window_size=32,
                force_steps=8,
                tactile_steps=8,
                future_force_steps=8,
                future_force_offset=1,
                action_representation="chunk_relative",
            )
            sample = policy[0]
            self.assertEqual(sample["action"].shape, (32, 10))
            self.assertEqual(sample["obs"]["image"].shape, (1, 2, 3, 224, 224))
            self.assertEqual(sample["future_force"].shape, (8, 12))
            with self.assertRaises(FileExistsError):
                convert_dataset(root / "raw", root / "dataset" / "train")

    def test_streams_align_by_time_instead_of_frame_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0000"
            write_episode(path)
            with (path / "camera/timestamps.pkl").open("rb") as stream:
                camera_times = pickle.load(stream)
            for key in camera_times:
                camera_times[key] += 100_000_000  # Cameras start 100 ms later.
            with (path / "camera/timestamps.pkl").open("wb") as stream:
                pickle.dump(camera_times, stream)
            times = 1_700_000_000_000 + np.arange(96) * 33
            force = {
                "raw_timestamps": times + 33,
                "ext_timestamps": times + 66,
                "raw": np.repeat(np.arange(96)[:, None], 6, axis=1),
                "ext": np.repeat((np.arange(96) + 100)[:, None], 6, axis=1),
            }
            with (path / "force.pkl").open("wb") as stream:
                pickle.dump(force, stream)
            arrays = process_episode(path, ["global", "wrist_left"], trim=False)
            # First common master sample is 132 ms: robot frame 4,
            # raw-force frame 3, and ext-force frame 2.
            self.assertAlmostEqual(float(arrays["state"][0, 0]), 0.104, places=6)
            np.testing.assert_array_equal(
                arrays["finger_force_30hz"][0], [3] * 6 + [102] * 6
            )

    def test_invalid_episode_does_not_leave_a_partial_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("0000", "0001"):
                write_episode(root / "raw" / name)
            (root / "raw/0001/force.pkl").unlink()
            with self.assertRaisesRegex(ValueError, "0001"):
                convert_dataset(root / "raw", root / "dataset", trim=False)
            self.assertFalse((root / "dataset/replay_buffer.zarr").exists())
            self.assertEqual(list((root / "dataset").iterdir()), [])

    def test_timestamp_units_and_out_of_order_samples(self):
        expected = np.array([1_700_000_000_000, 1_700_000_000_033])
        np.testing.assert_array_equal(timestamps_ms(expected), expected)
        np.testing.assert_array_equal(timestamps_ms(expected * 1000), expected)
        np.testing.assert_array_equal(timestamps_ms(expected * 1_000_000), expected)
        np.testing.assert_array_equal(timestamps_ms(expected / 1000), expected)
        np.testing.assert_array_equal(timestamps_ms([0, 33, 66]), [0, 33, 66])
        np.testing.assert_array_equal(
            timestamps_ms([0, 0.033, 0.066], "s"), [0, 33, 66]
        )
        with self.assertRaises(ValueError):
            timestamps_ms([66, 33])


if __name__ == "__main__":
    unittest.main()
