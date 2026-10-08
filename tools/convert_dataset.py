"""Convert recorded camera videos and sensor pickles to the CTP Zarr format.

Each episode contains camera/{global,wrist_left,wrist_right}.mp4,
camera/timestamps.pkl (camera name -> timestamps), and these sensor dictionaries:
  state.pkl: timestamps, eef_pose [N,6] (xyz in mm, xyz Euler angles in radians)
  gripper.pkl: timestamps, gripper_pos [N] (0..255)
  tactile.pkl: tactile1/tactile2 -> {timestamps, deform [N,700,3] or [N,35,20,3]}
  force.pkl: raw, ext [N,6], raw_timestamps, ext_timestamps

Camera decoding preserves BGR. Force channels are concatenated as raw then ext.
State positions are converted to metres, rotations to the first two matrix rows,
and gripper positions to 0..1. Action targets are the next aligned state, with
the last state repeated. Dataset normalization is fitted later during training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation
from tqdm import tqdm

CAMERA_KEYS = {
    "global": "global_rgb_cam",
    "wrist_left": "left_cam1",
    "wrist_right": "right_cam0",
}


def read_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def timestamps_ms(values, unit: str = "auto") -> np.ndarray:
    """Convert epoch timestamps or relative millisecond timestamps to milliseconds."""
    values = np.asarray(values).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Timestamps must be nonempty and finite.")
    if unit == "auto":
        magnitude = float(np.median(np.abs(values)))
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif 1e9 <= magnitude < 1e11:
            unit = "s"
        else:
            unit = "ms"
    if unit == "ns":
        result = (values // 1_000_000).astype(np.int64)
    elif unit == "us":
        result = (values // 1000).astype(np.int64)
    elif unit == "s":
        result = np.rint(values * 1000).astype(np.int64)
    else:
        result = values.astype(np.int64)
    if np.any(np.diff(result) < 0):
        raise ValueError("Timestamps must be sorted in increasing order.")
    return result


def nearest_indices(reference: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Match timestamps to their nearest source frame, preferring earlier ties."""
    right = np.searchsorted(source, reference, side="left").clip(0, len(source) - 1)
    left = (right - 1).clip(0, len(source) - 1)
    return np.where(
        np.abs(source[left] - reference) <= np.abs(source[right] - reference),
        left,
        right,
    )


def sensor_stream(timestamps, values, unit: str, name: str, dtype=np.float32):
    times = timestamps_ms(timestamps, unit)
    array = np.asarray(values, dtype=dtype)
    if array.ndim < 1 or len(array) != len(times):
        raise ValueError(f"{name}: sample count does not match its timestamps.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: sensor values contain NaN or infinity.")
    return times, array


def read_video(path: Path) -> np.ndarray:
    video = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not video.isOpened():
            raise ValueError(f"Cannot open camera video: {path}")
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        video.release()
    if not frames:
        raise ValueError(f"Camera video contains no frames: {path}")
    return np.stack(frames)


def load_episode(path: Path, cameras: list[str], unit: str) -> dict:
    streams = {}
    camera_times = read_pickle(path / "camera" / "timestamps.pkl")
    for name in cameras:
        frames = read_video(path / "camera" / f"{name}.mp4")
        times = timestamps_ms(camera_times[name], unit)
        if len(times) != len(frames):
            raise ValueError(f"{name}: video frame count does not match timestamps.")
        streams[CAMERA_KEYS[name]] = times, frames

    state = read_pickle(path / "state.pkl")
    streams["pose"] = sensor_stream(
        state["timestamps"], state["eef_pose"], unit, "pose", dtype=np.float64
    )
    if streams["pose"][1].ndim != 2 or streams["pose"][1].shape[1] != 6:
        raise ValueError("eef_pose must contain [x,y,z,rx,ry,rz] with shape [N,6].")
    gripper = read_pickle(path / "gripper.pkl")
    streams["gripper"] = sensor_stream(
        gripper["timestamps"],
        np.asarray(gripper["gripper_pos"]).reshape(-1, 1),
        unit,
        "gripper",
    )
    tactile = read_pickle(path / "tactile.pkl")
    for number in (1, 2):
        entry = tactile[f"tactile{number}"]
        name = f"gripper{number}_tactile_30hz"
        times, deform = sensor_stream(entry["timestamps"], entry["deform"], unit, name)
        if deform.shape[1:] == (700, 3):
            deform = deform.reshape(-1, 35, 20, 3)
        if deform.shape[1:] != (35, 20, 3):
            raise ValueError(f"{name}: expected a 35 x 20 x 3 displacement field.")
        streams[name] = times, deform

    force = read_pickle(path / "force.pkl")
    raw_times, raw = sensor_stream(
        force["raw_timestamps"], force["raw"], unit, "raw force"
    )
    ext_times, ext = sensor_stream(
        force["ext_timestamps"], force["ext"], unit, "ext force"
    )
    if raw.shape[1:] != (6,) or ext.shape[1:] != (6,):
        raise ValueError(
            "Each finger force stream must have six force/torque channels."
        )
    # The second force sensor supplies the common bilateral force timeline.
    bilateral = np.concatenate(
        [raw[nearest_indices(ext_times, raw_times)], ext], axis=-1
    )
    streams["finger_force_30hz"] = ext_times, bilateral
    return streams


def crop_to_common_interval(streams: dict) -> dict:
    # Gripper samples use an independent clock and nearest-neighbor matching.
    timed = [times for key, (times, _) in streams.items() if key != "gripper"]
    start, end = max(ts[0] for ts in timed), min(ts[-1] for ts in timed)
    master = streams["gripper1_tactile_30hz"][0]
    master = master[(master >= start) & (master <= end)]
    if not len(master):
        raise ValueError("Sensor streams do not have a common recording interval.")
    cropped = {}
    for key, (times, values) in streams.items():
        if key == "gripper":
            cropped[key] = times, values
            continue
        selected = (times >= master[0]) & (times <= master[-1])
        if not selected.any():
            raise ValueError(f"{key}: no samples within the common recording interval.")
        cropped[key] = times[selected], values[selected]
    return cropped


def motion_interval(position: np.ndarray, times: np.ndarray, threshold: float):
    """Trim idle margins using sustained TCP displacement in millimetres."""
    step, window, hits = 2, 5, 3
    if len(position) <= step:
        raise ValueError("Too few robot samples for motion trimming; use --no-trim.")
    moving = (
        np.linalg.norm(position[step:, :3] - position[:-step, :3], axis=1) > threshold
    )
    window = min(window, len(moving))
    support = np.convolve(
        moving.astype(np.int16), np.ones(window, dtype=np.int16), mode="valid"
    )
    stable = np.flatnonzero(support >= min(hits, window))
    if len(stable):
        first, last = int(stable[0]), int(stable[-1] + window - 1 + step)
    elif moving.any():
        found = np.flatnonzero(moving)
        first, last = int(found[0]), int(found[-1] + step)
    else:
        first, last = 0, len(position) - 1
    first = min(max(first - 10, 5), len(position) - 1)
    last = min(last + 20, len(position) - 1)
    if first >= last:
        raise ValueError("Motion trimming leaves no usable interval; use --no-trim.")
    return times[first], times[last]


def process_episode(
    path: Path,
    cameras: list[str],
    *,
    image_size=224,
    timestamp_unit="auto",
    trim=True,
    motion_threshold=0.1,
) -> dict:
    streams = crop_to_common_interval(load_episode(path, cameras, timestamp_unit))
    master = streams["gripper1_tactile_30hz"][0]
    targets = master[0] + np.arange(
        int(np.floor((master[-1] - master[0]) / (1000 / 30))) + 1
    ) * (1000 / 30)
    timeline = master[nearest_indices(targets, master)]
    if trim:
        pose_times, pose = streams["pose"]
        first, last = motion_interval(pose, pose_times, motion_threshold)
        timeline = timeline[(timeline >= first) & (timeline <= last)]
    if len(timeline) < 2:
        raise ValueError(f"{path.name}: fewer than two aligned frames.")

    arrays = {
        key: values[nearest_indices(timeline, times)]
        for key, (times, values) in streams.items()
    }
    for camera in cameras:
        key = CAMERA_KEYS[camera]
        arrays[key] = np.stack(
            [
                cv2.resize(
                    frame, (image_size, image_size), interpolation=cv2.INTER_AREA
                )
                for frame in arrays[key]
            ]
        ).astype(np.uint8)
    pose = arrays.pop("pose").astype(np.float32)
    rotation = Rotation.from_euler("xyz", pose[:, 3:6].astype(np.float64)).as_matrix()
    rot6d = np.concatenate([rotation[:, 0, :], rotation[:, 1, :]], axis=-1)
    state = np.concatenate(
        [pose[:, :3] / 1000, rot6d, arrays.pop("gripper") / 255], axis=-1
    )
    arrays["state"] = state.astype(np.float32)
    arrays["action"] = np.concatenate([arrays["state"][1:], arrays["state"][-1:]])
    return arrays


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    cameras=None,
    image_size=224,
    timestamp_unit="auto",
    trim=True,
    motion_threshold=0.1,
    behavior_labels: dict | None = None,
) -> Path:
    input_dir, output_dir = Path(input_dir).expanduser(), Path(output_dir).expanduser()
    cameras = list(cameras or ["global", "wrist_left"])
    if len(set(cameras)) != len(cameras) or any(
        name not in CAMERA_KEYS for name in cameras
    ):
        raise ValueError(
            "Select distinct camera names from global, wrist_left, wrist_right."
        )
    if image_size < 1 or motion_threshold < 0:
        raise ValueError(
            "Image size must be positive and motion threshold nonnegative."
        )
    if timestamp_unit not in {"auto", "s", "ms", "us", "ns"}:
        raise ValueError("Unsupported timestamp unit.")
    episodes = sorted(path.parent for path in input_dir.rglob("state.pkl"))
    if not episodes:
        raise ValueError(f"No episodes containing state.pkl found under {input_dir}.")
    labels = []
    if behavior_labels is not None:
        for episode in episodes:
            key = episode.relative_to(input_dir).as_posix()
            label = behavior_labels[key]
            if type(label) is not int or label < 0:
                raise ValueError(
                    f"Behavior label for {key} must be a nonnegative integer."
                )
            labels.append(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "replay_buffer.zarr"
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".replay_buffer-", dir=output_dir))
    try:
        root = zarr.open_group(str(staging), mode="w")
        data, meta = root.create_group("data"), root.create_group("meta")
        ends, total = [], 0
        for episode in tqdm(episodes, desc="Converting episodes"):
            try:
                arrays = process_episode(
                    episode,
                    cameras,
                    image_size=image_size,
                    timestamp_unit=timestamp_unit,
                    trim=trim,
                    motion_threshold=motion_threshold,
                )
            except Exception as exc:
                raise ValueError(f"Failed to convert episode {episode}: {exc}") from exc
            for key, array in arrays.items():
                if key not in data:
                    frame_bytes = int(np.prod(array.shape[1:])) * array.dtype.itemsize
                    chunk_frames = min(128, max(1, 4 * 1024 * 1024 // frame_bytes))
                    data.create_dataset(
                        key,
                        shape=(0, *array.shape[1:]),
                        chunks=(chunk_frames, *array.shape[1:]),
                        dtype=array.dtype,
                        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                    )
                data[key].append(array, axis=0)
            total += len(arrays["state"])
            ends.append(total)
        meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
        if labels:
            meta.create_dataset("behavior_id", data=np.asarray(labels, dtype=np.int64))
        root.attrs.update(
            fps=30,
            image_color_order="BGR",
            camera_keys=[CAMERA_KEYS[c] for c in cameras],
            position_unit="m",
            rotation_representation="rotation_matrix_first_two_rows",
            force_order=["raw", "ext"],
            action_target="next_aligned_state",
            motion_trimming=bool(trim),
            motion_threshold_mm=float(motion_threshold),
            source_timestamp_unit=timestamp_unit,
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Saved {len(ends)} episodes ({total} frames) to {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw episode directory or parent containing episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which to create replay_buffer.zarr",
    )
    parser.add_argument(
        "--cameras", nargs="+", choices=CAMERA_KEYS, default=["global", "wrist_left"]
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="auto detects epoch units and treats relative timestamps as milliseconds",
    )
    parser.add_argument(
        "--no-trim", action="store_true", help="Keep the full common sensor interval"
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.1,
        help="TCP displacement threshold in millimetres",
    )
    parser.add_argument(
        "--behavior-labels",
        type=Path,
        help='JSON mapping relative episode paths to IDs, e.g. {"0000": 0}',
    )
    args = parser.parse_args()
    labels = (
        json.loads(args.behavior_labels.read_text()) if args.behavior_labels else None
    )
    convert_dataset(
        args.input,
        args.output,
        cameras=args.cameras,
        image_size=args.image_size,
        timestamp_unit=args.timestamp_unit,
        trim=not args.no_trim,
        motion_threshold=args.motion_threshold,
        behavior_labels=labels,
    )


if __name__ == "__main__":
    main()
