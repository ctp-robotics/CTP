# Contact Trajectory Prompting: In-Context Transfer of Contact-Rich Behaviors from a Single Demonstration

Xian Nie<sup>*,1,2</sup>, Yujie Zang<sup>*,1,2,3</sup>, Yuhang Zheng<sup>*,3</sup>, Yupeng Zheng<sup>4</sup>, Songen Gu<sup>5</sup>, Wendi Chen<sup>1</sup>, Chuan Wen<sup>1</sup>, Cewu Lu<sup>1</sup>, Wenchao Ding<sup>2</sup>, Junchi Yan<sup>1</sup>, Shuicheng Yan<sup>3</sup>

¹ Shanghai Jiao Tong University · ² TARS Robotics · ³ National University of Singapore · ⁴ Institute of Automation, Chinese Academy of Sciences · ⁵ Fudan University

*Equal contribution: Xian Nie, Yujie Zang, and Yuhang Zheng.*

Contact Trajectory Prompting conditions a robot action policy on a reference
demonstration containing force, tactile, and position measurements. Online visual,
state, force, and tactile observations attend to continuous reference tokens. A
flow-matching policy predicts actions, with an action-conditioned force prediction
loss during training.

## Method

![Contact Trajectory Prompting overview](docs/method.webp)

Training has two stages:

1. **Contact encoder pretraining:** `ContactAutoencoder` learns a contact representation from relative
   position, force, and tactile sequences. Hybrid temporal/force sampling selects
   128 points, which are encoded into 16 tokens. The encoder uses hand-specific
   contact fusion, modality masking, and the original normalized sample times.
2. **Policy training:** `ContactPolicy` initializes the reference encoder from the
   pretrained checkpoint and fine-tunes its continuous tokens with the action
   policy. The auxiliary head uses online state, force, and predicted actions to
   predict the next 8 steps of force change. The action flow remains 10-dimensional.

The policy uses continuous encoder outputs. The codebook and commitment loss
are used during contact encoder pretraining.

## Installation

Use Python 3.10 or later in a dedicated environment:

```bash
pip install -r requirements.txt
```

The requirements record the environment used for CPU smoke checks. Full training
requires a suitable PyTorch/CUDA installation and the training data. The image
encoder loads pretrained DINOv2 weights through `timm` on first use.

## Data

Set `data.root_dir` in each configuration to a list of dataset directories. The
pretraining list can cover multiple tasks; the policy training list selects the
policy's training data. Paths are intentionally empty in the provided configs.

Each root can contain explicit splits:

```text
dataset_root/
  train/replay_buffer.zarr
  val/replay_buffer.zarr
```

Alternatively, `dataset_root/replay_buffer.zarr` is split by episode using
`data.val_ratio` and `data.split_seed` (or the experiment seed). Explicit split
directories take precedence.

The arrays share an aligned time axis. Configure key names in YAML:

| Array | Shape | Use |
| --- | --- | --- |
| `data/state` | `[T, 10]` | Robot state; the first three coordinates supply reference position |
| `data/action` | `[T, 10]` | Absolute action targets for policy training |
| `data/finger_force_30hz` | `[T, 12]` | Two six-dimensional wrench streams |
| `data/gripper1_tactile_30hz` | `[T, 35, 20, 3]` or `[T, 700, 3]` | First tactile sensor |
| `data/gripper2_tactile_30hz` | `[T, 35, 20, 3]` or `[T, 700, 3]` | Second tactile sensor |
| `data/global_rgb_cam`, `data/left_cam1` | `[T, H, W, 3]` | RGB observations for policy training |
| `meta/episode_ends` | `[N]` | Exclusive cumulative episode ends |
| `meta/behavior_id` | `[N]`, optional | Episode behavior labels for splitting and reference selection |

Keep sensor units, coordinate frames, and calibration consistent across datasets.
Actions are converted to differences from the current state; the final gripper
coordinate remains absolute. The configured auxiliary loss selects force channels
`[6, 7, 8]`. Check this mapping when adapting to a different robot.

References are selected within a dataset root and split, using the same behavior
when labels exist. Another episode is preferred; a singleton falls back to itself.
Episode splitting does not establish a held-out-object evaluation: define object
splits explicitly for that protocol.

## Contact encoder pretraining

Edit `config/pretrain.yaml` and set `data.root_dir`, then run:

```bash
python train.py --config config/pretrain.yaml
```

The default run writes checkpoints under `outputs/pretrain/checkpoints/`.

## Policy training

In `config/policy.yaml`, set:

- `data.root_dir`: policy dataset roots.
- `model.policy.reference_encoder.checkpoint`: the pretrained checkpoint, such as
  `outputs/pretrain/checkpoints/best.pt`.

```bash
python train.py --config config/policy.yaml
```

The trainer loads the encoder architecture from the pretrained checkpoint and
prepares reference caches for the policy dataset. Continuous encoder
training uses their sampling indices and re-encodes reference measurements on each
step. Cache files are written under each dataset root, so these directories must
be writable. Checkpoints are written under `outputs/policy/checkpoints/`.

For multiple GPUs:

```bash
bash scripts/train.sh --gpus 0,1 --ddp --config config/policy.yaml
```

Raw RGB is used by default. To cache frozen image features, set
`data.latent_cache_root_dir: auto`, keep the image backbone frozen, and run:

```bash
bash scripts/precompute.sh config/policy.yaml 0
```

This script computes train/validation image caches. Start policy training separately
with `python train.py --config config/policy.yaml`.

## Evaluation

```bash
python tools/eval_open_loop.py \
  --checkpoint outputs/policy/checkpoints/best.pt \
  --split val --max-batches 20
```

Open-loop evaluation measures predictions against recorded actions. It does not
measure robot task success or replace a closed-loop rollout protocol. Evaluating
on another data root also requires matching reference caches for that root.

## Code

| File | Responsibility |
| --- | --- |
| `models/policy/contact_autoencoder.py` | Contact sampling, representation learning, and checkpoint loading |
| `models/policy/contact_policy.py` | Continuous reference conditioning, action flow, and force prediction |
| `models/policy/contact_prompt.py` | Attention between online observations and reference tokens |
| `trainers/pretrain_trainer.py` | Contact representation pretraining |
| `trainers/policy_trainer.py` | Policy training |

Policy encoder architecture and token dimensions are loaded from the pretraining
checkpoint.

## Smoke check

```bash
python -m tools.smoke_test
```

This uses synthetic inputs and reduced action-network widths on CPU, without
downloading pretrained weights. It checks both training paths, force-loss
gradients, and checkpoint reloads. It does not validate task performance.

## Citation

```bibtex
@misc{nie2026contacttrajectoryprompting,
  title = {Contact Trajectory Prompting: In-Context Transfer of Contact-Rich Behaviors from a Single Demonstration},
  author = {Nie, Xian and Zang, Yujie and Zheng, Yuhang and Zheng, Yupeng and Gu, Songen and Chen, Wendi and Wen, Chuan and Lu, Cewu and Ding, Wenchao and Yan, Junchi and Yan, Shuicheng},
  year = {2026}
}
```
