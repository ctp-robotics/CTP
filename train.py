from __future__ import annotations

import argparse

from utils.ddp import maybe_reexec_torchrun
from utils.train_utils import build_canonical_config, validate_training_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact Trajectory Prompting training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["pretrain", "policy"],
        help="Pretrain the contact encoder or post-train the action policy; defaults to the config stage.",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Post-training DDP. Use scripts/train.sh --gpus 0,1 --ddp.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="none",
        choices=["none", "dataloader", "model", "both"],
    )
    parser.add_argument("--bench-steps", type=int, default=100)
    parser.add_argument("--bench-warmup", type=int, default=10)
    args = parser.parse_args()

    overrides = {
        "runtime.benchmark": args.benchmark,
        "runtime.bench_steps": args.bench_steps,
        "runtime.bench_warmup": args.bench_warmup,
        "runtime.ddp": bool(args.ddp),
    }
    config_path = args.config or f"config/{args.stage or 'policy'}.yaml"
    cfg = build_canonical_config(config_path, overrides=overrides)
    stage = cfg.get("stage", "policy")
    if args.stage is not None and args.stage != stage:
        parser.error(f"--stage {args.stage} does not match config stage {stage}.")
    validate_training_paths(cfg)
    maybe_reexec_torchrun(ddp=bool(args.ddp), stage=stage)

    if stage == "pretrain":
        from trainers.pretrain_trainer import main as stage_main
    else:
        from trainers.policy_trainer import main as stage_main

    stage_main(cfg)


if __name__ == "__main__":
    main()
