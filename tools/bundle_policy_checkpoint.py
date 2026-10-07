from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.policy.contact_autoencoder import bundle_force_vq_checkpoint  # noqa: E402


def _resolve_vq_checkpoint(policy_cfg: dict, override: Path | None) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
    else:
        mode_prompt = policy_cfg.get("mode_prompt") or {}
        configured = (mode_prompt.get("force_vq") or {}).get("ckpt")
        if not configured:
            raise KeyError("Policy checkpoint has no model.policy.mode_prompt.force_vq.ckpt.")
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"contact encoder checkpoint not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle a CTP policy and its pretrained contact encoder.")
    parser.add_argument("--policy", type=Path, required=True, help="Existing policy checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New bundled checkpoint path")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None, help="Override the contact encoder checkpoint")
    args = parser.parse_args()

    policy_path = args.policy.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path == policy_path:
        raise ValueError("--output must differ from --policy; this tool never overwrites the input checkpoint.")

    policy_ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
    cfg = policy_ckpt.get("config")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("model", {}).get("policy"), dict):
        raise KeyError("Policy checkpoint is missing config.model.policy.")
    vq_path = _resolve_vq_checkpoint(cfg["model"]["policy"], args.encoder_checkpoint)
    policy_ckpt["force_vq_bundle"] = bundle_force_vq_checkpoint(vq_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_ckpt, output_path)
    print(f"Bundled policy: {output_path}")
    print(f"  source policy: {policy_path}")
    print(f"  contact encoder: {vq_path}")


if __name__ == "__main__":
    main()
