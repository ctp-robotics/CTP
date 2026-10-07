from __future__ import annotations

from pathlib import Path

import torch

from models.policy.contact_autoencoder import load_force_vq
from models.policy.contact_policy import ContactPolicy
from utils.train_utils import build_canonical_config


def policy_weights(state_dict: dict) -> dict:
    """Read active policy weights, including checkpoints saved through DDP."""
    result = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        # These unused projections appeared in earlier mainline checkpoints.
        if not key.startswith(
            (
                "cond_fuse.",
                "mode_proj.",
                "obs_encoder.visual_query.",
                "obs_encoder.visual_key.",
                "obs_encoder.visual_value.",
            )
        ):
            result[key] = value
    return result


def load_policy_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[ContactPolicy, dict]:
    ckpt = torch.load(
        Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False
    )
    cfg = build_canonical_config(ckpt["config"])
    bundle = ckpt["force_vq_bundle"]
    encoder, encoder_normalizer, encoder_cfg = load_force_vq(bundle, device)
    cfg["model"]["force_vq"] = encoder_cfg["model"]["force_vq"]
    reference = cfg["model"]["policy"]["mode_prompt"]["force_vq"]
    reference.update(num_tokens=encoder.num_tokens, latent_dim=encoder.latent_dim)
    policy = ContactPolicy.from_cfg(cfg).to(device)
    policy.set_trainable_force_vq(encoder, encoder_normalizer)
    policy.load_state_dict(policy_weights(ckpt["policy_state_dict"]), strict=True)
    saved = ckpt.get("trainable_force_vq_normalizer_state_dict")
    if saved:
        policy.trainable_force_vq_normalizer.load_state_dict(saved)
        policy.trainable_force_vq_normalizer.to(device)
    policy.normalizer.load_state_dict(ckpt["normalizer_state_dict"])
    policy.normalizer.to(device)
    policy._force_vq_bundle = bundle
    policy.eval()
    return policy, cfg
