from __future__ import annotations
import math
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.normalizer import MultiFieldNormalizer
from utils.train_utils import build_canonical_config


def load_force_vq(
    source: str | Path | Mapping, device: torch.device | str
) -> tuple["ContactAutoencoder", MultiFieldNormalizer, dict]:
    """Load the pretrained contact encoder from a checkpoint or embedded bundle."""
    if isinstance(source, Mapping):
        ckpt = dict(source)
    else:
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    state_dict = ckpt.get("force_vq_state_dict")
    normalizer_state = ckpt.get("force_normalizer_state_dict")
    if (
        not isinstance(cfg, dict)
        or not isinstance(state_dict, dict)
        or (not isinstance(normalizer_state, dict))
    ):
        raise KeyError(
            "contact encoder source must contain config, force_vq_state_dict, and force_normalizer_state_dict."
        )
    cfg = build_canonical_config(cfg)
    model = ContactAutoencoder.from_cfg(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    normalizer = MultiFieldNormalizer()
    normalizer.load_state_dict(normalizer_state)
    normalizer.to(device)
    return (model, normalizer, cfg)


def bundle_force_vq_checkpoint(checkpoint: str | Path) -> dict:
    """Read a standalone contact encoder checkpoint into the policy-checkpoint bundle format."""
    path = Path(checkpoint).expanduser().resolve()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = ("config", "force_vq_state_dict", "force_normalizer_state_dict")
    missing = [key for key in required if not isinstance(ckpt.get(key), dict)]
    if missing:
        raise KeyError(f"contact encoder checkpoint missing bundle fields: {missing}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "format": "force_vq_bundle/v1",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": digest,
        "config": deepcopy(ckpt["config"]),
        "force_vq_state_dict": deepcopy(ckpt["force_vq_state_dict"]),
        "force_normalizer_state_dict": deepcopy(ckpt["force_normalizer_state_dict"]),
    }


def _sinusoidal_positional_encoding(
    length: int, dim: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if dim < 2:
        return torch.zeros(length, dim, device=device, dtype=dtype)
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half = dim // 2
    scale = torch.arange(half, device=device, dtype=dtype)
    scale = torch.exp(-math.log(10000.0) * scale / max(half - 1, 1))
    emb = torch.cat([torch.sin(pos * scale), torch.cos(pos * scale)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def _sinusoidal_coordinate_encoding(
    coordinates: torch.Tensor, dim: int
) -> torch.Tensor:
    """Sinusoidal encoding for explicit, normalized temporal coordinates.

    ``coordinates`` is ``[B,T]`` and normally contains the original selected
    frame index divided by the valid episode duration.  Keeping this separate
    from rank encoding matters for hybrid sampling: adjacent selected samples
    need not be equally spaced in physical time.
    """
    if coordinates.ndim != 2:
        raise ValueError(f"coordinates must be [B,T], got {tuple(coordinates.shape)}")
    if dim < 2:
        return coordinates.new_zeros(*coordinates.shape, dim)
    pos = coordinates.unsqueeze(-1)
    half = dim // 2
    scale = torch.arange(half, device=coordinates.device, dtype=coordinates.dtype)
    scale = torch.exp(-math.log(10000.0) * scale / max(half - 1, 1))
    emb = torch.cat([torch.sin(pos * scale), torch.cos(pos * scale)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[..., :dim]


class TemporalConvEncoder(nn.Module):
    """Residual temporal Conv1d encoder for fixed-length sampled sequences."""

    def __init__(
        self, dim: int, num_layers: int = 4, kernel_size: int = 5, dropout: float = 0.1
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            dilation = 2 ** (layer_idx % 4)
            padding = kernel_size // 2 * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Conv1d(
                        dim, dim, kernel_size, padding=padding, dilation=dilation
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(dim, dim, 1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            residual = x
            h = block[0](x).transpose(1, 2)
            h = block[1](h)
            h = block[2](h)
            h = block[3](h)
            h = block[4](h).transpose(1, 2)
            x = residual + h
        return x


class TokenVectorQuantizer(nn.Module):
    """Shared codebook quantizer for [B, K, D] token latents."""

    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.beta = float(beta)
        self.codebook = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_e.ndim != 3 or z_e.shape[-1] != self.dim:
            raise ValueError(
                f"TokenVectorQuantizer expected [B,K,{self.dim}], got {tuple(z_e.shape)}"
            )
        flat = z_e.reshape(-1, self.dim)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).reshape_as(z_e)
        commitment = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.beta * commitment
        z_q_st = z_e + (z_q - z_e).detach()
        return {
            "z_q": z_q_st,
            "z_q_hard": z_q,
            "indices": indices.reshape(z_e.shape[:2]),
            "vq_loss": vq_loss,
            "commitment_loss": commitment.detach(),
            "codebook_loss": codebook_loss.detach(),
        }


class TactileFrameEncoder(nn.Module):
    """Encode one tactile frame with one or two learned spatial reductions."""

    def __init__(self, channels: int, feature_dim: int, num_downsamples: int = 1):
        super().__init__()
        if num_downsamples not in {1, 2}:
            raise ValueError(f"num_downsamples must be 1 or 2, got {num_downsamples}")
        layers: list[nn.Module] = [
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        ]
        if num_downsamples == 2:
            layers.extend(
                [
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                ]
            )
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(64, feature_dim), nn.LayerNorm(feature_dim)
        )

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(tactile))


class HandContactFusion(nn.Module):
    """Shared force--tactile fusion for one hand at one timestep."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(
        self, force: torch.Tensor, tactile: torch.Tensor | None
    ) -> torch.Tensor:
        if tactile is None:
            return self.out(force)
        tokens = torch.stack([force, tactile], dim=2)
        bsz, steps, _, dim = tokens.shape
        tokens = tokens.reshape(bsz * steps, 2, dim)
        fused, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.out(fused.mean(dim=1).reshape(bsz, steps, dim))


class MotionContactFusion(nn.Module):
    """Use motion as a query over the two hand-contact descriptors."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(
        self,
        motion: torch.Tensor,
        left_contact: torch.Tensor,
        right_contact: torch.Tensor,
    ) -> torch.Tensor:
        bsz, steps, dim = motion.shape
        query = motion.reshape(bsz * steps, 1, dim)
        contacts = torch.stack([left_contact, right_contact], dim=2).reshape(
            bsz * steps, 2, dim
        )
        attended, _ = self.attn(query, contacts, contacts, need_weights=False)
        return self.out(torch.cat([query, attended], dim=-1)).reshape(bsz, steps, dim)


class ContactAutoencoder(nn.Module):
    """Encode force, position, and tactile trajectories; pretrain by reconstructing sampled trajectories."""

    @classmethod
    def from_cfg(cls, cfg):
        architecture = cfg["model"]["force_vq"]
        expected = {
            "encoder_type": "temporal_conv",
            "pooling_type": "adaptive_avg",
            "sampling_type": "hybrid",
            "fps_feature_source": "force",
            "tactile_preprocess": "strided_cnn",
            "contact_fusion": True,
            "temporal_position_mode": "normalized_phase",
        }
        for key, value in expected.items():
            if architecture.get(key, value) != value:
                raise ValueError(
                    f"Unsupported contact encoder setting: {key}={architecture[key]!r}"
                )
        fields = [
            "beta",
            "codebook_size",
            "conv_kernel_size",
            "dropout",
            "force_dim",
            "fps_smoothing_kernel",
            "hidden_dim",
            "latent_dim",
            "masked_modality_probability",
            "num_heads",
            "num_layers",
            "num_tokens",
            "pos_dim",
            "recon_force_weight",
            "recon_pos_weight",
            "recon_tactile_weight",
            "sample_points",
            "tactile_channels",
            "uniform_ratio",
        ]
        return cls(
            **{key: value for key, value in architecture.items() if key in fields},
            tactile_shape=cfg.get("data", {}).get("tactile_shape", [35, 20]),
        )

    tactile_preprocess = "strided_cnn"
    encoder_type = "temporal_conv"
    pooling_type = "adaptive_avg"
    sampling_type = "hybrid"
    tokenization = "sample_pool"
    fps_feature_source = "force"
    contact_fusion = True
    temporal_position_mode = "normalized_phase"

    def __init__(
        self,
        force_dim: int = 12,
        pos_dim: int = 3,
        tactile_channels: int = 6,
        tactile_shape: tuple[int, int] | list[int] | None = None,
        sample_points: int = 128,
        num_tokens: int = 16,
        latent_dim: int = 128,
        codebook_size: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        conv_kernel_size: int = 5,
        dropout: float = 0.1,
        beta: float = 0.25,
        uniform_ratio: float = 0.5,
        fps_smoothing_kernel: int = 3,
        recon_pos_weight: float = 1.0,
        recon_force_weight: float = 1.0,
        recon_tactile_weight: float = 1.0,
        masked_modality_probability: float = 0.25,
    ):
        super().__init__()
        self.force_dim = int(force_dim)
        self.pos_dim = int(pos_dim)
        self.tactile_channels = int(tactile_channels)
        if tactile_shape is None or len(tactile_shape) != 2:
            raise ValueError("strided_cnn requires data.tactile_shape=[height, width].")
        self.tactile_input_h, self.tactile_input_w = (int(v) for v in tactile_shape)
        if self.tactile_input_h < 1 or self.tactile_input_w < 1:
            raise ValueError(
                f"data.tactile_shape must be positive, got {tactile_shape}"
            )
        self.tactile_cnn_downsamples = 2
        self.tactile_dim = (
            self.tactile_channels * self.tactile_input_h * self.tactile_input_w
        )
        self.traj_dim = self.pos_dim + self.force_dim + self.tactile_dim
        self.sample_points = int(sample_points)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = int(codebook_size)
        self.uniform_ratio = float(uniform_ratio)
        self.fps_smoothing_kernel = int(fps_smoothing_kernel)
        if self.sample_points < 1:
            raise ValueError(f"sample_points must be >= 1, got {sample_points}")
        if self.num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if self.traj_dim < 1:
            raise ValueError("At least one input dimension is required.")
        if not 0.0 <= self.uniform_ratio <= 1.0:
            raise ValueError(f"uniform_ratio must be in [0,1], got {uniform_ratio}")
        if self.force_dim < 1:
            raise ValueError("fps_feature_source='force' requires force_dim > 0")
        if self.fps_smoothing_kernel < 1 or self.fps_smoothing_kernel % 2 == 0:
            raise ValueError(
                f"fps_smoothing_kernel must be a positive odd integer, got {fps_smoothing_kernel}"
            )
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if self.num_tokens > self.sample_points:
            raise ValueError(
                f"adaptive_avg pooling requires num_tokens <= sample_points, got {self.num_tokens} > {self.sample_points}"
            )
        self.recon_pos_weight = float(recon_pos_weight)
        self.recon_force_weight = float(recon_force_weight)
        self.recon_tactile_weight = float(recon_tactile_weight)
        self.masked_modality_probability = float(masked_modality_probability)
        if not 0.0 <= self.masked_modality_probability <= 1.0:
            raise ValueError("masked_modality_probability must be in [0, 1]")
        if self.force_dim < 2 or self.force_dim % 2 != 0:
            raise ValueError(
                "contact_fusion requires an even force_dim with left/right wrench channels"
            )
        if self.tactile_channels % 2 != 0:
            raise ValueError("contact_fusion requires an even tactile_channels count")
        if self.pos_dim < 1:
            raise ValueError(
                "contact_fusion requires position or rotation motion input"
            )
        hand_force_dim = self.force_dim // 2
        hand_tactile_channels = self.tactile_channels // 2
        motion_dim = self.pos_dim
        self.motion_encoder = self._vector_modality_encoder(motion_dim, self.hidden_dim)
        self.hand_force_encoder = self._vector_modality_encoder(
            hand_force_dim, self.hidden_dim
        )
        self.hand_tactile_encoder = (
            TactileFrameEncoder(
                hand_tactile_channels,
                self.hidden_dim,
                num_downsamples=self.tactile_cnn_downsamples,
            )
            if hand_tactile_channels > 0
            else None
        )
        self.hand_contact_fusion = HandContactFusion(
            self.hidden_dim, int(num_heads), float(dropout)
        )
        self.motion_contact_fusion = MotionContactFusion(
            self.hidden_dim, int(num_heads), float(dropout)
        )
        self.hand_embedding = nn.Embedding(2, self.hidden_dim)
        self.missing_modality = nn.ParameterDict(
            {
                key: nn.Parameter(torch.randn(self.hidden_dim) * 0.02)
                for key in (
                    "left_force",
                    "right_force",
                    "left_tactile",
                    "right_tactile",
                )
            }
        )
        self.temporal_encoder = TemporalConvEncoder(
            dim=self.hidden_dim,
            num_layers=int(num_layers),
            kernel_size=int(conv_kernel_size),
            dropout=float(dropout),
        )
        self.pool_norm = nn.LayerNorm(self.hidden_dim)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.latent_dim)
        )
        self.quantizer = TokenVectorQuantizer(
            self.codebook_size, self.latent_dim, beta=beta
        )
        self.from_latent = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decode_queries = nn.Parameter(
            torch.randn(self.sample_points, self.hidden_dim) * 0.02
        )
        self.decode_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.decode_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 4, self.traj_dim),
        )

    def input_stem_modules(self) -> list[nn.Module]:
        """Return per-frame encoders and contact fusion modules."""
        modules: list[nn.Module] = [
            self.motion_encoder,
            self.hand_force_encoder,
            self.hand_contact_fusion,
            self.motion_contact_fusion,
            self.missing_modality,
        ]
        if self.hand_tactile_encoder is not None:
            modules.append(self.hand_tactile_encoder)
        return modules

    @staticmethod
    def _vector_modality_encoder(input_dim: int, feature_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim)
        )

    def _input_features(
        self,
        sequence: torch.Tensor,
        masked_modalities: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode each frame before temporal processing."""
        self._validate_sequence(sequence)
        return self._contact_features(sequence, masked_modalities)

    def _contact_features(
        self, sequence: torch.Tensor, masked_modalities: Dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """Encode hand-local contact before mixing it with motion.

        Force and tactile have the same left/right partition.  The encoders and
        fusion block are shared between hands, while the subsequent motion query
        can still compare their different contact states.
        """
        bsz, steps = sequence.shape[:2]
        offset = 0
        pos = sequence[..., offset : offset + self.pos_dim]
        offset += self.pos_dim
        hand_force_dim = self.force_dim // 2
        force = sequence[..., offset : offset + self.force_dim]
        offset += self.force_dim
        motion = pos
        motion = self.motion_encoder(motion)
        left_force = self.hand_force_encoder(force[..., :hand_force_dim])
        right_force = self.hand_force_encoder(force[..., hand_force_dim:])

        def replace_masked(feature: torch.Tensor, name: str) -> torch.Tensor:
            if masked_modalities is None or name not in masked_modalities:
                return feature
            masked = masked_modalities[name].to(
                device=sequence.device, dtype=torch.bool
            )
            if masked.shape != (bsz,):
                raise ValueError(
                    f"masked modality '{name}' expected [{bsz}], got {tuple(masked.shape)}"
                )
            return torch.where(
                masked.view(bsz, 1, 1),
                self.missing_modality[name].view(1, 1, -1),
                feature,
            )

        left_force = replace_masked(left_force, "left_force")
        right_force = replace_masked(right_force, "right_force")
        left_tactile = right_tactile = None
        if self.tactile_dim:
            tactile = sequence[..., offset : offset + self.tactile_dim]
            tactile = tactile.reshape(
                bsz * steps,
                self.tactile_channels,
                self.tactile_input_h,
                self.tactile_input_w,
            )
            hand_channels = self.tactile_channels // 2
            left_tactile = self.hand_tactile_encoder(
                tactile[:, :hand_channels]
            ).reshape(bsz, steps, -1)
            right_tactile = self.hand_tactile_encoder(
                tactile[:, hand_channels:]
            ).reshape(bsz, steps, -1)
            left_tactile = replace_masked(left_tactile, "left_tactile")
            right_tactile = replace_masked(right_tactile, "right_tactile")
        left_contact = self.hand_contact_fusion(left_force, left_tactile)
        right_contact = self.hand_contact_fusion(right_force, right_tactile)
        left_contact = left_contact + self.hand_embedding.weight[0].view(1, 1, -1)
        right_contact = right_contact + self.hand_embedding.weight[1].view(1, 1, -1)
        return self.motion_contact_fusion(motion, left_contact, right_contact)

    def pack_sequence(
        self,
        pos: torch.Tensor,
        force: torch.Tensor,
        tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos.ndim != 3 or force.ndim != 3:
            raise ValueError(
                f"pack_sequence expected pos/force [B,T,D], got {tuple(pos.shape)} / {tuple(force.shape)}"
            )
        if pos.shape[:2] != force.shape[:2]:
            raise ValueError(
                f"pos/force time mismatch: {tuple(pos.shape)} vs {tuple(force.shape)}"
            )
        parts = []
        if self.pos_dim > 0:
            if pos.shape[-1] != self.pos_dim:
                raise ValueError(f"pos dim={pos.shape[-1]}, expected {self.pos_dim}")
            parts.append(pos)
        if self.force_dim > 0:
            if force.shape[-1] != self.force_dim:
                raise ValueError(
                    f"force dim={force.shape[-1]}, expected {self.force_dim}"
                )
            parts.append(force)
        if self.tactile_dim > 0:
            if tactile is None:
                raise KeyError("tactile is required when tactile_channels > 0")
            parts.append(self._tactile_descriptor(tactile, expected_time=pos.shape[:2]))
        if not parts:
            raise ValueError("At least one reference modality must be enabled.")
        return torch.cat(parts, dim=-1)

    def _tactile_descriptor(
        self, tactile: torch.Tensor, expected_time: tuple[int, int]
    ) -> torch.Tensor:
        if tactile.ndim == 3:
            expected_dim = self.tactile_dim
            if tactile.shape[:2] != expected_time or tactile.shape[-1] != expected_dim:
                raise ValueError(
                    f"tactile descriptor expected [{expected_time[0]},{expected_time[1]},{expected_dim}], got {tuple(tactile.shape)}"
                )
            return tactile
        if tactile.ndim != 5 or tactile.shape[:2] != expected_time:
            raise ValueError(
                f"tactile must be [B,T,H,W,C] aligned with pos/force, got {tuple(tactile.shape)}"
            )
        if tactile.shape[-1] != self.tactile_channels:
            raise ValueError(
                f"tactile channels={tactile.shape[-1]}, expected {self.tactile_channels}"
            )
        bsz, steps = tactile.shape[:2]
        x = (
            tactile.reshape(bsz * steps, *tactile.shape[2:])
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return x.flatten(1).reshape(bsz, steps, self.tactile_dim)

    def _validate_sequence(self, sequence: torch.Tensor) -> None:
        if sequence.ndim != 3 or sequence.shape[-1] != self.traj_dim:
            raise ValueError(
                f"ContactAutoencoder expected sequence [B,T,{self.traj_dim}], got {tuple(sequence.shape)}"
            )

    def _normalize_mask(
        self, mask: torch.Tensor | None, batch: int, time: int, device: torch.device
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch, time, dtype=torch.bool, device=device)
        if mask.shape != (batch, time):
            raise ValueError(f"mask expected [{batch},{time}], got {tuple(mask.shape)}")
        mask = mask.to(device=device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid timestep.")
        return mask

    def _sampled_normalized_phase(
        self, sample_indices: torch.Tensor, mask: torch.Tensor | None, source_time: int
    ) -> torch.Tensor:
        """Map selected source-frame indices to [0,1] within each valid episode."""
        if sample_indices.ndim != 2:
            raise ValueError(
                f"sample_indices must be [B,T], got {tuple(sample_indices.shape)}"
            )
        valid = self._normalize_mask(
            mask, sample_indices.shape[0], source_time, sample_indices.device
        )
        positions = torch.arange(source_time, device=sample_indices.device).view(1, -1)
        lo = torch.where(valid, positions, source_time).min(dim=1).values
        hi = torch.where(valid, positions, -1).max(dim=1).values
        denominator = (hi - lo).clamp_min(1).to(sample_indices.dtype)
        return (sample_indices - lo.unsqueeze(1)).to(
            torch.float32
        ) / denominator.unsqueeze(1)

    def _fps_features(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        start = self.pos_dim
        features = sequence[..., start : start + self.force_dim]
        features = features
        return self._smooth_fps_features(features, mask)

    def _smooth_fps_features(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Optionally smooth features used for index selection, not sampled raw values."""
        features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        kernel = self.fps_smoothing_kernel
        if kernel == 1:
            return features
        x = features.transpose(1, 2)
        valid = mask.to(dtype=features.dtype).unsqueeze(1)
        weight = features.new_ones(1, 1, kernel)
        numerator = F.conv1d(
            x * valid,
            weight.expand(x.shape[1], 1, -1),
            padding=kernel // 2,
            groups=x.shape[1],
        )
        denominator = F.conv1d(valid, weight, padding=kernel // 2).clamp_min(1.0)
        smoothed = (numerator / denominator).transpose(1, 2)
        return smoothed.masked_fill(~mask.unsqueeze(-1), 0.0)

    def _hybrid_indices(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Select uniform temporal anchors, then fill with complementary FPS points."""
        batch = features.shape[0]
        out = torch.empty(
            batch, self.sample_points, dtype=torch.long, device=features.device
        )
        requested_uniform = int(round(self.sample_points * self.uniform_ratio))
        for bidx in range(batch):
            valid_idx = torch.nonzero(mask[bidx], as_tuple=False).squeeze(1)
            valid_feat = features[bidx, valid_idx]
            n_valid = int(valid_idx.numel())
            target_unique = min(self.sample_points, n_valid)
            n_uniform = min(requested_uniform, target_unique)
            if n_uniform > 0:
                uniform_local = (
                    torch.linspace(
                        0, n_valid - 1, steps=n_uniform, device=features.device
                    )
                    .round()
                    .long()
                    .unique(sorted=True)
                )
            else:
                uniform_local = torch.empty(0, dtype=torch.long, device=features.device)
            selected = torch.zeros(n_valid, dtype=torch.bool, device=features.device)
            selected[uniform_local] = True
            selected_local = uniform_local.tolist()
            if selected_local:
                seed_feat = valid_feat[uniform_local]
                min_dist = torch.cdist(valid_feat, seed_feat).pow(2).min(dim=1).values
            else:
                centroid = valid_feat.mean(dim=0, keepdim=True)
                first = torch.sum((valid_feat - centroid) ** 2, dim=-1).argmax()
                selected[first] = True
                selected_local.append(int(first))
                min_dist = torch.sum((valid_feat - valid_feat[first]) ** 2, dim=-1)
            while len(selected_local) < target_unique:
                candidate_dist = min_dist.masked_fill(selected, -1.0)
                current = candidate_dist.argmax()
                selected[current] = True
                selected_local.append(int(current))
                dist = torch.sum((valid_feat - valid_feat[current]) ** 2, dim=-1)
                min_dist = torch.minimum(min_dist, dist)
            selected_tensor = torch.tensor(
                selected_local, dtype=torch.long, device=features.device
            )
            selected_idx = valid_idx[selected_tensor]
            if selected_idx.numel() < self.sample_points:
                fill_count = self.sample_points - int(selected_idx.numel())
                fill_local = (
                    torch.linspace(
                        0, n_valid - 1, steps=fill_count, device=features.device
                    )
                    .round()
                    .long()
                )
                selected_idx = torch.cat([selected_idx, valid_idx[fill_local]], dim=0)
            out[bidx] = torch.sort(selected_idx).values
        return out

    def sample_sequence(
        self, sequence: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_sequence(sequence)
        bsz, steps = sequence.shape[:2]
        mask = self._normalize_mask(mask, bsz, steps, sequence.device)
        features = self._fps_features(sequence, mask)
        indices = self._hybrid_indices(features, mask)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
        sampled = torch.gather(sequence, dim=1, index=gather_idx)
        return (sampled, indices)

    def _pool_encoded(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), self.num_tokens).transpose(
            1, 2
        )
        attn = h.new_zeros(h.shape[0], self.num_tokens, h.shape[1])
        for token_idx in range(self.num_tokens):
            start = token_idx * h.shape[1] // self.num_tokens
            end = (token_idx + 1) * h.shape[1] // self.num_tokens
            end = max(end, start + 1)
            attn[:, token_idx, start:end] = 1.0 / float(end - start)
        return (self.pool_norm(pooled), attn)

    def encode_sampled(
        self,
        sampled: torch.Tensor,
        masked_modalities: Dict[str, torch.Tensor] | None = None,
        temporal_phase: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        token_count = self.sample_points
        if sampled.ndim != 3 or sampled.shape[1:] != (token_count, self.traj_dim):
            raise ValueError(
                f"encode_sampled expected [B,{token_count},{self.traj_dim}], got {tuple(sampled.shape)}"
            )
        if temporal_phase is None or temporal_phase.shape != sampled.shape[:2]:
            shape = None if temporal_phase is None else tuple(temporal_phase.shape)
            raise ValueError(
                f"normalized_phase encoding requires temporal_phase [B,T] matching sampled, got {shape} for sampled={tuple(sampled.shape)}"
            )
        pos = _sinusoidal_coordinate_encoding(
            temporal_phase.to(device=sampled.device, dtype=sampled.dtype)
            * float(max(token_count - 1, 1)),
            self.hidden_dim,
        )
        h = self._input_features(sampled, masked_modalities) + pos
        h = self.temporal_encoder(h)
        pooled, attn = self._pool_encoded(h)
        z_e = self.to_latent(pooled)
        q = self.quantizer(z_e)
        q["z_e"] = z_e
        q["pool_attn"] = attn
        return q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        if z_q.ndim != 3 or z_q.shape[1:] != (self.num_tokens, self.latent_dim):
            raise ValueError(
                f"decode expected [B,{self.num_tokens},{self.latent_dim}], got {tuple(z_q.shape)}"
            )
        context = self.from_latent(z_q)
        pos = _sinusoidal_positional_encoding(
            self.sample_points, self.hidden_dim, device=z_q.device, dtype=z_q.dtype
        )
        queries = self.decode_queries.unsqueeze(0).expand(
            z_q.shape[0], -1, -1
        ) + pos.unsqueeze(0)
        decoded, _ = self.decode_attn(
            query=queries, key=context, value=context, need_weights=False
        )
        return self.decode_mlp(decoded + queries)

    def _reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:

        def mse(pred: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            if mask is None:
                return F.mse_loss(pred, value)
            weight = mask.unsqueeze(-1).to(pred.dtype)
            return ((pred - value).square() * weight).sum() / (
                weight.sum() * pred.shape[-1]
            ).clamp_min(1)

        losses = {}
        offset = 0
        weighted = []
        if self.pos_dim > 0:
            pos_loss = mse(
                recon[..., offset : offset + self.pos_dim],
                target[..., offset : offset + self.pos_dim],
            )
            losses["recon_pos_loss"] = pos_loss.detach()
            weighted.append(pos_loss * self.recon_pos_weight)
            offset += self.pos_dim
        if self.force_dim > 0:
            force_loss = mse(
                recon[..., offset : offset + self.force_dim],
                target[..., offset : offset + self.force_dim],
            )
            losses["recon_force_loss"] = force_loss.detach()
            weighted.append(force_loss * self.recon_force_weight)
            offset += self.force_dim
        if self.tactile_dim > 0:
            tac_pred = recon[..., offset : offset + self.tactile_dim]
            tac_target = target[..., offset : offset + self.tactile_dim]
            tactile_loss = mse(tac_pred, tac_target)
            losses["recon_tactile_loss"] = tactile_loss.detach()
            weighted.append(tactile_loss * self.recon_tactile_weight)
        recon_loss = torch.stack(weighted).sum()
        losses["recon_loss"] = recon_loss.detach()
        losses["_recon_loss"] = recon_loss
        return losses

    def _sample_masked_modalities(
        self, batch: int, device: torch.device, probability: float
    ) -> Dict[str, torch.Tensor] | None:
        """Mask one contact stream per selected example, never an entire hand."""
        if probability <= 0.0:
            return None
        names = ["left_force", "right_force"]
        if self.tactile_dim:
            names.extend(["left_tactile", "right_tactile"])
        choose = torch.rand(batch, device=device) < probability
        selected = torch.randint(len(names), (batch,), device=device)
        return {name: choose & (selected == index) for index, name in enumerate(names)}

    def _masked_modality_reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None,
        masked_modalities: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if not masked_modalities:
            return None
        offsets: dict[str, slice] = {}
        force_start = self.pos_dim
        hand_force_dim = self.force_dim // 2
        offsets["left_force"] = slice(force_start, force_start + hand_force_dim)
        offsets["right_force"] = slice(
            force_start + hand_force_dim, force_start + self.force_dim
        )
        if self.tactile_dim:
            tactile_start = self.pos_dim + self.force_dim
            hand_tactile_dim = self.tactile_dim // 2
            offsets["left_tactile"] = slice(
                tactile_start, tactile_start + hand_tactile_dim
            )
            offsets["right_tactile"] = slice(
                tactile_start + hand_tactile_dim, tactile_start + self.tactile_dim
            )
        losses = []
        for name, masked in masked_modalities.items():
            if not masked.any():
                continue
            error = (
                (recon[..., offsets[name]] - target[..., offsets[name]])
                .square()
                .mean(dim=-1)
            )
            if valid_mask is not None:
                weight = valid_mask.to(error.dtype)
                numerator = (error * weight).reshape(error.shape[0], -1).sum(dim=1)
                denominator = (
                    weight.reshape(weight.shape[0], -1).sum(dim=1).clamp_min(1.0)
                )
                per_example = numerator / denominator
            else:
                per_example = error.reshape(error.shape[0], -1).mean(dim=1)
            losses.append(per_example[masked].mean())
        return torch.stack(losses).mean() if losses else None

    def forward(
        self,
        sequence: torch.Tensor | None = None,
        *,
        pos: torch.Tensor | None = None,
        force: torch.Tensor | None = None,
        tactile: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        mask_modalities: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        if sequence is None:
            if pos is None or force is None:
                raise KeyError("Provide either sequence or both pos and force.")
            sequence = self.pack_sequence(pos=pos, force=force, tactile=tactile)
        mask_probability = (
            (self.masked_modality_probability if self.training else 0.0)
            if mask_modalities is None
            else 1.0 if mask_modalities else 0.0
        )
        masked_modalities = self._sample_masked_modalities(
            sequence.shape[0], sequence.device, mask_probability
        )
        sampled, sample_indices = self.sample_sequence(sequence, mask=mask)
        phase = self._sampled_normalized_phase(sample_indices, mask, sequence.shape[1])
        q = self.encode_sampled(sampled, masked_modalities, phase)
        recon = self.decode(q["z_q"])
        recon_mask = None
        recon_losses = self._reconstruction_loss(recon, sampled, recon_mask)
        recon_loss = recon_losses.pop("_recon_loss")
        loss = recon_loss + q["vq_loss"]
        masked_recon_loss = self._masked_modality_reconstruction_loss(
            recon, sampled, recon_mask, masked_modalities
        )
        out = {
            "loss": loss,
            "recon": recon,
            "sampled": sampled,
            "sample_indices": sample_indices,
            "z_q": q["z_q"],
            "z_q_hard": q["z_q_hard"],
            "z_e": q["z_e"],
            "indices": q["indices"],
            "vq_loss": q["vq_loss"].detach(),
            "commitment_loss": q["commitment_loss"],
            "codebook_loss": q["codebook_loss"],
            "pool_attn": q["pool_attn"],
            **recon_losses,
        }
        out["masked_modality_recon_loss"] = (
            masked_recon_loss.detach()
            if masked_recon_loss is not None
            else recon_loss.detach().new_zeros(())
        )
        return out

    @torch.no_grad()
    def encode_prompt(
        self,
        sequence: torch.Tensor | None = None,
        *,
        pos: torch.Tensor | None = None,
        force: torch.Tensor | None = None,
        tactile: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        hard: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if sequence is None:
            if pos is None or force is None:
                raise KeyError("Provide either sequence or both pos and force.")
            sequence = self.pack_sequence(pos=pos, force=force, tactile=tactile)
        sampled, sample_indices = self.sample_sequence(sequence, mask=mask)
        phase = self._sampled_normalized_phase(sample_indices, mask, sequence.shape[1])
        q = self.encode_sampled(sampled, temporal_phase=phase)
        return {
            "prompt": q["z_q_hard"] if hard else q["z_q"],
            "continuous_prompt": q["z_e"],
            "indices": q["indices"],
            "sample_indices": sample_indices,
            "pool_attn": q["pool_attn"],
            "padding_mask": None,
        }

    def encode_continuous_sampled(
        self,
        sampled: torch.Tensor,
        temporal_phase: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode preselected FPT samples with gradients, without quantization or reconstruction."""
        token_count = self.sample_points
        if sampled.ndim != 3 or sampled.shape[1:] != (token_count, self.traj_dim):
            raise ValueError(
                f"encode_continuous_sampled expected [B,{token_count},{self.traj_dim}], got {tuple(sampled.shape)}"
            )
        if temporal_phase is None or temporal_phase.shape != sampled.shape[:2]:
            shape = None if temporal_phase is None else tuple(temporal_phase.shape)
            raise ValueError(
                f"normalized_phase encoding requires temporal_phase [B,T] matching sampled, got {shape} for sampled={tuple(sampled.shape)}"
            )
        pos = _sinusoidal_coordinate_encoding(
            temporal_phase.to(device=sampled.device, dtype=sampled.dtype)
            * float(max(token_count - 1, 1)),
            self.hidden_dim,
        )
        h = self._input_features(sampled) + pos
        h = self.temporal_encoder(h)
        pooled = self.pool_norm(
            F.adaptive_avg_pool1d(h.transpose(1, 2), self.num_tokens).transpose(1, 2)
        )
        return {"z_e": self.to_latent(pooled)}
