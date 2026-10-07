from __future__ import annotations
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class DinoV2SmallEncoder(nn.Module):

    def __init__(
        self,
        out_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
    ):
        super().__init__()
        if timm is None:
            raise ImportError(
                "DINOv2 encoder requires timm. Install with: pip install timm"
            )
        self.freeze = bool(freeze)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, img_size=224
        )
        backbone_dim = getattr(self.backbone, "num_features", None)
        if backbone_dim is None:
            raise RuntimeError("Cannot infer DINOv2 output dim from timm model.")
        self.head = nn.Sequential(
            nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, out_dim), nn.SiLU()
        )
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = (x + 1.0) * 0.5
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    def extract_backbone_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Return the global CLS token from the DINO backbone."""
        return self.extract_backbone_tokens(x)[:, 0]

    @staticmethod
    def _as_tokens(feat: object) -> torch.Tensor:
        if isinstance(feat, dict):
            cls = feat.get("x_norm_clstoken")
            patches = feat.get("x_norm_patchtokens")
            if cls is not None and patches is not None:
                return torch.cat([cls.unsqueeze(1), patches], dim=1)
            feat = feat.get("x", feat.get("tokens"))
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if not isinstance(feat, torch.Tensor) or feat.ndim != 3:
            raise RuntimeError(
                "DINOv2 backbone must return [N, tokens, channels] from forward_features."
            )
        return feat

    def extract_backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self._imagenet_normalize(x)
        if self.freeze:
            with torch.no_grad():
                feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone.forward_features(x)
        return self._as_tokens(feat)

    def forward_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_backbone_feat(x)
        return self.forward_from_backbone_feat(feat)


class ObsEncoder(nn.Module):
    """Fuse [visual_feature, state] history into a single observation embedding."""

    def __init__(
        self,
        state_dim: int,
        out_dim: int,
        freeze_image_encoder: bool = True,
        image_pretrained: bool = True,
        dino_model_name: str = "vit_small_patch14_dinov2.lvd142m",
        image_feat_dim: int = 256,
        state_feat_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        obs_steps: int = 8,
    ):
        super().__init__()
        self.image_encoder = DinoV2SmallEncoder(
            out_dim=image_feat_dim,
            pretrained=image_pretrained,
            freeze=freeze_image_encoder,
            model_name=dino_model_name,
        )
        self.obs_steps = int(obs_steps)
        self.visual_feat_dim = int(image_feat_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(self.obs_steps * int(state_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_feat_dim),
            nn.SiLU(),
        )
        backbone_dim = int(self.image_encoder.backbone.num_features)
        self.view_embedding = nn.Embedding(8, backbone_dim)
        self.image_proj = nn.Linear(image_feat_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + int(state_feat_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(out_dim)),
        )

    def _pool_visual_tokens(
        self, tokens: torch.Tensor, state_feat: torch.Tensor
    ) -> torch.Tensor:
        """Pool [B,T,V,N,D] DINO tokens into one visual feature."""
        bsz, steps, views = tokens.shape[:3]
        if views > self.view_embedding.num_embeddings:
            raise ValueError(
                f"At most {self.view_embedding.num_embeddings} camera views are supported, got {views}"
            )
        tokens = tokens[:, :, :, :1]
        view_ids = torch.arange(views, device=tokens.device)
        tokens = tokens + self.view_embedding(view_ids).view(1, 1, views, 1, -1)
        tokens = tokens.reshape(bsz, steps * views * tokens.shape[3], tokens.shape[-1])
        visual_feat = tokens.mean(dim=1)
        return self.image_encoder.forward_from_backbone_feat(visual_feat)

    def encode_image(
        self, image: torch.Tensor, state_feat: torch.Tensor
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("Expected RGB image observations.")
        bsz, steps, views = image.shape[:3]
        tokens = self.image_encoder.extract_backbone_tokens(
            image.reshape(bsz * steps * views, *image.shape[3:])
        )
        tokens = tokens.reshape(bsz, steps, views, tokens.shape[1], tokens.shape[-1])
        return self._pool_visual_tokens(tokens, state_feat)

    def encode_cached_image(
        self, image_backbone_feat: torch.Tensor, state_feat: torch.Tensor
    ) -> torch.Tensor:
        """Pool cached per-view DINO CLS features."""
        if image_backbone_feat.ndim != 4:
            raise ValueError(
                f"Expected cached image features [B,T,V,D], got {tuple(image_backbone_feat.shape)}"
            )
        return self._pool_visual_tokens(image_backbone_feat.unsqueeze(3), state_feat)

    def forward(
        self,
        image: torch.Tensor | None,
        state: torch.Tensor,
        image_backbone_feat: torch.Tensor | None = None,
        return_visual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if state.shape[1] != self.obs_steps:
            raise ValueError(
                f"ObsEncoder expected T={self.obs_steps}, got {state.shape[1]}"
            )
        state_feat = self.state_proj(state.flatten(1, 2))
        if image_backbone_feat is None:
            visual_feat = self.encode_image(image=image, state_feat=state_feat)
        else:
            visual_feat = self.encode_cached_image(
                image_backbone_feat, state_feat=state_feat
            )
        img_feat = self.image_proj(visual_feat)
        obs_feat = self.fuse(torch.cat([img_feat, state_feat], dim=-1))
        if return_visual:
            return (obs_feat, visual_feat)
        return obs_feat
