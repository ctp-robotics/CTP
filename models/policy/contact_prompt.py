from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.policy.tactile_encoder import SingleHandSpatialEncoder


@dataclass(frozen=True)
class ContactPromptTrace:
    """Small diagnostics bundle returned by the contact-prompt aligner."""

    attention: torch.Tensor
    modality_ids: torch.Tensor


@dataclass(frozen=True)
class ContactPromptOutput:
    """Prompt-conditioned representation of the current observation."""

    global_cond: torch.Tensor
    prompt_tokens: torch.Tensor
    trace: ContactPromptTrace | None = None


def summarize_contact_trace(trace: ContactPromptTrace) -> dict[str, torch.Tensor]:
    """Reduce cross-attention to a modality-balanced soft prompt progress."""
    attention = trace.attention.mean(dim=1)
    positions = torch.linspace(
        0.0, 1.0, attention.shape[-1], device=attention.device, dtype=attention.dtype
    )
    distributions = []
    present = []
    for modality in (0, 1, 2, 3):
        mask = trace.modality_ids == modality
        count = mask.sum(dim=1, keepdim=True)
        distribution = (attention * mask.unsqueeze(-1).to(attention.dtype)).sum(
            dim=1
        ) / count.clamp_min(1).to(attention.dtype)
        distributions.append(distribution)
        present.append(count.squeeze(1) > 0)
    modality_attention = torch.stack(distributions, dim=1)
    modality_present = torch.stack(present, dim=1)
    weights = modality_present.to(attention.dtype)
    prompt_attention = (modality_attention * weights.unsqueeze(-1)).sum(dim=1)
    prompt_attention = prompt_attention / weights.sum(dim=1, keepdim=True).clamp_min(1)
    alpha = (prompt_attention * positions).sum(dim=-1)
    entropy = -(prompt_attention.clamp_min(1e-08).log() * prompt_attention).sum(dim=-1)
    if attention.shape[-1] > 1:
        entropy = entropy / torch.log(attention.new_tensor(float(attention.shape[-1])))
    modality_alpha = (modality_attention * positions).sum(dim=-1)
    modality_alpha = modality_alpha.masked_fill(~modality_present, torch.nan)
    return {
        "contact_alpha": alpha,
        "contact_peak": prompt_attention.argmax(dim=-1),
        "contact_entropy": entropy,
        "contact_attention": prompt_attention,
        "contact_modality_alpha": modality_alpha,
    }


def attention_weights(
    module: nn.MultiheadAttention, query: torch.Tensor, memory: torch.Tensor
) -> torch.Tensor:
    """Compute per-head weights without changing the attention output kernel."""
    dim = module.embed_dim
    q = F.linear(query, module.in_proj_weight[:dim], module.in_proj_bias[:dim])
    k = F.linear(
        memory, module.in_proj_weight[dim : 2 * dim], module.in_proj_bias[dim : 2 * dim]
    )
    head_dim = dim // module.num_heads
    q = q.view(q.shape[0], q.shape[1], module.num_heads, head_dim).transpose(1, 2)
    k = k.view(k.shape[0], k.shape[1], module.num_heads, head_dim).transpose(1, 2)
    return torch.softmax(q @ k.transpose(-2, -1) * head_dim ** (-0.5), dim=-1)


class ContactPromptDecoderLayer(nn.Module):
    """Small pre-norm decoder block that exposes prompt cross-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.self_norm(query)
        query = query + self.self_attn(x, x, x, need_weights=False)[0]
        cross_query = self.cross_norm(query)
        cross_memory = self.memory_norm(memory)
        attended = self.cross_attn(
            cross_query,
            cross_memory,
            cross_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        weights = (
            attention_weights(self.cross_attn, cross_query, cross_memory)
            if need_weights
            else None
        )
        query = query + attended
        query = query + self.ff(self.ff_norm(query))
        return (query, weights)


class ContactPromptAligner(nn.Module):
    """Cross-attention from online visual, state, force, and tactile observations to reference tokens."""

    MOD_STATE = 0
    MOD_FORCE = 1
    MOD_TACTILE = 2
    MOD_VISUAL = 3

    def __init__(
        self,
        *,
        state_dim: int,
        force_dim: int,
        cond_dim: int,
        prompt_tokens: int = 16,
        prompt_dim: int = 128,
        token_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        state_steps: int = 1,
        force_steps: int = 4,
        tactile_steps: int = 1,
        visual_dim: int = 256,
        tactile_hand_dim: int = 48,
        tactile_hidden_channels: int = 128,
        tactile_mid_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.force_dim = int(force_dim)
        self.cond_dim = int(cond_dim)
        self.prompt_tokens = int(prompt_tokens)
        self.prompt_dim = int(prompt_dim)
        self.token_dim = int(token_dim)
        self.force_steps = int(force_steps)
        self.state_steps = int(state_steps)
        self.tactile_steps = int(tactile_steps)
        if self.force_steps < 1:
            raise ValueError(f"force_steps must be >=1 got {self.force_steps}")
        if self.tactile_steps < 1:
            raise ValueError(f"tactile_steps must be >=1 got {self.tactile_steps}")
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >=1, got {num_layers}")
        if self.token_dim % int(num_heads) != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by num_heads={num_heads}"
            )
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(self.prompt_dim), nn.Linear(self.prompt_dim, self.token_dim)
        )
        self.prompt_pos = nn.Parameter(
            torch.zeros(1, self.prompt_tokens, self.token_dim)
        )
        self.state_token = nn.Sequential(
            nn.LayerNorm(self.state_dim), nn.Linear(self.state_dim, self.token_dim)
        )
        self.state_pos = nn.Parameter(torch.zeros(1, self.state_steps, self.token_dim))
        self.force_token = nn.Sequential(
            nn.LayerNorm(self.force_dim), nn.Linear(self.force_dim, self.token_dim)
        )
        self.force_pos = nn.Parameter(torch.zeros(1, self.force_steps, self.token_dim))
        self.tactile_hand_encoder = SingleHandSpatialEncoder(
            in_channels=3,
            hand_dim=int(tactile_hand_dim),
            hidden=int(tactile_hidden_channels),
            mid=int(tactile_mid_channels),
            dropout=dropout,
        )
        tactile_flat_dim = 9 * 5 * int(tactile_hand_dim)
        self.tactile_token = nn.Sequential(
            nn.LayerNorm(tactile_flat_dim), nn.Linear(tactile_flat_dim, self.token_dim)
        )
        self.tactile_time_pos = nn.Parameter(
            torch.zeros(1, self.tactile_steps, 1, self.token_dim)
        )
        self.tactile_hand_pos = nn.Parameter(torch.zeros(1, 1, 2, self.token_dim))
        self.visual_token = nn.Sequential(
            nn.LayerNorm(int(visual_dim)), nn.Linear(int(visual_dim), self.token_dim)
        )
        self.modality_embedding = nn.Embedding(4, self.token_dim)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.prompt_norm = nn.LayerNorm(self.token_dim)
        self.decoder_layers = nn.ModuleList(
            [
                ContactPromptDecoderLayer(
                    self.token_dim, int(num_heads), float(dropout)
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.cond_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

    @staticmethod
    def _last_or_pad_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        t = x.shape[1]
        if t >= target_t:
            return x[:, -target_t:]
        pad = x[:, -1:].expand(-1, target_t - t, *x.shape[2:])
        return torch.cat([x, pad], dim=1)

    def _encode_prompt(
        self, prompt: torch.Tensor, prompt_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if prompt.ndim != 3 or prompt.shape[1:] != (
            self.prompt_tokens,
            self.prompt_dim,
        ):
            raise ValueError(
                f"force_vq_prompt expected [B,{self.prompt_tokens},{self.prompt_dim}], got {tuple(prompt.shape)}"
            )
        if self.prompt_proj is None:
            tokens = prompt.new_zeros(
                prompt.shape[0], self.prompt_tokens, self.token_dim
            )
        else:
            tokens = self.prompt_proj(prompt)
        tokens = tokens + self.prompt_pos.to(device=tokens.device, dtype=tokens.dtype)
        if prompt_padding_mask is not None:
            if tuple(prompt_padding_mask.shape) != tuple(prompt.shape[:2]):
                raise ValueError("force_vq_prompt padding mask shape mismatch")
            tokens = tokens.masked_fill(
                prompt_padding_mask.to(tokens.device).bool().unsqueeze(-1), 0
            )
        return tokens

    def _append_state_tokens(
        self, tokens: list[torch.Tensor], mods: list[torch.Tensor], state: torch.Tensor
    ) -> None:
        if state.ndim != 3 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state expected [B,T,{self.state_dim}], got {tuple(state.shape)}"
            )
        state = self._last_or_pad_time(state, self.state_steps)
        token = self.state_token(state)
        token = token + self.state_pos.to(device=state.device, dtype=token.dtype)
        state_mod = self.modality_embedding.weight[self.MOD_STATE].to(
            device=state.device, dtype=token.dtype
        )
        token = token + state_mod.view(1, 1, -1)
        tokens.append(token)
        mods.append(
            torch.full(
                (state.shape[0], self.state_steps),
                self.MOD_STATE,
                device=state.device,
                dtype=torch.long,
            )
        )

    def _append_force_tokens(
        self,
        tokens: list[torch.Tensor],
        mods: list[torch.Tensor],
        force: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        if force is None:
            raise KeyError("Contact conditioning requires obs['force'].")
        if force.ndim != 3 or force.shape[-1] != self.force_dim:
            raise ValueError(
                f"force expected [B,T,{self.force_dim}], got {tuple(force.shape)}"
            )
        force = self._last_or_pad_time(force, self.force_steps)
        token = self.force_token(force)
        token = token + self.force_pos.to(device=force.device, dtype=token.dtype)
        force_mod = self.modality_embedding.weight[self.MOD_FORCE].to(
            device=force.device, dtype=token.dtype
        )
        token = token + force_mod.view(1, 1, -1)
        tokens.append(token)
        mods.append(
            torch.full(
                (force.shape[0], self.force_steps),
                self.MOD_FORCE,
                device=device,
                dtype=torch.long,
            )
        )

    def _append_tactile_tokens(
        self,
        tokens: list[torch.Tensor],
        mods: list[torch.Tensor],
        tactile: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        if tactile is None:
            raise KeyError("Contact conditioning requires obs['tactile'].")
        if tactile.ndim != 5 or tactile.shape[-1] != 6:
            raise ValueError(
                f"tactile expected [B,T,H,W,6], got {tuple(tactile.shape)}"
            )
        tactile = self._last_or_pad_time(tactile, self.tactile_steps)
        bsz, steps = tactile.shape[:2]
        x = tactile.reshape(bsz * steps, *tactile.shape[2:]).float()
        left = self.tactile_hand_encoder(x[..., :3]).reshape(bsz, steps, -1)
        right = self.tactile_hand_encoder(x[..., 3:]).reshape(bsz, steps, -1)
        hand = torch.stack([left, right], dim=2)
        token = self.tactile_token(hand)
        token = token + self.tactile_time_pos.to(device=token.device, dtype=token.dtype)
        token = token + self.tactile_hand_pos.to(device=token.device, dtype=token.dtype)
        token = token.reshape(bsz, steps * 2, self.token_dim)
        tactile_mod = self.modality_embedding.weight[self.MOD_TACTILE].to(
            device=token.device, dtype=token.dtype
        )
        token = token + tactile_mod.view(1, 1, -1)
        tokens.append(token)
        mods.append(
            torch.full(
                (bsz, steps * 2), self.MOD_TACTILE, device=device, dtype=torch.long
            )
        )

    def _online_tokens(
        self, obs: dict[str, torch.Tensor], visual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "state" not in obs:
            raise KeyError("contact prompt conditioning requires obs['state'].")
        state = obs["state"]
        tokens: list[torch.Tensor] = []
        mods: list[torch.Tensor] = []
        self._append_state_tokens(tokens, mods, state)
        self._append_force_tokens(tokens, mods, obs.get("force"), state.device)
        self._append_tactile_tokens(tokens, mods, obs.get("tactile"), state.device)
        if visual is None:
            raise ValueError("Contact conditioning requires a visual feature.")
        token = self.visual_token(visual).unsqueeze(1)
        token = token + self.modality_embedding.weight[self.MOD_VISUAL].to(
            device=token.device, dtype=token.dtype
        ).view(1, 1, -1)
        tokens.append(token)
        mods.append(
            torch.full(
                (visual.shape[0], 1),
                self.MOD_VISUAL,
                device=state.device,
                dtype=torch.long,
            )
        )
        return (torch.cat(tokens, dim=1), torch.cat(mods, dim=1))

    def forward(
        self,
        *,
        prompt: torch.Tensor,
        prompt_padding_mask: torch.Tensor | None = None,
        obs: dict[str, torch.Tensor],
        visual: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> ContactPromptOutput:
        prompt_tokens = self.prompt_norm(
            self._encode_prompt(prompt, prompt_padding_mask)
        )
        query_tokens, modality_ids = self._online_tokens(obs, visual)
        query_tokens = self.query_norm(query_tokens)
        attn = None
        attended = query_tokens
        for layer_idx, layer in enumerate(self.decoder_layers):
            attended, layer_attn = layer(
                attended,
                prompt_tokens,
                memory_padding_mask=prompt_padding_mask,
                need_weights=return_trace and layer_idx == len(self.decoder_layers) - 1,
            )
            if layer_attn is not None:
                attn = layer_attn
        pooled_parts = []
        for mod in (self.MOD_STATE, self.MOD_FORCE, self.MOD_TACTILE, self.MOD_VISUAL):
            mask = modality_ids == mod
            if bool(mask.any()):
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(attended.dtype)
                pooled = (attended * mask.unsqueeze(-1).to(attended.dtype)).sum(
                    dim=1
                ) / denom
                pooled_parts.append(pooled)
        pooled = torch.stack(pooled_parts, dim=1).mean(dim=1)
        cond = self.out(pooled)
        trace = None
        if return_trace:
            if attn is None:
                raise RuntimeError("Prompt decoder did not return attention weights.")
            trace = ContactPromptTrace(attention=attn, modality_ids=modality_ids)
        return ContactPromptOutput(
            global_cond=cond, prompt_tokens=prompt_tokens, trace=trace
        )
