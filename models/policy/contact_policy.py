from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.policy.contact_prompt import ContactPromptAligner, summarize_contact_trace
from models.policy.contact_autoencoder import ContactAutoencoder
from models.policy.obs_encoder import ObsEncoder
from utils.action import relative_actions_to_absolute_tensor
from utils.normalizer import MultiFieldNormalizer


class OnlineActionConditionedForceHead(nn.Module):
    """Predict normalized force changes from online observations and planned actions."""

    def __init__(
        self, action_dim: int, state_dim: int, force_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.force_dim = int(force_dim)
        context_dim = 2 * (self.state_dim + self.force_dim)
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.SiLU()
        )
        self.input_proj = nn.Conv1d(action_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, force_dim, kernel_size=1),
        )

    def forward(
        self, action: torch.Tensor, state: torch.Tensor, force: torch.Tensor
    ) -> torch.Tensor:
        if action.ndim != 3:
            raise ValueError(f"action must be [B,T,D], got {tuple(action.shape)}")
        if state.ndim != 3 or force.ndim != 3:
            raise ValueError(
                f"online state/force must be [B,T,D], got state={tuple(state.shape)}, force={tuple(force.shape)}"
            )
        if state.shape[0] != action.shape[0] or force.shape[0] != action.shape[0]:
            raise ValueError("online state/force batch size must match action.")
        if state.shape[-1] != self.state_dim or force.shape[-1] != self.force_dim:
            raise ValueError(
                f"online state/force feature dimensions must match the head: expected ({self.state_dim}, {self.force_dim}), got ({state.shape[-1]}, {force.shape[-1]})."
            )
        context = torch.cat(
            [
                state[:, -1],
                state[:, -1] - state[:, 0],
                force[:, -1],
                force[:, -1] - force[:, 0],
            ],
            dim=-1,
        )
        context = self.context(context).unsqueeze(1).expand(-1, action.shape[1], -1)
        x = torch.cat([action, context], dim=-1).transpose(1, 2)
        return self.temporal(self.input_proj(x)).transpose(1, 2)


class ContactPolicy(nn.Module):
    """Continuous contact-trajectory conditioning for a flow-matching action policy."""

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ContactPolicy":
        policy = cfg["model"]["policy"]
        encoder = (policy.get("mode_prompt") or {}).get("force_vq") or {}
        contact = policy["contact_prompt"]
        tactile = contact["tactile"]
        force = policy["force_consistency"]
        return cls(
            action_dim=int(policy.get("action_dim", 10)),
            force_dim=int(cfg["model"].get("force_vq", {}).get("force_dim", 12)),
            state_dim=int(policy.get("state_dim", 10)),
            cond_dim=int(policy.get("cond_dim", 256)),
            curr_steps=int(cfg["data"]["window_size"]),
            action_horizon=int(cfg["data"]["action_window_size"]),
            n_action_steps=int(policy.get("n_action_steps", 32)),
            action_representation=cfg["data"]["action_representation"],
            down_dims=policy.get("down_dims", [256, 512, 1024]),
            diffusion_step_embed_dim=int(policy.get("diffusion_step_embed_dim", 256)),
            kernel_size=int(policy.get("kernel_size", 5)),
            n_groups=int(policy.get("n_groups", 8)),
            image_pretrained=bool(policy.get("image_pretrained", True)),
            freeze_image_encoder=bool(policy.get("freeze_image_encoder", True)),
            dino_model_name=policy.get(
                "dino_model_name", "vit_small_patch14_dinov2.lvd142m"
            ),
            prompt_tokens=int(encoder.get("num_tokens", 16)),
            prompt_dim=int(encoder.get("latent_dim", 128)),
            token_dim=int(contact.get("token_dim", 128)),
            num_heads=int(contact.get("num_heads", 4)),
            num_layers=int(contact.get("num_layers", 3)),
            tactile_hand_dim=int(tactile.get("hand_dim", 16)),
            tactile_hidden_channels=int(tactile.get("hidden_channels", 32)),
            tactile_mid_channels=int(tactile.get("mid_channels", 32)),
            force_indices=force["force_indices"],
            force_horizon=int(force["horizon"]),
            force_hidden_dim=int(force.get("hidden_dim", 256)),
            force_loss_weight=float(force.get("force_loss_weight", 0.1)),
            num_inference_steps=int(
                policy.get("inference", {}).get("num_inference_steps", 16)
            ),
            hidden_dim=int(policy.get("hidden_dim", 512)),
            dropout=float(policy.get("dropout", 0.1)),
        )

    def __init__(
        self,
        *,
        action_dim=10,
        force_dim=12,
        state_dim=10,
        cond_dim=256,
        curr_steps=8,
        action_horizon=32,
        n_action_steps=32,
        action_representation="chunk_relative",
        down_dims=(256, 512, 1024),
        diffusion_step_embed_dim=256,
        kernel_size=5,
        n_groups=8,
        image_pretrained=True,
        freeze_image_encoder=True,
        dino_model_name="vit_small_patch14_dinov2.lvd142m",
        prompt_tokens=16,
        prompt_dim=128,
        token_dim=128,
        num_heads=4,
        num_layers=3,
        tactile_hand_dim=16,
        tactile_hidden_channels=32,
        tactile_mid_channels=32,
        force_indices=(6, 7, 8),
        force_horizon=8,
        force_hidden_dim=256,
        force_loss_weight=0.1,
        num_inference_steps=16,
        hidden_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.action_dim, self.force_dim, self.state_dim = (
            action_dim,
            force_dim,
            state_dim,
        )
        self.curr_steps, self.action_horizon, self.n_action_steps = (
            curr_steps,
            action_horizon,
            n_action_steps,
        )
        self.action_representation = action_representation
        self.force_prompt_tokens, self.force_prompt_dim = (prompt_tokens, prompt_dim)
        self.force_consistency_indices = tuple(force_indices)
        self.force_consistency_horizon = force_horizon
        self.force_consistency_loss_weight = force_loss_weight
        self.num_inference_steps = num_inference_steps
        self.trajectory_dim = action_dim
        if curr_steps < 1 or not 1 <= n_action_steps <= action_horizon:
            raise ValueError(
                "Require positive observation history and 1 <= n_action_steps <= action_horizon."
            )
        if action_representation not in {"absolute", "chunk_relative"}:
            raise ValueError(
                "action_representation must be absolute or chunk_relative."
            )
        if not 1 <= force_horizon <= action_horizon or force_loss_weight < 0:
            raise ValueError(
                "Require a force horizon within the action horizon and a nonnegative loss weight."
            )
        if (
            not force_indices
            or len(set(force_indices)) != len(force_indices)
            or any((i < 0 or i >= force_dim for i in force_indices))
        ):
            raise ValueError("force_indices must be distinct valid force channels.")
        self.obs_encoder = ObsEncoder(
            state_dim=state_dim,
            out_dim=cond_dim,
            obs_steps=curr_steps,
            freeze_image_encoder=freeze_image_encoder,
            image_pretrained=image_pretrained,
            dino_model_name=dino_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.contact_prompt = ContactPromptAligner(
            state_dim=state_dim,
            force_dim=force_dim,
            cond_dim=cond_dim,
            prompt_tokens=prompt_tokens,
            prompt_dim=prompt_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            state_steps=curr_steps,
            force_steps=curr_steps,
            tactile_steps=curr_steps,
            tactile_hand_dim=tactile_hand_dim,
            tactile_hidden_channels=tactile_hidden_channels,
            tactile_mid_channels=tactile_mid_channels,
            visual_dim=self.obs_encoder.visual_feat_dim,
            dropout=dropout,
        )
        self.velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=2 * cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=True,
        )
        self.force_consistency_head = OnlineActionConditionedForceHead(
            action_dim=action_dim,
            state_dim=state_dim,
            force_dim=len(force_indices),
            hidden_dim=force_hidden_dim,
        )
        self.trainable_force_vq = None
        self.trainable_force_vq_normalizer = MultiFieldNormalizer()
        self._trainable_fpt_pos = self._trainable_fpt_force = None
        self._trainable_fpt_tactile = self._trainable_fpt_phase = None
        self.normalizer = MultiFieldNormalizer()

    def set_normalizer(self, normalizer: MultiFieldNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_trainable_force_vq(
        self, vq: ContactAutoencoder, normalizer: MultiFieldNormalizer
    ) -> None:
        """Attach the continuous FPT encoder used directly by the policy loss."""
        if (
            vq.num_tokens != self.force_prompt_tokens
            or vq.latent_dim != self.force_prompt_dim
        ):
            raise ValueError(
                "Trainable contact encoder token/dimension does not match mode_prompt.force_vq config."
            )
        self.trainable_force_vq = vq
        self.trainable_force_vq_normalizer.load_state_dict(normalizer.state_dict())
        if vq.tactile_channels > 0 and "tactile" in self.trainable_force_vq_normalizer:
            scale_dim = int(self.trainable_force_vq_normalizer["tactile"].scale.numel())
            if scale_dim != int(vq.tactile_channels):
                raise ValueError(
                    f"Trainable contact encoder tactile normalizer must be per-channel (dim={vq.tactile_channels}), got dim={scale_dim}. Refit with last-axis channels; do not flatten spatial maps."
                )
        keep = [
            *vq.input_stem_modules(),
            vq.temporal_encoder,
            vq.pool_norm,
            vq.to_latent,
        ]
        for param in vq.parameters():
            param.requires_grad = False
        for module in keep:
            for param in module.parameters():
                param.requires_grad = True

    def set_trainable_force_vq_reference_table(
        self, pos, force, tactile=None, phase=None
    ) -> None:
        if self.trainable_force_vq is None:
            raise RuntimeError(
                "Attach the trainable contact encoder encoder before its reference table."
            )
        device = next(self.trainable_force_vq.parameters()).device
        self._trainable_fpt_pos = torch.as_tensor(
            pos, dtype=torch.float32, device=device
        )
        self._trainable_fpt_force = torch.as_tensor(
            force, dtype=torch.float32, device=device
        )
        self._trainable_fpt_tactile = (
            None
            if tactile is None
            else torch.as_tensor(tactile, dtype=torch.float32, device=device)
        )
        self._trainable_fpt_phase = (
            None
            if phase is None
            else torch.as_tensor(phase, dtype=torch.float32, device=device)
        )

    def _compact_trainable_force_vq_ids(
        self, ids: torch.Tensor, table_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = ids.reshape(-1).to(dtype=torch.long)
        used = torch.zeros(table_size, dtype=torch.bool, device=ids.device)
        used[ids] = True
        unique_ids = used.nonzero(as_tuple=False).view(-1)
        rank = torch.empty(table_size, dtype=torch.long, device=ids.device)
        rank[unique_ids] = torch.arange(unique_ids.numel(), device=ids.device)
        return (unique_ids, rank[ids])

    def _encode_trainable_force_vq_reference(
        self, source: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.trainable_force_vq is None:
            raise RuntimeError(
                "Trainable continuous contact encoder encoder is not attached."
            )
        inverse = None
        if self._trainable_fpt_pos is not None:
            episode_ids = source.get("force_vq_source_episode_index")
            if episode_ids is None:
                raise KeyError(
                    "trainable continuous contact encoder table requires force_vq_source_episode_index."
                )
            unique_ids, inverse = self._compact_trainable_force_vq_ids(
                episode_ids, int(self._trainable_fpt_pos.shape[0])
            )
            pos = self._trainable_fpt_pos[unique_ids]
            force = self._trainable_fpt_force[unique_ids]
            tactile = (
                None
                if self._trainable_fpt_tactile is None
                else self._trainable_fpt_tactile[unique_ids]
            )
            phase = (
                None
                if self._trainable_fpt_phase is None
                else self._trainable_fpt_phase[unique_ids]
            )
        else:
            required = ("force_vq_reference_pos", "force_vq_reference_force")
            if self.trainable_force_vq.tactile_dim > 0:
                required = (*required, "force_vq_reference_tactile")
            missing = [key for key in required if key not in source]
            if missing:
                raise KeyError(
                    f"trainable continuous contact encoder requires reference fields: {missing}"
                )
            pos = source["force_vq_reference_pos"]
            force = source["force_vq_reference_force"]
            tactile = (
                source.get("force_vq_reference_tactile")
                if self.trainable_force_vq.tactile_dim > 0
                else None
            )
            phase = source.get("force_vq_reference_phase")
            episode_ids = source.get("force_vq_source_episode_index")
            if episode_ids is not None and pos.shape[0] > 1:
                ids = episode_ids.reshape(-1).long()
                unique_ids, inverse = torch.unique(ids, return_inverse=True)
                if unique_ids.numel() < ids.numel():
                    first = torch.full(
                        (unique_ids.numel(),),
                        ids.numel(),
                        device=ids.device,
                        dtype=torch.long,
                    )
                    first.scatter_reduce_(
                        0,
                        inverse,
                        torch.arange(ids.numel(), device=ids.device),
                        reduce="amin",
                        include_self=True,
                    )
                    pos = pos[first]
                    force = force[first]
                    if tactile is not None:
                        tactile = tactile[first]
                    if phase is not None:
                        phase = phase[first]
                else:
                    inverse = None
        if "mode_pos" in self.trainable_force_vq_normalizer:
            pos = self.trainable_force_vq_normalizer["mode_pos"].normalize(pos)
        if "force" in self.trainable_force_vq_normalizer:
            force = self.trainable_force_vq_normalizer["force"].normalize(force)
        if tactile is not None and "tactile" in self.trainable_force_vq_normalizer:
            tactile = self.trainable_force_vq_normalizer["tactile"].normalize(tactile)
        sequence = self.trainable_force_vq.pack_sequence(
            pos=pos, force=force, tactile=tactile
        )
        if (
            self.trainable_force_vq.temporal_position_mode == "normalized_phase"
            and phase is None
        ):
            raise KeyError(
                "trainable continuous contact encoder with normalized_phase requires reference sample phases."
            )
        z_e = self.trainable_force_vq.encode_continuous_sampled(
            sequence, temporal_phase=phase
        )["z_e"]
        if inverse is not None:
            z_e = z_e[inverse]
        return z_e

    @staticmethod
    def _pad_or_trim_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        t = x.shape[1]
        if t == target_t:
            return x
        if t > target_t:
            return x[:, :target_t]
        pad = x[:, -1:].expand(-1, target_t - t, *x.shape[2:])
        return torch.cat([x, pad], dim=1)

    @staticmethod
    def _last_or_pad_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        t = x.shape[1]
        if t >= target_t:
            return x[:, -target_t:]
        pad = x[:, -1:].expand(-1, target_t - t, *x.shape[2:])
        return torch.cat([x, pad], dim=1)

    def _normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(self.normalizer.fields) == 0:
            return obs
        return self.normalizer.normalize_obs(obs)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if "action" in self.normalizer:
            return self.normalizer["action"].normalize(action)
        return action

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if "action" in self.normalizer:
            return self.normalizer["action"].unnormalize(action)
        return action

    def _normalize_force(self, force: torch.Tensor) -> torch.Tensor:
        if "force" in self.normalizer:
            return self.normalizer["force"].normalize(force)
        return force

    def _select_force_indices(
        self, force: torch.Tensor, indices: tuple[int, ...]
    ) -> torch.Tensor:
        if force.shape[-1] != self.force_dim:
            raise ValueError(
                f"Future force dim={force.shape[-1]}, expected force_dim={self.force_dim}."
            )
        index = torch.as_tensor(indices, device=force.device, dtype=torch.long)
        return self._normalize_force(force).index_select(-1, index)

    def _select_normalized_force_indices(self, force: torch.Tensor) -> torch.Tensor:
        """Select force-consistency dimensions from an already normalized force history."""
        if force.ndim != 3 or force.shape[-1] != self.force_dim:
            raise ValueError(
                f"online force expected [B,T,force_dim], got {tuple(force.shape)} with force_dim={self.force_dim}"
            )
        index = torch.as_tensor(
            self.force_consistency_indices, device=force.device, dtype=torch.long
        )
        return force.index_select(-1, index)

    def _online_force_delta(
        self, action: torch.Tensor, obs_norm: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted normalized force change and the current normalized base force."""
        if self.force_consistency_head is None:
            raise RuntimeError("Force prediction head is missing.")
        if "state" not in obs_norm or "force" not in obs_norm:
            raise KeyError(
                "force_consistency.conditioning='online' requires normalized online state and force observations."
            )
        selected_force = self._select_normalized_force_indices(obs_norm["force"])
        delta = self.force_consistency_head(action, obs_norm["state"], selected_force)
        base = selected_force[:, -1:]
        return (delta, base)

    def _unnormalize_force_indices(
        self, force: torch.Tensor, indices: tuple[int, ...]
    ) -> torch.Tensor:
        if "force" not in self.normalizer:
            return force
        field = self.normalizer["force"]
        index = torch.as_tensor(indices, device=force.device, dtype=torch.long)
        scale = field.scale.to(force.device, force.dtype).index_select(0, index)
        offset = field.offset.to(force.device, force.dtype).index_select(0, index)
        return (force - offset) / scale

    def _action_to_absolute(
        self, action: torch.Tensor, obs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.action_representation == "absolute":
            return action
        if "state" not in obs:
            raise KeyError(
                "chunk_relative action representation requires obs['state']."
            )
        state = obs["state"]
        if state.shape[-1] < self.action_dim:
            raise ValueError(
                f"chunk_relative requires state_dim >= action_dim, got state_dim={state.shape[-1]}, action_dim={self.action_dim}"
            )
        base = (
            state[:, -1, : self.action_dim].unsqueeze(1).expand(-1, action.shape[1], -1)
        )
        return relative_actions_to_absolute_tensor(action, base)

    def _encode_obs(
        self, obs_raw: Dict[str, torch.Tensor], obs_norm: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image = obs_norm.get("image")
        image_backbone_feat = obs_raw.get("image_backbone_feat")
        if image is None and image_backbone_feat is None:
            raise ValueError("Expected RGB image observations.")
        state_sel = self._last_or_pad_time(obs_norm["state"], self.curr_steps)
        image_sel = image
        encoded = self.obs_encoder(
            image=image_sel,
            state=state_sel,
            image_backbone_feat=image_backbone_feat,
            return_visual=True,
        )
        return encoded

    def _sample_actions(
        self,
        global_cond: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: int,
        solver: str,
    ) -> Dict[str, torch.Tensor]:
        bsz = global_cond.shape[0]
        trajectory = torch.randn(
            bsz,
            self.action_horizon,
            self.trajectory_dim,
            device=global_cond.device,
            dtype=global_cond.dtype,
        )
        times = torch.linspace(
            0.0,
            1.0,
            num_inference_steps + 1,
            device=global_cond.device,
            dtype=global_cond.dtype,
        )
        for i in range(num_inference_steps):
            t0 = times[i]
            t1 = times[i + 1]
            dt = t1 - t0
            velocity = self.velocity_net(
                trajectory, t0.expand(bsz), global_cond=global_cond
            )
            if solver.lower() == "heun" and i < num_inference_steps - 1:
                x_euler = trajectory + dt * velocity
                velocity_next = self.velocity_net(
                    x_euler, t1.expand(bsz), global_cond=global_cond
                )
                trajectory = trajectory + 0.5 * dt * (velocity + velocity_next)
            else:
                trajectory = trajectory + dt * velocity
        action_trajectory = trajectory[..., : self.action_dim]
        action_pred_model = self._unnormalize_action(action_trajectory)
        action_model = action_pred_model[:, : self.n_action_steps]
        action_pred = self._action_to_absolute(action_pred_model, obs)
        action = action_pred[:, : self.n_action_steps]
        out = {
            "action": action,
            "action_model": action_model,
            "action_pred": action_pred,
            "action_pred_model": action_pred_model,
        }
        if self.force_consistency_head is None:
            raise RuntimeError("Force prediction head is missing.")
        force_delta, force_base = self._online_force_delta(
            action_trajectory[:, : self.force_consistency_horizon],
            self._normalize_obs(obs),
        )
        force_trajectory = force_delta + force_base
        force_pred_model = self._unnormalize_force_indices(
            force_trajectory, self.force_consistency_indices
        )
        out["force_pred_model"] = force_pred_model
        out["force_pred"] = force_pred_model[:, : self.n_action_steps]
        out["force_pred_indices"] = torch.as_tensor(
            self.force_consistency_indices,
            device=force_pred_model.device,
            dtype=torch.long,
        )
        return out

    @torch.no_grad()
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: Optional[int] = None,
        solver: str = "euler",
        return_contact_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_inference(
            obs=obs,
            num_inference_steps=num_inference_steps,
            solver=solver,
            return_contact_diagnostics=return_contact_diagnostics,
        )

    def _read_contact(self, obs, source, visual, return_trace=False):
        prompt = source.get("force_vq_prompt")
        if prompt is None:
            prompt = self._encode_trainable_force_vq_reference(source)
        return self.contact_prompt(
            prompt=prompt,
            obs=obs,
            visual=visual,
            prompt_padding_mask=source.get("force_vq_padding_mask"),
            return_trace=return_trace,
        )

    def forward_train(self, batch):
        obs = self._normalize_obs(batch["obs"])
        obs_feat, visual = self._encode_obs(batch["obs"], obs)
        contact = self._read_contact(obs, batch, visual).global_cond
        condition = torch.cat([obs_feat, contact], dim=-1)
        target = self._normalize_action(batch["action"])
        if target.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                "Action target must match the configured horizon and action dimension."
            )
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        fraction = t[:, None, None]
        interpolated = (1 - fraction) * noise + fraction * target
        velocity = self.velocity_net(interpolated, t, global_cond=condition)
        flow_loss = F.mse_loss(velocity, target - noise)
        # Estimate the clean action chunk; the force loss also trains the action flow.
        action_estimate = (interpolated + (1 - fraction) * velocity)[
            :, : self.force_consistency_horizon
        ]
        predicted_delta, base_force = self._online_force_delta(action_estimate, obs)
        target_force = self._select_force_indices(
            batch["future_force"], self.force_consistency_indices
        )
        if target_force.shape[1] != self.force_consistency_horizon:
            raise ValueError(
                "Future-force target must match the configured force horizon."
            )
        force_loss = F.mse_loss(predicted_delta, target_force - base_force)
        return {
            "loss": flow_loss + self.force_consistency_loss_weight * force_loss,
            "metrics": {
                "flow_matching_loss": flow_loss.detach(),
                "force_consistency_loss": force_loss.detach(),
            },
            "contact_cond": contact,
        }

    def forward(self, batch):
        return self.forward_train(batch)

    @torch.no_grad()
    def forward_inference(
        self,
        obs,
        num_inference_steps=None,
        solver="euler",
        return_contact_diagnostics=False,
    ):
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun.")
        normalized = self._normalize_obs(obs)
        obs_feat, visual = self._encode_obs(obs, normalized)
        contact = self._read_contact(
            normalized, obs, visual, return_contact_diagnostics
        )
        condition = torch.cat([obs_feat, contact.global_cond], dim=-1)
        steps = (
            self.num_inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        out = self._sample_actions(condition, obs, steps, solver)
        out.update(
            contact_cond=contact.global_cond, num_inference_steps=torch.tensor(steps)
        )
        if contact.trace is not None:
            out.update(summarize_contact_trace(contact.trace))
        return out
