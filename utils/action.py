from __future__ import annotations

import numpy as np
import torch


def _gripper_indices(action_dim: int) -> tuple[int, ...]:
    if action_dim == 8:
        return (7,)
    if action_dim == 10:
        return (9,)
    if action_dim == 16:
        return (7, 15)
    raise NotImplementedError(f"Unsupported joint action dim: {action_dim}")


def absolute_actions_to_relative_actions(
    actions: np.ndarray,
    base_absolute_action: np.ndarray | None = None,
) -> np.ndarray:
    if base_absolute_action is None:
        base_absolute_action = actions[0].copy()

    out = actions.copy() - np.asarray(base_absolute_action, dtype=actions.dtype)[None, :]
    for idx in _gripper_indices(actions.shape[-1]):
        out[..., idx] = actions[..., idx]
    return out


def absolute_actions_to_relative_tensor(actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    return _convert_joint_action_tensor(actions, base_actions, inverse=True)


def relative_actions_to_absolute_tensor(actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    return _convert_joint_action_tensor(actions, base_actions, inverse=False)


def action_base_from_state(state_hist: torch.Tensor, action_dim: int) -> torch.Tensor:
    return state_hist[..., -1, :action_dim]


def _convert_joint_action_tensor(actions: torch.Tensor, base_actions: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    action_dim = int(actions.shape[-1])
    if int(base_actions.shape[-1]) != action_dim or actions.numel() != base_actions.numel():
        raise ValueError(
            "actions and base_actions must have matching leading dimensions, "
            f"got {actions.shape} and {base_actions.shape}"
        )

    out = actions - base_actions if inverse else base_actions + actions
    out = out.clone()
    for idx in _gripper_indices(action_dim):
        out[..., idx] = actions[..., idx]
    return out
