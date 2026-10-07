from typing import Callable, Dict

import torch


def dict_apply(x: Dict, func: Callable):
    out = {}
    for k, v in x.items():
        if isinstance(v, dict):
            out[k] = dict_apply(v, func)
        else:
            out[k] = func(v)
    return out


def move_to_device(x: Dict, device: torch.device):
    def _to(v):
        if torch.is_tensor(v):
            return v.to(device, non_blocking=True)
        return v

    return dict_apply(x, _to)
