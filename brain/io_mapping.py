"""GPU-resident decoder for the named descending command populations."""
from __future__ import annotations
import numpy as np
import torch
from .connectome import Connectome


class BatchedMotorDecoder:
    """Decode all flies in one GPU pass and return only 5 scalars/fly."""
    def __init__(self, connectome: Connectome, device: str):
        self.device = device
        self.c = connectome
        self.indices = {}
        self.left_masks = {}
        self.right_masks = {}
        self.has_left = {}
        self.has_right = {}
        for role in ("escape", "steering", "flight_steering", "forward", "escape_wing"):
            idx = connectome.role_indices(role)
            t = torch.as_tensor(idx, dtype=torch.long, device=device)
            self.indices[role] = t
            side = connectome.side[idx]
            self.left_masks[role] = torch.as_tensor(side < 0, dtype=torch.bool, device=device)
            self.right_masks[role] = torch.as_tensor(side > 0, dtype=torch.bool, device=device)
            self.has_left[role] = bool((side < 0).any())
            self.has_right[role] = bool((side > 0).any())

    @torch.inference_mode()
    def decode(self, rates: torch.Tensor) -> torch.Tensor:
        batch = rates.shape[1]
        out = torch.zeros((batch, 5), device=rates.device, dtype=torch.float32)
        for role, col in (("escape", 0), ("forward", 2), ("escape_wing", 3)):
            idx = self.indices[role]
            if idx.numel():
                out[:, col] = rates.index_select(0, idx).mean(dim=0)
        # DNa01/DNa02 are the core bilateral steering pair. DNa11/DNg13
        # add additional descending steering pathways present in this dataset.
        # We combine them as a neural population rather than adding a physics
        # side turn. The bilateral difference is the actual motor command.
        idx = self.indices["flight_steering"]
        if idx.numel():
            r = rates.index_select(0, idx)
            lm, rm = self.left_masks["flight_steering"], self.right_masks["flight_steering"]
            left = r[lm].mean(dim=0) if self.has_left["flight_steering"] else torch.zeros(batch, device=rates.device)
            right = r[rm].mean(dim=0) if self.has_right["flight_steering"] else torch.zeros(batch, device=rates.device)
            out[:, 1] = torch.tanh((right - left) * 0.18) * 0.16
            out[:, 4] = r.mean(dim=0)
        return out


def brain_to_motor_command(connectome: Connectome, rates: torch.Tensor, fly_index: int = 0):
    decoder = BatchedMotorDecoder(connectome, rates.device.type)
    x = decoder.decode(rates)[fly_index].detach().cpu().numpy()
    escape_rate, turn, forward_rate, wing_rate, steering_rate = x
    return {
        "escape": bool(escape_rate >= 3.0), "escape_rate": float(escape_rate),
        "turn": float(turn), "walk_drive": float(min(1.0, forward_rate / 8.0)),
        "wing_drive": float(min(1.0, wing_rate / 8.0)), "steering_rate": float(steering_rate),
    }


def rates_to_motor_command(connectome, rates):
    signal = brain_to_motor_command(connectome, rates)
    return float(signal["turn"]), float(signal["walk_drive"])
