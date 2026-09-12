"""
Vision: cursor position relative to the fly -> injected current on visual
neurons, split left/right hemisphere by `side`, scaled by proximity.

Motor: descending + motor neuron firing rates -> (turn, thrust) command,
using left/right rate asymmetry for turning and overall rate for thrust.
"""
import numpy as np
import torch

from .connectome import Connectome


def cursor_to_stimulus(
    connectome: Connectome,
    fly_pos: np.ndarray,
    cursor_pos: np.ndarray,
    heading: float,
    gain: float = 20.0,
    device: str = "cpu",
) -> torch.Tensor:
    stim = torch.zeros(connectome.n_neurons, device=device)
    visual_idx = connectome.role_indices("visual")
    if len(visual_idx) == 0:
        return stim

    delta = cursor_pos - fly_pos
    distance = np.linalg.norm(delta) + 1e-6
    angle_to_cursor = np.arctan2(delta[1], delta[0]) - heading
    angle_to_cursor = (angle_to_cursor + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    proximity = gain / distance  # closer cursor = stronger drive
    # positive angle (cursor to the fly's left) drives left-side visual neurons more
    left_drive = proximity * max(0.0, np.sin(angle_to_cursor))
    right_drive = proximity * max(0.0, -np.sin(angle_to_cursor))
    base_drive = proximity * 0.3  # small omnidirectional component so it notices things behind it too

    sides = connectome.side[visual_idx]
    drive = np.where(sides < 0, left_drive, np.where(sides > 0, right_drive, 0.0)) + base_drive
    stim[visual_idx] = torch.tensor(drive, dtype=torch.float32, device=device)
    return stim


def rates_to_motor_command(connectome: Connectome, rates: torch.Tensor) -> tuple[float, float]:
    """Returns (turn, thrust). turn > 0 = turn left, thrust >= 0 = forward speed."""
    motor_idx = np.concatenate([connectome.role_indices("motor"), connectome.role_indices("descending")])
    if len(motor_idx) == 0:
        return 0.0, 0.0

    r = rates[motor_idx].cpu().numpy()
    sides = connectome.side[motor_idx]

    left_rate = r[sides < 0].mean() if (sides < 0).any() else 0.0
    right_rate = r[sides > 0].mean() if (sides > 0).any() else 0.0

    turn = float(right_rate - left_rate) * 0.05   # more right-side drive -> turn left, matches crossed motor control
    thrust = float((left_rate + right_rate) / 2) * 0.02
    return turn, thrust
