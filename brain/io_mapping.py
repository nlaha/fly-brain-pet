"""Sensory and motor interfaces for the fly connectome.

There is deliberately no target-near -> escape behavior rule here.
Visual input is converted into a retinal *looming* stimulus and injected into
identified LC4/LPLC2 neurons. Whether the fly escapes is then determined by
the simulated network: DNp01 (the Giant Fiber) is read as the escape command.

The conversion from neural activity to desktop motion is necessarily a model:
the CNS connectome does not contain a desktop-scale body/flight mechanics
model.  The important boundary is that the decision to escape comes from the
connectome, not from application code.
"""
import math
import numpy as np
import torch

from .connectome import Connectome


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def retina_to_stimulus(
    connectome: Connectome,
    retina: np.ndarray,
    metadata: list[dict],
    device: str = "cpu",
    gain: float = 8.0,
) -> torch.Tensor:
    """Convert the fly's virtual retinal image into visual input.

    There is no target-specific escape rule here. The retinal image and its
    temporal expansion are the sensory signal; the connectome decides what to
    do with it.
    """
    stim = torch.zeros(connectome.n_neurons, device=device)
    idx = connectome.role_indices("looming")
    if len(idx) == 0:
        return stim

    looming = 0.0
    left = 0.0
    right = 0.0
    for obj in metadata:
        if not obj["visible"]:
            continue
        expansion = max(0.0, obj["expansion_rate"])
        apparent = obj["apparent_radius"]
        local = min(1.0, expansion / 0.30) * min(1.0, apparent / 0.25)
        x = obj["retina_x"]
        if x < 0.5:
            left += local
        else:
            right += local
        looming = max(looming, local)

    total_visual = float(np.mean(retina))
    left += total_visual * 0.05
    right += total_visual * 0.05

    sides = connectome.side[idx]
    drive = np.where(
        sides < 0, gain * left,
        np.where(sides > 0, gain * right, gain * 0.5 * (left + right)),
    )
    stim[idx] = torch.as_tensor(
        np.clip(drive, 0.0, 50.0), dtype=torch.float32, device=device
    )
    return stim


def _population_rate(connectome: Connectome, rates: torch.Tensor, role: str) -> tuple[float, float, float]:
    idx = connectome.role_indices(role)
    if len(idx) == 0:
        return 0.0, 0.0, 0.0

    r = rates[idx].detach().cpu().numpy()
    sides = connectome.side[idx]
    left = float(r[sides < 0].mean()) if (sides < 0).any() else 0.0
    right = float(r[sides > 0].mean()) if (sides > 0).any() else 0.0
    return float(r.mean()), left, right


def brain_to_motor_command(connectome: Connectome, rates: torch.Tensor) -> dict[str, float | bool]:
    """Decode named motor populations, including the Giant Fiber.

    ``escape`` is true only when the simulated DNp01 population actually
    crosses its spike/command threshold.  This is the neural decision point.
    """
    escape_rate, _, _ = _population_rate(connectome, rates, "escape")
    steering_rate, steering_left, steering_right = _population_rate(connectome, rates, "steering")
    forward_rate, _, _ = _population_rate(connectome, rates, "forward")
    wing_rate, _, _ = _population_rate(connectome, rates, "escape_wing")

    # The rate simulator has a normalized 0..50 range.  DNp01 is only two
    # neurons in the identified circuit, so a sustained high rate is treated
    # as a command spike rather than ordinary locomotion.
    escape = escape_rate >= 3.0

    turn = (steering_right - steering_left) * 0.035
    walk_drive = min(1.0, forward_rate / 8.0)
    wing_drive = min(1.0, wing_rate / 8.0)

    return {
        "escape": escape,
        "escape_rate": escape_rate,
        "turn": float(turn),
        "walk_drive": float(walk_drive),
        "wing_drive": float(wing_drive),
        "steering_rate": steering_rate,
    }


# Backwards-compatible alias for callers that used the old generic decoder.
def rates_to_motor_command(connectome: Connectome, rates: torch.Tensor) -> tuple[float, float]:
    signal = brain_to_motor_command(connectome, rates)
    return float(signal["turn"]), float(signal["walk_drive"])
