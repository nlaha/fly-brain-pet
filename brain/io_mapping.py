"""Sensory and motor interfaces for the fly connectome.

There is deliberately no ``if cursor_near: escape`` behavior rule here.
The cursor is converted into a visual *looming* stimulus and injected into
identified LC4/LPLC2 neurons.  Whether the fly escapes is then determined by
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


def looming_to_stimulus(
    connectome: Connectome,
    fly_pos: np.ndarray,
    cursor_pos: np.ndarray,
    heading: float,
    previous_distance: float | None,
    dt: float,
    gain: float = 8.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Convert cursor motion into directional looming input.

    LC4/LPLC2 are the identified looming-detector populations used by the
    reference DesktopFly implementation.  The sensory transform is not an
    escape rule: it only describes what the fly's visual system sees.

    We model looming using both apparent proximity and positive approach rate.
    A stationary cursor therefore produces little/no looming drive, while a
    cursor rapidly approaching the fly produces a much stronger signal.
    """
    stim = torch.zeros(connectome.n_neurons, device=device)
    idx = connectome.role_indices("looming")

    delta = cursor_pos - fly_pos
    distance = float(np.linalg.norm(delta))
    safe_distance = max(distance, 1.0)

    previous = safe_distance if previous_distance is None else max(previous_distance, 1.0)
    approach_speed = max(0.0, (previous - safe_distance) / max(dt, 1e-6))

    # A finite visual field.  The two populations receive asymmetric drive,
    # allowing downstream steering circuitry to carry spatial information.
    angle = _wrap_angle(float(np.arctan2(delta[1], delta[0]) - heading))
    front = max(0.0, math.cos(angle))
    left = max(0.0, math.sin(angle))
    right = max(0.0, -math.sin(angle))

    # Apparent angular expansion grows rapidly as an object gets close.  The
    # approach term is intentionally more important than static proximity.
    proximity = min(1.0, 120.0 / safe_distance)
    looming = min(1.0, approach_speed / 900.0) * (0.35 + 0.65 * front)
    static_visual = 0.08 * proximity * front
    total = gain * (looming + static_visual)

    if len(idx):
        sides = connectome.side[idx]
        # LC4/LPLC2 side asymmetry preserves the location of the looming object.
        drive = np.where(
            sides < 0,
            total * (0.35 + 0.65 * left),
            np.where(sides > 0, total * (0.35 + 0.65 * right), total * 0.5),
        )
        stim[idx] = torch.as_tensor(drive, dtype=torch.float32, device=device)

    return stim, {
        "distance": safe_distance,
        "approach_speed": approach_speed,
        "looming": float(looming),
        "angle": angle,
    }


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
