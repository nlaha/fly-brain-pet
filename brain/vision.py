"""Coarse 2.5-D panoramic compound-eye model.

Targets are projected into angular retina coordinates. Their apparent angular
size and change in angular size provide the temporal looming cue. No distance
or target-specific escape command is passed to the brain.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from .connectome import Connectome

RETINA_WIDTH = 144
RETINA_HEIGHT = 10
HALF_FOV = math.radians(150.0)  # 300 degrees
TARGET_FLY_RADIUS = 6.0


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _retina_x(angle: float) -> float | None:
    if abs(angle) > HALF_FOV:
        return None
    return (angle + HALF_FOV) / (2.0 * HALF_FOV)


def project_target(fly_pos, heading, target_pos, target_radius, previous, dt: float):
    delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(fly_pos, dtype=np.float64)
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    angle = wrap_angle(math.atan2(delta[1], delta[0]) - heading)

    x = _retina_x(angle)
    # Preserve the real angular size all the way out to long range.  A floor
    # here would make distant targets stop producing temporal expansion until
    # they got close enough to cross that artificial threshold.
    apparent_radius = math.atan2(target_radius, distance)
    if isinstance(previous, dict):
        previous_radius = float(previous.get("apparent_radius", apparent_radius))
        previous_angle = float(previous.get("angle", angle))
        previous_distance = float(previous.get("distance", distance))
    elif previous is None:
        previous_radius = apparent_radius
        previous_angle = angle
        previous_distance = distance
    else:
        previous_radius = float(previous)
        previous_angle = angle
        previous_distance = distance

    expansion = max(0.0, apparent_radius - previous_radius)
    expansion_rate = expansion / max(previous_radius, 1e-9)
    angular_velocity = wrap_angle(angle - previous_angle)
    # Use the actual simulation timestep; render frequency is not necessarily 60 Hz.
    # A first observation has previous_distance == distance, so it cannot create
    # a spurious closing-speed spike.
    closing_speed = max(0.0, (previous_distance - distance) / max(float(dt), 1e-6))

    return {
        "distance": distance,
        "angle": angle,
        "retina_x": x,
        "apparent_radius": apparent_radius,
        "expansion_rate": expansion_rate,
        "angular_velocity": angular_velocity,
        "closing_speed": closing_speed,
        "visible": x is not None,
    }


def render_retina(targets: list[dict], previous_by_id: dict, dt: float):
    retina = np.zeros((RETINA_HEIGHT, RETINA_WIDTH), dtype=np.float32)
    metadata = []
    xs = np.arange(RETINA_WIDTH, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, RETINA_HEIGHT, dtype=np.float32)
    vertical = np.exp(-0.5 * (ys / 0.62) ** 2)

    for target in targets:
        p = project_target(target["fly_pos"], target["heading"], target["pos"],
                           target["radius"], previous_by_id.get(target["id"]), dt)
        metadata.append({**p, "id": target["id"], "kind": target["kind"]})
        previous_by_id[target["id"]] = {"apparent_radius": p["apparent_radius"], "angle": p["angle"], "distance": p["distance"]}
        if not p["visible"]:
            continue

        cx = p["retina_x"] * (RETINA_WIDTH - 1)
        width = max(1.0, (2.0 * p["apparent_radius"]) / (2.0 * HALF_FOV) * RETINA_WIDTH)
        sigma = max(0.75, width * 0.55)
        profile = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
        retina += vertical[:, None] * profile[None, :]

    np.clip(retina, 0.0, 1.0, out=retina)
    return retina, metadata


def vision_drive(connectome: Connectome, retina: np.ndarray, metadata: list[dict], gain: float = 20.0):
    """Return (looming neuron indices, drives) without allocating an N-neuron GPU tensor."""
    idx = connectome.role_indices("looming")
    if len(idx) == 0:
        return idx, np.empty(0, dtype=np.float32)

    left = right = 0.0
    for m in metadata:
        if not m["visible"]:
            continue
        expansion = max(0.0, m["expansion_rate"])
        apparent = m["apparent_radius"]
        angular_speed = abs(m.get("angular_velocity", 0.0))
        distance = m["distance"]

        # LC4 is velocity-sensitive while LPLC2 is sensitive to the terminal
        # size of a loom. We therefore preserve both cues: fractional radial
        # expansion plus retinal motion. A close fly crossing the visual field
        # can now drive the same visual pathway even when its size does not
        # increase monotonically from one frame to the next.
        expansion_term = min(1.0, expansion / 0.004)
        motion_term = min(1.0, angular_speed / 0.035)
        closing_term = min(1.0, m.get("closing_speed", 0.0) / 180.0)
        size_term = min(1.0, apparent / 0.035)
        # Keep a long visual horizon: distant targets are weak, but never
        # abruptly disappear. Fast lateral crossings and closing motion both
        # remain visible to the looming pathway.
        proximity = 1.0 / (1.0 + distance / 450.0)
        local = expansion_term * (0.30 + 0.70 * size_term)
        local += motion_term * proximity * 0.65
        local += closing_term * proximity * 0.75
        x = m["retina_x"]
        angle_weight = 0.55 + 0.45 * math.cos((x - 0.5) * math.pi)
        local *= angle_weight
        if x < 0.5:
            left += local
        else:
            right += local

    total_visual = float(np.mean(retina))
    left += total_visual * 0.03
    right += total_visual * 0.03
    sides = connectome.side[idx]
    drive = np.where(sides < 0, gain * left,
                     np.where(sides > 0, gain * right, gain * 0.5 * (left + right)))
    return idx, np.clip(drive, 0.0, 50.0).astype(np.float32)


def vision_to_stimulus(connectome: Connectome, retina: np.ndarray, metadata: list[dict],
                        device="cpu", gain: float = 8.0):
    """Compatibility wrapper; use vision_drive in the batched simulator."""
    stim = torch.zeros(connectome.n_neurons, device=device)
    idx, drive = vision_drive(connectome, retina, metadata, gain=gain)
    if len(idx):
        stim[torch.as_tensor(idx, dtype=torch.long, device=device)] = torch.as_tensor(drive, device=device)
    return stim
