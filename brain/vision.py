"""A small 2.5-D compound-eye model used to turn the desktop world into vision.

The desktop is the XY plane, while distance from the fly acts as virtual depth.
Targets are projected onto a coarse panoramic retina.  The brain never receives
a distance value; it receives retinal intensity and temporal expansion.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .connectome import Connectome


RETINA_WIDTH = 72
RETINA_HEIGHT = 8
HALF_FOV = math.radians(150.0)  # 300 degree panoramic field
TARGET_FLY_RADIUS = 5.0


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _retina_x(angle: float) -> float | None:
    """Map relative angle to [0, 1], or None outside the visual field."""
    if abs(angle) > HALF_FOV:
        return None
    return (angle + HALF_FOV) / (2.0 * HALF_FOV)


def project_target(fly_pos, heading, target_pos, target_radius, previous):
    delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(fly_pos, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        distance = 1e-6
    angle = wrap_angle(math.atan2(delta[1], delta[0]) - heading)

    x = _retina_x(angle)
    apparent_radius = math.atan2(target_radius, distance)
    previous_radius = apparent_radius if previous is None else float(previous)
    expansion = max(0.0, (apparent_radius - previous_radius))
    # Convert angular expansion to a convenient per-frame scale.
    expansion_rate = expansion * 60.0

    return {
        "distance": distance,
        "angle": angle,
        "retina_x": x,
        "apparent_radius": apparent_radius,
        "expansion_rate": expansion_rate,
        "visible": x is not None,
    }


def render_retina(targets: list[dict], previous_by_id: dict, dt: float):
    """Render target silhouettes onto a coarse panoramic retina.

    Returns intensity [height,width] plus target metadata. This is deliberately
    an image-like representation: downstream code can be replaced by a richer
    photoreceptor/Lamina model later without changing world physics.
    """
    retina = np.zeros((RETINA_HEIGHT, RETINA_WIDTH), dtype=np.float32)
    metadata = []

    for target in targets:
        p = project_target(
            target["fly_pos"], target["heading"], target["pos"],
            target["radius"], previous_by_id.get(target["id"]),
        )
        metadata.append({**p, "id": target["id"], "kind": target["kind"]})
        previous_by_id[target["id"]] = p["apparent_radius"]
        if not p["visible"]:
            continue

        cx = p["retina_x"] * (RETINA_WIDTH - 1)
        # A projected object's angular diameter occupies multiple receptors.
        width = max(1.0, (2.0 * p["apparent_radius"]) / (2.0 * HALF_FOV) * RETINA_WIDTH)
        sigma = max(0.9, width * 0.55)
        xs = np.arange(RETINA_WIDTH, dtype=np.float32)
        profile = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)

        # A crude vertical compound-eye profile: stronger in the middle rows.
        ys = np.linspace(-1.0, 1.0, RETINA_HEIGHT)
        vertical = np.exp(-0.5 * (ys / 0.55) ** 2)
        retina += vertical[:, None] * profile[None, :]

    np.clip(retina, 0.0, 1.0, out=retina)
    return retina, metadata


def vision_to_stimulus(connectome: Connectome, retina: np.ndarray, metadata: list[dict],
                        device="cpu", gain: float = 8.0):
    """Convert retinal activity to directional looming-detector input.

    This is the sensory interface only. It does not decide to escape. The
    connectome receives stronger activity on the side where the retinal image
    is expanding and leaves the motor decision to its downstream circuitry.
    """
    stim = torch.zeros(connectome.n_neurons, device=device)
    idx = connectome.role_indices("looming")
    if len(idx) == 0:
        return stim

    # Per-target temporal expansion is the looming cue. Static nearby objects
    # still produce a weak visual response, but do not automatically trigger it.
    looming = 0.0
    left = 0.0
    right = 0.0
    for m in metadata:
        if not m["visible"]:
            continue
        x = m["retina_x"]
        expansion = max(0.0, m["expansion_rate"])
        apparent = m["apparent_radius"]
        local = min(1.0, expansion / 0.30) * min(1.0, apparent / 0.25)
        # Retinal position controls which side gets stronger drive.
        if x < 0.5:
            left += local
        else:
            right += local
        looming = max(looming, local)

    # The retina itself also supplies weak spatial visual drive.
    total_visual = float(np.mean(retina))
    left += total_visual * 0.05
    right += total_visual * 0.05

    sides = connectome.side[idx]
    drive = np.where(
        sides < 0, gain * left,
        np.where(sides > 0, gain * right, gain * 0.5 * (left + right)),
    )
    stim[idx] = torch.as_tensor(np.clip(drive, 0.0, 50.0), dtype=torch.float32, device=device)
    return stim
