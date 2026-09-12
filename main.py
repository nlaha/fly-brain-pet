"""Desktop fruit fly driven by a simulated connectome.

The application does not contain a scripted "cursor near -> flee" rule.  The
cursor is transformed into a looming visual stimulus for the identified LC4 /
LPLC2 neurons.  DNp01 (the Giant Fiber) is then allowed to emerge from the
network and is decoded as the fly's escape command.
"""
import math
from pathlib import Path

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from brain.connectome import load_connectome, build_curated_subgraph
from brain.simulator import LeakyRateSimulator
from brain.io_mapping import looming_to_stimulus, brain_to_motor_command
from pet.cursor import get_cursor_provider
from pet.overlay import run_overlay

DATA_DIR = Path(__file__).parent / "data" / "raw"
ANNOTATIONS = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"


def build_state():
    app = QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    cursor_provider = get_cursor_provider((screen.width(), screen.height()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    full = load_connectome(
        str(ANNOTATIONS),
        str(WEIGHTS),
        nt_path=str(NEUROTRANSMITTERS),
        device=device,
    )

    if device == "cuda":
        connectome = full
        print(f"running full connectome: {connectome.n_neurons} neurons")
    else:
        # Keep the identified looming -> escape pathway and its local partners,
        # rather than reducing the graph to broad visual/motor categories.
        connectome = build_curated_subgraph(full, hops=2)
        print(f"CPU circuit: {connectome.n_neurons} neurons")

    for role in ("looming", "escape", "steering", "forward", "escape_wing"):
        print(f"  {role}: {len(connectome.role_indices(role))}")

    sim = LeakyRateSimulator(connectome, device=device, tau=6.0, dt=1.0)

    # A fly needs an internal motor drive even when the visual system is
    # quiet. This is not an escape controller: it is tonic activity entering
    # the connectome, analogous to spontaneous locomotor drive. The brain
    # still has to transform it into the decoded motor output.
    forward_idx = connectome.role_indices("forward")
    tonic_motor_input = torch.zeros(connectome.n_neurons, device=device)
    if len(forward_idx):
        tonic_motor_input[forward_idx] = 1.5
        print(f"tonic locomotor input: {len(forward_idx)} forward neurons")
    else:
        print("WARNING: no forward neurons found; fly will have no tonic locomotion")

    state = {
        "pos": np.array([screen.width() * 0.5, screen.height() * 0.5], dtype=np.float64),
        "velocity": np.array([0.0, 0.0], dtype=np.float64),
        "heading": 0.0,
        "previous_distance": None,
        "previous_escape_rate": 0.0,
        "connectome": connectome,
        "sim": sim,
        "device": device,
        "app": app,
        "cursor_provider": cursor_provider,
        "screen_size": (screen.width(), screen.height()),
        "tonic_motor_input": tonic_motor_input,
        "debug_frame": 0,
    }
    return state


def make_step_callback(state):
    dt = 1.0 / 60.0
    margin = 30.0

    # Desktop-scale mechanics are deliberately simple.  They are not a
    # behavioral controller: the brain supplies the motor command, while this
    # layer turns that command into position/velocity.
    normal_accel = 0.65
    escape_impulse = 10.0
    drag = 0.965
    max_speed = 14.0

    def step():
        cursor = np.array(state["cursor_provider"].position(), dtype=np.float64)

        stim, sensory = looming_to_stimulus(
            state["connectome"],
            state["pos"],
            cursor,
            state["heading"],
            state["previous_distance"],
            dt,
            device=state["device"],
        )
        state["previous_distance"] = sensory["distance"]

        # The only inputs to the brain are sensory looming plus tonic
        # locomotor drive. There is still no application-level "flee" rule.
        rates = state["sim"].step(stim + state["tonic_motor_input"])
        signal = brain_to_motor_command(state["connectome"], rates)

        # Useful diagnostics while tuning the biological mapping. These are
        # deliberately throttled so the terminal does not become the bottleneck.
        state["debug_frame"] += 1
        if state["debug_frame"] % 120 == 0:
            print(
                f"pos=({state['pos'][0]:.1f},{state['pos'][1]:.1f}) "
                f"speed={np.linalg.norm(state['velocity']):.2f} "
                f"loom={sensory['looming']:.3f} "
                f"escape={signal['escape_rate']:.3f} "
                f"forward={signal['walk_drive']:.3f} "
                f"turn={signal['turn']:.4f}"
            )

        # Normal locomotion: the brain's steering and forward command influence
        # the physical fly continuously.
        state["heading"] += float(signal["turn"])
        forward = np.array([
            math.cos(state["heading"]),
            math.sin(state["heading"]),
        ])
        state["velocity"] += forward * (normal_accel * float(signal["walk_drive"]))

        # IMPORTANT: there is no cursor-distance escape condition here.
        # A flight impulse occurs only on the rising edge of the simulated
        # Giant Fiber population.  The neural network made the decision.
        escape_rate = float(signal["escape_rate"])
        gf_threshold = 3.0
        gf_spike = escape_rate >= gf_threshold and state["previous_escape_rate"] < gf_threshold
        state["previous_escape_rate"] = escape_rate

        if gf_spike:
            # The Giant Fiber is the escape command.  The visual input also
            # provides the spatial direction of the threat; translating that
            # command into a body impulse is the mechanical interface, not a
            # second behavioral decision.
            away = state["pos"] - cursor
            distance = float(np.linalg.norm(away))
            if distance > 1e-6:
                away /= distance
                # Small stochastic component prevents perfectly straight,
                # unnatural desktop trajectories while preserving avoidance.
                tangent = np.array([-away[1], away[0]])
                escape_direction = away * 0.92 + tangent * np.random.uniform(-0.18, 0.18)
                escape_direction /= max(float(np.linalg.norm(escape_direction)), 1e-6)
                state["velocity"] += escape_direction * escape_impulse
                state["heading"] = math.atan2(escape_direction[1], escape_direction[0])

        state["velocity"] *= drag
        speed = float(np.linalg.norm(state["velocity"]))
        if speed > max_speed:
            state["velocity"] *= max_speed / speed

        state["pos"] += state["velocity"]

        # Reflect from screen edges instead of teleporting or using a behavior
        # state.  This is just desktop physics.
        width, height = state["screen_size"]
        if state["pos"][0] < margin:
            state["pos"][0] = margin
            state["velocity"][0] = abs(state["velocity"][0])
        elif state["pos"][0] > width - margin:
            state["pos"][0] = width - margin
            state["velocity"][0] = -abs(state["velocity"][0])

        if state["pos"][1] < margin:
            state["pos"][1] = margin
            state["velocity"][1] = abs(state["velocity"][1])
        elif state["pos"][1] > height - margin:
            state["pos"][1] = height - margin
            state["velocity"][1] = -abs(state["velocity"][1])

        # Let heading follow the actual body velocity after an escape.
        if speed > 0.25:
            state["heading"] = math.atan2(state["velocity"][1], state["velocity"][0])

        return tuple(state["pos"]), math.degrees(state["heading"]), bool(signal["escape"]), float(signal["wing_drive"])

    return step


if __name__ == "__main__":
    state = build_state()
    run_overlay(make_step_callback(state), app=state["app"])
