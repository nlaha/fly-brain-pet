import math
from pathlib import Path

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from brain.connectome import load_connectome, build_curated_subgraph
from brain.simulator import LeakyRateSimulator
from brain.io_mapping import cursor_to_stimulus, rates_to_motor_command
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

    full = load_connectome(str(ANNOTATIONS), str(WEIGHTS), nt_path=str(NEUROTRANSMITTERS), device=device)
    if device == "cuda":
        connectome = full
        print(f"running full connectome: {connectome.n_neurons} neurons")
    else:
        connectome = build_curated_subgraph(full, hops=2)
        print(f"no GPU found — curated subgraph: {connectome.n_neurons} neurons")

    sim = LeakyRateSimulator(connectome, device=device)

    state = {
        "pos": np.array([700.0, 400.0]),
        "heading": 0.0,  # radians
        "connectome": connectome,
        "sim": sim,
        "device": device,
        "app": app,
        "cursor_provider": cursor_provider,
    }
    return state


def make_step_callback(state):
    def step():
        cursor = np.array(state["cursor_provider"].position())
        stim = cursor_to_stimulus(
            state["connectome"], state["pos"], cursor, state["heading"], device=state["device"]
        )
        rates = state["sim"].step(stim)
        turn, thrust = rates_to_motor_command(state["connectome"], rates)

        state["heading"] += turn
        state["pos"] += thrust * np.array([math.cos(state["heading"]), math.sin(state["heading"])])

        return tuple(state["pos"]), math.degrees(state["heading"])

    return step


if __name__ == "__main__":
    state = build_state()
    run_overlay(make_step_callback(state), app=state["app"])
