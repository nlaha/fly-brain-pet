"""Desktop fruit flies driven by independent simulated connectomes.

Each fly owns its own brain simulator. Visual input is generated from the
other flies, so social/avoidance behavior can emerge from the same looming
pathway rather than from a scripted "near -> flee" controller.
"""
import math
import argparse
import signal
from pathlib import Path

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from brain.connectome import load_connectome, build_curated_subgraph
from brain.simulator import LeakyRateSimulator
from brain.io_mapping import retina_to_stimulus, brain_to_motor_command
from brain.vision import render_retina, vision_to_stimulus, TARGET_FLY_RADIUS
from pet.overlay import run_overlay

DATA_DIR = Path(__file__).parent / "data" / "raw"
ANNOTATIONS = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"


def build_fly(full, device, screen_size, index):
    # Every fly gets an independent simulator state.  The graph structure can
    # be shared because it is immutable; neuron rates/membranes are not shared.
    if device == "cuda":
        connectome = full
    else:
        connectome = build_curated_subgraph(full, hops=2)

    sim = LeakyRateSimulator(connectome, device=device, tau=6.0, dt=1.0)

    forward_idx = connectome.role_indices("forward")
    tonic_indices = torch.as_tensor(forward_idx, dtype=torch.long, device=device)
    if len(forward_idx):
        sim.external_input[tonic_indices] = 1.5

    width, height = screen_size
    # Spread initial flies out so they don't immediately overlap.
    angle = index * (2.0 * math.pi / max(1, 7))
    center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
    offset = np.array([math.cos(angle), math.sin(angle)]) * min(width, height) * 0.22

    return {
        "id": index,
        "pos": center + offset,
        "velocity": np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * 1.0,
        "heading": angle,
        "previous_escape_rate": 0.0,
        "previous_distances": {},
        "previous_retina": {},
        "connectome": connectome,
        "sim": sim,
        "device": device,
        "screen_size": screen_size,
        "debug_data": {},
    }


def build_world(num_flies=1):
    app = QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    screen_size = (screen.width(), screen.height())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    full = load_connectome(
        str(ANNOTATIONS),
        str(WEIGHTS),
        nt_path=str(NEUROTRANSMITTERS),
        device=device,
    )
    print(f"connectome graph: {full.n_neurons} neurons")

    flies = [build_fly(full, device, screen_size, i) for i in range(num_flies)]
    # Roles are identical for shared graph objects, so report once.
    graph = flies[0]["connectome"]
    print(f"running {num_flies} independent brain simulation(s)")
    if device == "cuda":
        print(f"  full connectome: {graph.n_neurons} neurons/fly")
    else:
        print(f"  CPU circuit: {graph.n_neurons} neurons/fly")
    for role in ("looming", "escape", "steering", "forward", "escape_wing"):
        print(f"  {role}: {len(graph.role_indices(role))}")
    if not len(graph.role_indices("forward")):
        print("WARNING: no forward neurons found; fly will have no tonic locomotion")
    else:
        print(f"  tonic locomotor input: {len(graph.role_indices('forward'))} forward neurons/fly")

    return {
        "app": app,
        "flies": flies,
        "screen_size": screen_size,
        "device": device,
        "frame": 0,
    }


def make_world_step_callback(world):
    flies = world["flies"]
    screen_width, screen_height = world["screen_size"]
    dt = 1.0 / 60.0
    margin = 30.0
    normal_accel = 0.65
    escape_impulse = 10.0
    drag = 0.965
    max_speed = 14.0

    def step():
        world["frame"] += 1
        # Snapshot positions so all brains see the same world at the beginning
        # of this frame. Other flies are the only visual targets.
        positions = {fly["id"]: fly["pos"].copy() for fly in flies}

        views = []
        for fly in flies:
            targets = []
            for other in flies:
                if other["id"] != fly["id"]:
                    targets.append({"id": f"fly-{other['id']}", "kind": "fly", "pos": positions[other["id"]], "radius": TARGET_FLY_RADIUS})

            vision_targets = [
                {**t, "fly_pos": fly["pos"], "heading": fly["heading"]}
                for t in targets
            ]
            retina, visual_objects = render_retina(vision_targets, fly["previous_retina"], dt)
            stim = retina_to_stimulus(
                fly["connectome"], retina, visual_objects, device=fly["device"]
            )

            # Add sensory drive into the simulator-owned persistent input buffer.
            fly["sim"].external_input.add_(stim)
            rates = fly["sim"].step(fly["sim"].external_input)
            fly["sim"].external_input.sub_(stim)
            signal = brain_to_motor_command(fly["connectome"], rates)

            fly["heading"] += float(signal["turn"])
            forward = np.array([math.cos(fly["heading"]), math.sin(fly["heading"])])
            fly["velocity"] += forward * (normal_accel * float(signal["walk_drive"]))

            escape_rate = float(signal["escape_rate"])
            gf_threshold = 3.0
            gf_spike = escape_rate >= gf_threshold and fly["previous_escape_rate"] < gf_threshold
            fly["previous_escape_rate"] = escape_rate

            if gf_spike:
                # DNp01 supplies the escape command. The mechanical interface
                # converts the strongest expanding visual target into a flight
                # direction; there is no distance/target behavioral escape rule.
                weighted_away = np.zeros(2, dtype=np.float64)
                total_weight = 0.0
                for obj in visual_objects:
                    if not obj["visible"] or obj["expansion_rate"] <= 0:
                        continue
                    target = next(t["pos"] for t in targets if t["id"] == obj["id"])
                    away = fly["pos"] - target
                    d = float(np.linalg.norm(away))
                    if d > 1e-6:
                        weight = max(0.0, obj["expansion_rate"])
                        weighted_away += away / d * weight
                        total_weight += weight
                if total_weight > 0:
                    escape_direction = weighted_away / total_weight
                    norm = float(np.linalg.norm(escape_direction))
                    if norm > 1e-6:
                        escape_direction /= norm
                        fly["velocity"] += escape_direction * escape_impulse
                        fly["heading"] = math.atan2(escape_direction[1], escape_direction[0])

            fly["velocity"] *= drag
            speed = float(np.linalg.norm(fly["velocity"]))
            if speed > max_speed:
                fly["velocity"] *= max_speed / speed
                speed = max_speed
            fly["pos"] += fly["velocity"]

            if fly["pos"][0] < margin:
                fly["pos"][0] = margin; fly["velocity"][0] = abs(fly["velocity"][0])
            elif fly["pos"][0] > screen_width - margin:
                fly["pos"][0] = screen_width - margin; fly["velocity"][0] = -abs(fly["velocity"][0])
            if fly["pos"][1] < margin:
                fly["pos"][1] = margin; fly["velocity"][1] = abs(fly["velocity"][1])
            elif fly["pos"][1] > screen_height - margin:
                fly["pos"][1] = screen_height - margin; fly["velocity"][1] = -abs(fly["velocity"][1])

            speed = float(np.linalg.norm(fly["velocity"]))
            if speed > 0.25:
                fly["heading"] = math.atan2(fly["velocity"][1], fly["velocity"][0])

            looming_value = max((max(0.0, o["expansion_rate"]) for o in visual_objects if o["visible"]), default=0.0)
            fly["debug_data"] = {
                "vision": retina,
                "visual_objects": visual_objects,
                "sensory": {
                    "looming": float(min(1.0, looming_value / 0.30)),
                    "approach_speed": float(max((o["expansion_rate"] for o in visual_objects), default=0.0)),
                },
                "signal": signal,
                "speed": speed,
            }
            for role in ("looming", "escape", "forward", "steering", "escape_wing"):
                idx = fly["connectome"].role_indices(role)
                fly["debug_data"][role] = rates[idx].detach().float().cpu().numpy().tolist() if len(idx) else []

            views.append({
                "pos": tuple(fly["pos"]),
                "heading": math.degrees(fly["heading"]),
                "escape": bool(signal["escape"]),
                "wing_drive": float(signal["wing_drive"]),
                "debug_data": fly["debug_data"],
                "id": fly["id"],
            })

        if world["frame"] % 120 == 0:
            print(" | ".join(
                f"fly={v['id']} speed={v['debug_data']['speed']:.2f} "
                f"loom={v['debug_data']['sensory']['looming']:.3f} "
                f"escape={v['debug_data']['signal']['escape_rate']:.3f} "
                f"walk={v['debug_data']['signal']['walk_drive']:.3f}"
                for v in views
            ))

        return views

    return step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Desktop fruit fly connectome simulation")
    parser.add_argument("--debug", action="store_true", help="draw visual fields and neural activity")
    parser.add_argument("--flies", type=int, default=1, help="number of independent fly brains (default: 1)")
    args = parser.parse_args()
    if args.flies < 1:
        parser.error("--flies must be at least 1")

    world = build_world(args.flies)
    print(f"debug mode: {'on' if args.debug else 'off'}")
    print(f"flies: {args.flies} (each has an independent neural state)")
    # Qt's event loop otherwise owns the main thread and can make Ctrl+C feel
    # like it is being ignored. Convert SIGINT into a normal Qt shutdown.
    signal.signal(signal.SIGINT, lambda *_: world["app"].quit())
    run_overlay(make_world_step_callback(world), app=world["app"], debug=args.debug, num_flies=args.flies)
