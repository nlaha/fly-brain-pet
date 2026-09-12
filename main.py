"""Desktop fruit flies driven by independent simulated connectomes.

The connectome graph/weights are shared, while each fly has independent neural
state. Brain updates are batched on the GPU to avoid launching one sparse-matrix
operation per fly. Other flies are the only visual targets.
"""
from __future__ import annotations
import math
import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from brain.connectome import load_connectome, build_curated_subgraph
from brain.simulator import BatchedLeakyRateSimulator
from brain.io_mapping import BatchedMotorDecoder
from brain.vision import render_retina, vision_drive, TARGET_FLY_RADIUS
from pet.overlay import run_overlay

DATA_DIR = Path(__file__).parent / "data" / "raw"
ANNOTATIONS = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"


def build_world(num_flies=1, debug=False):
    app = QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    screen_size = (screen.width(), screen.height())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.perf_counter()
    full = load_connectome(str(ANNOTATIONS), str(WEIGHTS), nt_path=str(NEUROTRANSMITTERS), device=device)
    print(f"connectome graph: {full.n_neurons} neurons ({(time.perf_counter()-t0)*1000:.0f} ms load)")

    connectome = full if device == "cuda" else build_curated_subgraph(full, hops=2)
    roles = {r: np.asarray(connectome.role_indices(r), dtype=np.int64) for r in
             ("looming", "escape", "steering", "flight_steering", "saccade_inhibitory", "saccade_excitory", "forward", "escape_wing")}
    simulator = BatchedLeakyRateSimulator(connectome, num_flies, tau=6.0, dt=1.0, device=device)
    motor_decoder = BatchedMotorDecoder(connectome, device)
    xyz = np.asarray(connectome.xyz, dtype=np.float32)
    valid_xyz = np.any(xyz != 0, axis=1)
    if valid_xyz.any():
        lo = xyz[valid_xyz].min(axis=0)
        hi = xyz[valid_xyz].max(axis=0)
        span = np.maximum(hi - lo, 1.0)
        brain_xyz_norm = (xyz - lo) / span
    else:
        brain_xyz_norm = np.zeros_like(xyz)

    # Baseline locomotor drive lives inside the neural input, not in the
    # physics layer. Forward and steering command neurons receive a small
    # tonic/stochastic background, giving the connectome something to modulate
    # during spontaneous flight. No movement is injected directly.
    if len(roles["forward"]):
        idx = torch.as_tensor(roles["forward"], dtype=torch.long, device=device)
        simulator.external_input[idx, :] = 2.5
    else:
        print("WARNING: no forward neurons found; fly may remain stationary")
    if len(roles["flight_steering"]):
        sidx = torch.as_tensor(roles["flight_steering"], dtype=torch.long, device=device)
        simulator.external_input[sidx, :] = 0.10
    
    width, height = screen_size
    center = np.array([width * .5, height * .5], dtype=np.float64)
    flies = []
    for i in range(num_flies):
        angle = i * (2.0 * math.pi / max(1, num_flies))
        offset = np.array([math.cos(angle), math.sin(angle)]) * min(width, height) * .22
        flies.append({
            "id": i,
            "pos": center + offset,
            "velocity": np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * 1.0,
            "heading": angle,
            "previous_escape_rate": 0.0,
            "previous_retina": {},
            "debug_data": {},
        })

    # Distinct initial neural states prevent identical connectome copies from
    # following the same trajectory while preserving the same graph and weights.
    with torch.no_grad():
        simulator.rates.normal_(0.0, 0.02).clamp_(min=0.0)

    print(f"running {num_flies} independent brain simulation(s) as one GPU batch")
    print(f"  full connectome: {connectome.n_neurons} neurons/fly")
    for role in roles:
        print(f"  {role}: {len(roles[role])}")
    if len(roles["forward"]):
        print(f"  tonic locomotor input: {len(roles['forward'])} forward neurons/fly")

    return {
        "app": app, "flies": flies, "screen_size": screen_size, "device": device,
        "connectome": connectome, "sim": simulator, "roles": roles, "motor_decoder": motor_decoder,
        "brain_xyz_norm": brain_xyz_norm, "brain_debug_cache": [None] * num_flies,
        "debug_role_cache": [None] * num_flies,
        "debug_last_sample": 0.0,
        "debug_generation": 0,
        "debug_sample_interval": max(0.125, 0.125 * math.sqrt(num_flies)),
        "frame": 0, "debug": debug,
    }


def make_world_step_callback(world):
    flies = world["flies"]
    connectome = world["connectome"]
    sim = world["sim"]
    roles = world["roles"]
    screen_width, screen_height = world["screen_size"]
    dt = 1.0 / 60.0
    margin = 30.0
    normal_accel = 0.9
    escape_impulse = 10.0
    drag = 0.985
    max_speed = 14.0
    min_cruise_speed = 1.5
    n = len(flies)
    debug = world["debug"]

    def step():
        world["frame"] += 1
        positions = np.stack([f["pos"] for f in flies], axis=0).copy()
        headings = [f["heading"] for f in flies]

        # Reuse one dense input matrix. Clear only the transient sensory input;
        # tonic locomotion remains in the persistent baseline columns.
        # Reuse the sensory buffer; only looming rows can contain transient input.
        sensory = world.get("sensory_buffer")
        if sensory is None:
            sensory = torch.zeros_like(sim.external_input)
            world["sensory_buffer"] = sensory
        looming_idx = roles["looming"]
        if len(looming_idx):
            sensory[torch.as_tensor(looming_idx, dtype=torch.long, device=world["device"]), :].zero_()

        all_visual = []
        retinas = []
        for fi, fly in enumerate(flies):
            targets = []
            for oi, other in enumerate(flies):
                if oi == fi:
                    continue
                targets.append({
                    "id": f"fly-{oi}", "kind": "fly", "pos": positions[oi],
                    "radius": TARGET_FLY_RADIUS, "fly_pos": positions[fi],
                    "heading": headings[fi],
                })
            # Screen edges are visual obstacles. They enter the same retinal
            # pathway as other objects; there is no physics-side bounce rule.
            # The four targets are placed just beyond the visible boundary so
            # the wall can loom before the fly reaches it.
            wall_pad = 90.0
            wall_targets = (
                ("wall-left", np.array([-wall_pad, positions[fi, 1]])),
                ("wall-right", np.array([screen_width + wall_pad, positions[fi, 1]])),
                ("wall-top", np.array([positions[fi, 0], -wall_pad])),
                ("wall-bottom", np.array([positions[fi, 0], screen_height + wall_pad])),
            )
            for wall_id, wall_pos in wall_targets:
                targets.append({
                    "id": wall_id, "kind": "wall", "pos": wall_pos,
                    "radius": 75.0, "fly_pos": positions[fi],
                    "heading": headings[fi],
                })
            retina, objects = render_retina(targets, fly["previous_retina"], dt)
            retinas.append(retina)
            all_visual.append(objects)
            idx, drive = vision_drive(connectome, retina, objects)
            if len(idx):
                sensory[torch.as_tensor(idx, dtype=torch.long, device=world["device"]), fi] = torch.as_tensor(drive, device=world["device"])

        # Spontaneous turning is NOT injected into the decoded heading.
        # The simulator itself supplies intrinsic neural noise to the full
        # connectome; DNa/DNa-like command activity must therefore emerge from
        # the network before it can affect flight.

        # Exactly one sparse matrix multiplication advances every independent
        # brain. The graph/weights are shared; columns are independent states.
        sim.external_input.add_(sensory)
        brain_t0 = time.perf_counter()
        rates = sim.step()
        if world["device"] == "cuda":
            torch.cuda.synchronize()
        brain_ms = (time.perf_counter() - brain_t0) * 1000.0
        sim.external_input.sub_(sensory)
        # Decode every fly's command populations on the GPU in one operation.
        # This avoids N * 4 tiny GPU->CPU synchronizations per frame.
        motor = world["motor_decoder"].decode(rates).detach().cpu().numpy()

        # Debug sampling is deliberately decoupled from the render loop. The
        # old implementation ran torch.topk over the entire 211K-neuron x N
        # matrix every few frames and copied several role populations to CPU on
        # every frame. That is useful diagnostically but it can dominate frame
        # time and force a GPU synchronization. Instead, keep a fixed anatomical
        # sample and refresh its activity at ~8 Hz. The simulation itself is
        # unchanged; this only limits debug observation overhead.
        if debug:
            now = time.perf_counter()
            # Scale the expensive brain visualization sampling down as fly
            # count rises. The brains still run every simulation step; only
            # the debug snapshot is less frequent. sqrt(N) keeps a single fly
            # responsive while avoiding N-fold debug overhead at high counts.
            sample_interval = world.get("debug_sample_interval", 0.125)
            if now - world.get("debug_last_sample", 0.0) >= sample_interval or world.get("brain_debug_cache")[0] is None:
                xyz_norm = world["brain_xyz_norm"]
                display_indices = world.get("brain_display_indices")
                if display_indices is None:
                    valid = np.flatnonzero(np.any(xyz_norm != 0, axis=1))
                    if len(valid) > 3000:
                        # Dense anatomical sample for a detailed brain view.
                        # The snapshot itself is rate-limited above, so detail
                        # does not have to be paid for every rendered frame.
                        display_indices = valid[np.linspace(0, len(valid) - 1, 3000, dtype=np.int64)]
                    else:
                        display_indices = valid
                    world["brain_display_indices"] = display_indices.astype(np.int64)
                d_idx = torch.as_tensor(display_indices, dtype=torch.long, device=world["device"])
                # One compact GPU->CPU transfer for all flies. No full-matrix
                # top-k, and no per-fly synchronization.
                sampled = rates.index_select(0, d_idx).transpose(0, 1).detach().cpu().numpy()
                # Batch all role readbacks into one compact CPU transfer.
                # The old implementation did one GPU indexing + CPU copy per
                # role per fly, which becomes surprisingly expensive at 10+.
                role_names = tuple(roles.keys())
                role_ranges = {}
                flat_role_indices = []
                for role in role_names:
                    idx = roles[role]
                    start = len(flat_role_indices)
                    flat_role_indices.extend(idx.tolist())
                    role_ranges[role] = (start, len(flat_role_indices))
                if flat_role_indices:
                    flat_idx = torch.as_tensor(flat_role_indices, dtype=torch.long, device=world["device"])
                    flat_rates = rates.index_select(0, flat_idx).detach().float().cpu().numpy()
                else:
                    flat_rates = np.empty((0, n), dtype=np.float32)

                role_cache = []
                for fi in range(n):
                    role_data = {}
                    for role in role_names:
                        a, b = role_ranges[role]
                        role_data[role] = flat_rates[a:b, fi].tolist()
                    role_cache.append(role_data)
                    vals = sampled[fi]
                    # Keep a detailed but bounded active cloud. Tiny points are
                    # rendered in the cached panel, so this remains readable
                    # without thousands of expensive painter operations.
                    keep = min(1200, len(vals))
                    if keep:
                        order = np.argpartition(vals, -keep)[-keep:]
                        order = order[np.argsort(vals[order])[::-1]]
                        ids = display_indices[order]
                        world["brain_debug_cache"][fi] = (
                            ids.astype(np.int64), xyz_norm[ids].copy(), vals[order].astype(np.float32)
                        )
                    else:
                        world["brain_debug_cache"][fi] = (np.empty(0, dtype=np.int64),
                                                            np.empty((0, 3), dtype=np.float32),
                                                            np.empty(0, dtype=np.float32))
                world["debug_role_cache"] = role_cache
                world["debug_last_sample"] = now
                world["debug_generation"] += 1

        views = []
        for fi, fly in enumerate(flies):
            escape_rate, turn, forward_rate, wing_rate, steering_rate = motor[fi]
            signal = {
                "escape": bool(escape_rate >= 3.0),
                "escape_rate": float(escape_rate),
                "turn": float(turn),
                "walk_drive": float(min(1.0, forward_rate / 8.0)),
                "wing_drive": float(min(1.0, wing_rate / 8.0)),
                "steering_rate": float(steering_rate),
            }

            # The connectome controls both propulsion and heading. Small
            # intrinsic fluctuations are applied to the motor *input* (not
            # the decoded movement) to model spontaneous variability. This
            # prevents identical flies from drifting in parallel while keeping
            # all actual movement downstream of the brain.
            # Only the decoded neural steering command changes heading.
            fly["heading"] += float(turn)
            forward = np.array([math.cos(fly["heading"]), math.sin(fly["heading"])])
            walk_drive = float(min(1.0, forward_rate / 8.0))
            fly["velocity"] += forward * (normal_accel * walk_drive)

            gf_threshold = 3.0
            gf_spike = escape_rate >= gf_threshold and fly["previous_escape_rate"] < gf_threshold
            fly["previous_escape_rate"] = escape_rate

            # DNp01 supplies the escape command. Direction is decoded from the
            # visual expansion pattern, not from a near-target rule.
            if gf_spike:
                weighted_away = np.zeros(2, dtype=np.float64)
                total_weight = 0.0
                for obj in all_visual[fi]:
                    if not obj["visible"] or obj["expansion_rate"] <= 0:
                        continue
                    oi = int(obj["id"].split("-")[1])
                    away = fly["pos"] - positions[oi]
                    d = float(np.linalg.norm(away))
                    if d > 1e-6:
                        w = max(0.0, obj["expansion_rate"])
                        weighted_away += away / d * w
                        total_weight += w
                if total_weight > 0:
                    direction = weighted_away / total_weight
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-6:
                        direction /= norm
                        fly["velocity"] += direction * escape_impulse
                        fly["heading"] = math.atan2(direction[1], direction[0])

            fly["velocity"] *= drag
            speed = float(np.linalg.norm(fly["velocity"]))
            # DNp09 is the locomotor command. If its tonic activity is low,
            # do not invent propulsion; preserve only existing inertia. When
            # it is active, keep a modest cruise floor so normal locomotion
            # looks like flight rather than a dead stop between commands.
            if walk_drive > 0.15 and speed < min_cruise_speed:
                fly["velocity"] = forward * min_cruise_speed
                speed = min_cruise_speed
            if speed > max_speed:
                fly["velocity"] *= max_speed / speed
                speed = max_speed
            fly["pos"] += fly["velocity"]

            # Hard boundary is only a safety net. Normal avoidance should
            # happen through wall vision and the connectome before contact.
            if fly["pos"][0] < 2.0:
                fly["pos"][0] = 2.0
                fly["velocity"][0] = abs(fly["velocity"][0]) * 0.25
            elif fly["pos"][0] > screen_width - 2.0:
                fly["pos"][0] = screen_width - 2.0
                fly["velocity"][0] = -abs(fly["velocity"][0]) * 0.25
            if fly["pos"][1] < 2.0:
                fly["pos"][1] = 2.0
                fly["velocity"][1] = abs(fly["velocity"][1]) * 0.25
            elif fly["pos"][1] > screen_height - 2.0:
                fly["pos"][1] = screen_height - 2.0
                fly["velocity"][1] = -abs(fly["velocity"][1]) * 0.25

            speed = float(np.linalg.norm(fly["velocity"]))
            if speed > .25:
                fly["heading"] = math.atan2(fly["velocity"][1], fly["velocity"][0])

            looming_value = max((max(0.0, o["expansion_rate"]) for o in all_visual[fi] if o["visible"]), default=0.0)
            if debug:
                data = {
                    "brain_generation": world.get("debug_generation", 0),
                    "vision": retinas[fi], "visual_objects": all_visual[fi],
                    "sensory": {"looming": float(min(1.0, looming_value / .08)),
                                 "approach_speed": float(looming_value)},
                    "signal": signal, "speed": speed, "brain_ms": brain_ms,
                    "brain_cloud": world["brain_debug_cache"][fi],
                }
                cached_roles = world.get("debug_role_cache")
                if cached_roles and fi < len(cached_roles):
                    data.update(cached_roles[fi])
                else:
                    for role in roles:
                        data[role] = []
                fly["debug_data"] = data
            else:
                fly["debug_data"] = {"speed": speed, "sensory": {"looming": float(min(1.0, looming_value/.08))}, "signal": signal}

            views.append({"pos": tuple(fly["pos"]), "heading": math.degrees(fly["heading"]),
                          "escape": bool(signal["escape"]), "wing_drive": float(signal["wing_drive"]),
                          "debug_data": fly["debug_data"], "id": fly["id"]})

        if world["frame"] % 60 == 0:
            print(f"brain={brain_ms:.2f} ms | " + " | ".join(f"fly={v['id']} speed={v['debug_data']['speed']:.2f} "
                             f"loom={v['debug_data']['sensory']['looming']:.3f} "
                             f"escape={v['debug_data']['signal']['escape_rate']:.3f} "
                             f"walk={v['debug_data']['signal']['walk_drive']:.3f}" for v in views))
        return views
    return step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Desktop fruit fly connectome simulation")
    parser.add_argument("--debug", action="store_true", help="draw visual fields and neural activity")
    parser.add_argument("--flies", type=int, default=1, help="number of independent fly brains (default: 1)")
    args = parser.parse_args()
    if args.flies < 1:
        parser.error("--flies must be at least 1")

    world = build_world(args.flies, debug=args.debug)
    print(f"debug mode: {'on' if args.debug else 'off'}")
    print(f"flies: {args.flies} (independent neural states, batched GPU update)")
    signal.signal(signal.SIGINT, lambda *_: world["app"].quit())
    run_overlay(make_world_step_callback(world), app=world["app"], debug=args.debug, num_flies=args.flies)
