# fly_pet

Desktop pet driven by the male-cns Drosophila connectome.

## Setup
```
uv sync
uv run data/download.py   # ~1.2 GB total, mostly connectome-weights
```

Layout expected:
```
main.py
pyproject.toml
data/{download.py, raw/*.feather}
brain/{__init__.py, connectome.py, simulator.py, io_mapping.py}
pet/{__init__.py, overlay.py}
```

Schema is already confirmed against the real files (see `COLUMN_MAP` in
`brain/connectome.py`) — `inspect_data.py` is there if you re-download a
newer dataset version and it changes.

Run:
```
uv run main.py
```
(add yourself to the `input` group first if you're on Wayland — see below)

## What's real vs. approximated
- **Real**: neuron identities, the full weighted synaptic graph, and
  excitatory/inhibitory sign (from `body-neurotransmitters-*.feather`'s
  `consensus_nt`, falling back to `predicted_nt` — GABA/glutamate flip
  the synapse to inhibitory, everything else stays excitatory), all run
  as a leaky-rate network (`brain/simulator.py`).
- **Approximated**: there's no muscle/biomechanics model in the CNS
  connectome, so "movement" is a hand-built readout — left/right firing
  rate asymmetry in descending + motor neurons becomes turning, mean
  rate becomes forward thrust (`brain/io_mapping.py`). Vision is a
  synthetic drive to visual neurons based on cursor angle/distance, not
  actual photoreceptor-realistic input.
- Raw synapse-count weights run into the thousands; `load_connectome`
  scales them by `weight_scale` (default `1e-3`) before building the
  adjacency so the leaky-rate sim doesn't immediately saturate. This is
  a guess, not a calibrated value — tune it if the pet twitches
  erratically or barely reacts at all.

## GPU vs CPU
`main.py` runs the full ~211k-neuron graph on CUDA if available, and
automatically falls back to a curated visual+central-complex+descending
+motor subgraph (`build_curated_subgraph`) on CPU, since a dense-ish
sparse mm over the full graph every frame is not real-time on CPU.

## Cursor tracking on Wayland
Native Wayland deliberately blocks any client from reading the global
pointer position — that's not a bug, it's the security model (X11
allows it via `XQueryPointer`; Wayland doesn't have an equivalent for
surfaces that aren't receiving input). `pet/cursor.py` sidesteps this
by reading raw mouse motion from `/dev/input/event*` via `evdev`,
which works identically under Wayland, X11, or any compositor.

This needs read access to those device files — add yourself to the
`input` group once:
```
sudo usermod -aG input $USER
```
then **log out and back in** (group membership doesn't apply to an
already-running session). If that's not done yet, `main.py` will print
a permission warning and fall back to Qt's `QCursor`, which only works
if the whole app runs through XWayland:
```
QT_QPA_PLATFORM=xcb uv run main.py
```

Note the evdev path only tracks *relative* motion (accumulated from
mouse deltas, starting at screen center) — it's a virtual position, not
a true query of where the OS cursor currently is, so it can drift from
reality if the real cursor gets warped by something else (e.g. moving
across multiple monitors with different scaling). Good enough for
"something for the fly to react to"; not perfectly ground-truth.

## Known rough edges to expect on first run
- First load will still take a bit: reading a 1.1GB feather file and
  building a ~150M-entry sparse tensor isn't instant, even with the
  vectorized id lookups.
- `build_curated_subgraph`'s BFS hop-expansion does a fresh `torch.isin`
  scan over all ~152M edges per hop — fine once, but don't call it
  repeatedly in a loop.
- The overlay sprite is a placeholder (three ellipses drawn in
  `pet/overlay.py`) — swap in a real sprite whenever.

## Looming-driven escape

The fly does **not** contain a scripted `cursor_near -> escape` behavior rule.
The cursor is converted into a directional looming signal and injected into the
identified **LC4/LPLC2** visual population. The connectome is then simulated;
**DNp01 / Giant Fiber** activity is decoded as the escape command. A Giant
Fiber threshold crossing gives the body a flight impulse, while the desktop
physics layer handles velocity, drag, and screen boundaries.

This follows the important architectural idea in DenisSergeevitch's
DesktopFly: cursor approach drives the real looming pathway and the escape
command comes from the simulated Giant Fiber, rather than a hand-written
proximity state machine. See the upstream project for the biological circuit
selection and rationale.
