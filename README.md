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
uv run main.py               # X11 / most desktops
QT_QPA_PLATFORM=xcb uv run main.py   # Wayland — see note below
```

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

## Running on Wayland
Native Wayland does not let a client read the global cursor position
while also being click-through — that's an intentional Wayland
restriction (X11 allows both at once via `XQueryPointer` + the shape
extension; Wayland's security model forbids the former for
surfaces that aren't receiving input). So this needs to run through
XWayland:
```
QT_QPA_PLATFORM=xcb uv run main.py
```
If you still hit an Xlib/auth error with that set, XWayland's Xauthority
isn't being picked up from your shell — check `echo $DISPLAY` and
`echo $XAUTHORITY` in the same terminal you're launching from, and
export them explicitly if they're empty (GNOME's XWayland auth file is
usually under `/run/user/$(id -u)/.mutter-Xwaylandauth.*`).

## Known rough edges to expect on first run
- First load will still take a bit: reading a 1.1GB feather file and
  building a ~150M-entry sparse tensor isn't instant, even with the
  vectorized id lookups.
- `build_curated_subgraph`'s BFS hop-expansion does a fresh `torch.isin`
  scan over all ~152M edges per hop — fine once, but don't call it
  repeatedly in a loop.
- The overlay sprite is a placeholder (three ellipses drawn in
  `pet/overlay.py`) — swap in a real sprite whenever.
