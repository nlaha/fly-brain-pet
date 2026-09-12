# fly_pet

Desktop pet driven by the male-cns Drosophila connectome.

## Setup
```
uv sync
uv run data/download.py   # verify URLs against https://male-cns.janelia.org/download/ first
```

Then, once you have the real feather files:
```python
from brain.connectome import inspect_columns
inspect_columns("data/raw/body-annotations-....feather", "data/raw/connectome-weights-....feather")
```
Check the printed column names against `COLUMN_MAP` and the regexes in
`ROLE_PATTERNS` (both in `brain/connectome.py`) — I haven't been able to
verify these against the live files, so they're best-effort guesses at
the real schema and cell-type naming conventions.

Run:
```
uv run main.py
```

## What's real vs. approximated
- **Real**: neuron identities and the full weighted synaptic graph, run
  as a leaky-rate network (`brain/simulator.py`).
- **Approximated**: there's no muscle/biomechanics model in the CNS
  connectome, so "movement" is a hand-built readout — left/right firing
  rate asymmetry in descending + motor neurons becomes turning, mean
  rate becomes forward thrust (`brain/io_mapping.py`). Vision is a
  synthetic drive to visual neurons based on cursor angle/distance, not
  actual photoreceptor-realistic input.
- Neurotransmitter sign (excitatory/inhibitory) isn't wired in yet —
  every synapse currently acts as excitatory. The `neurotransmitter-
  predictions` file listed on the download page has per-neuron
  predicted transmitter; worth folding into `connectome.py` next to
  flip sign on GABA/glutamatergic neurons.

## GPU vs CPU
`main.py` runs the full ~166k-neuron graph on CUDA if available, and
automatically falls back to a curated visual+central-complex+descending
+motor subgraph (`build_curated_subgraph`) on CPU, since a dense-ish
sparse mm over the full graph every frame is not real-time on CPU.
