"""
Loads the male-cns annotation + weight tables into a sparse adjacency
matrix, and tags neuron indices by functional role (visual / central
complex / descending / motor) using regex over the annotation columns.

Schema note (confirmed via inspect_data.py against the real files):
- annotations: `bodyId`, `superclass` (e.g. `descending_neuron`,
  `visual_projection`, `visual_centrifugal`), `type`, `somaSide` (`R`/`L`)
- weights: `body_pre`, `body_post`, `weight` (raw synapse counts, can be
  in the thousands — scaled down in `load_connectome`)
- neurotransmitters: `body`, `consensus_nt` (falls back to `predicted_nt`
  where consensus is null) — now wired in below to sign synapses.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

COLUMN_MAP = {
    "body_id": "bodyId",
    "cls": "superclass",       # descending_neuron / visual_projection / visual_centrifugal / etc.
    "type": "type",            # e.g. DNp01, EPG, LC10
    "side": "somaSide",        # 'R' / 'L'
    "pre": "body_pre",
    "post": "body_post",
    "weight": "weight",        # raw synapse counts, not normalized
    "nt_body_id": "body",      # neurotransmitter table uses a different id column name
    "nt_consensus": "consensus_nt",
    "nt_predicted": "predicted_nt",
}

INHIBITORY_TRANSMITTERS = {"gaba", "glutamate"}

ROLE_PATTERNS = {
    # matched against `superclass`, which uses clean category strings
    "visual": re.compile(r"visual|photoreceptor", re.I),
    "descending": re.compile(r"descending", re.I),
    "motor": re.compile(r"motor", re.I),
    # central complex isn't its own superclass — these are central-complex
    # cell type prefixes instead, matched against `type`
    "central_complex": re.compile(r"^(EPG|PEN|PFN|PFL|FB|EB|hDelta|PFR)\d*", re.I),
    # Identified neurons in the looming -> escape pathway.  Keeping these
    # separate from the broad visual/motor roles lets the I/O layer stimulate
    # and read the circuit by biological identity rather than by a hand-written
    # behavioral rule.
    "looming": re.compile(r"^(?:LC4|LPLC2)(?:\s|$)", re.I),
    "escape": re.compile(r"^DNp01(?:\s|$)", re.I),
    "steering": re.compile(r"^(?:DNa01|DNa02)(?:\s|$)", re.I),
    "forward": re.compile(r"^DNp09(?:\s|$)", re.I),
    "escape_wing": re.compile(r"^(?:DNp02|DNp04|DNp11)(?:\s|$)", re.I),
}


@dataclass
class Connectome:
    n_neurons: int
    body_ids: np.ndarray               # index -> bodyId
    id_to_index: dict                  # bodyId -> index
    roles: dict = field(default_factory=dict)   # role -> bool mask (np.ndarray)
    side: np.ndarray = None            # +1 right, -1 left, 0 unknown
    adjacency: torch.Tensor = None      # sparse [n_neurons, n_neurons], signed weights

    def role_indices(self, role: str) -> np.ndarray:
        return np.nonzero(self.roles[role])[0]


def _sign_from_side(value: str) -> int:
    if not isinstance(value, str):
        return 0
    v = value.lower()
    if v.startswith("r"):
        return 1
    if v.startswith("l"):
        return -1
    return 0


def _build_id_lookup(body_ids: np.ndarray):
    """Vectorized bodyId -> index lookup via searchsorted. Dict.map() on a
    152M-row column is painfully slow; this does the same job in C."""
    sort_order = np.argsort(body_ids)
    sorted_ids = body_ids[sort_order]

    def lookup(values: np.ndarray):
        positions = np.searchsorted(sorted_ids, values)
        positions = np.clip(positions, 0, len(sorted_ids) - 1)
        valid = sorted_ids[positions] == values
        indices = np.full(len(values), -1, dtype=np.int64)
        indices[valid] = sort_order[positions[valid]]
        return indices, valid

    return lookup


def _nt_sign(value) -> float:
    if not isinstance(value, str):
        return 1.0  # unknown -> treat as excitatory (acetylcholine is the common default)
    return -1.0 if value.lower() in INHIBITORY_TRANSMITTERS else 1.0


def load_connectome(
    annotations_path: str,
    weights_path: str,
    nt_path: str | None = None,
    device: str = "cpu",
    weight_scale: float = 1e-3,
) -> Connectome:
    ann = pd.read_feather(annotations_path)
    wt = pd.read_feather(weights_path)

    body_ids = ann[COLUMN_MAP["body_id"]].to_numpy()
    id_to_index = {bid: i for i, bid in enumerate(body_ids)}
    n = len(body_ids)

    roles = {}
    combined_label = (
        ann[COLUMN_MAP["type"]].fillna("").astype(str)
        + " " + ann[COLUMN_MAP["cls"]].fillna("").astype(str)
    )
    for role, pattern in ROLE_PATTERNS.items():
        roles[role] = combined_label.str.contains(pattern).to_numpy()

    side = ann[COLUMN_MAP["side"]].map(_sign_from_side).to_numpy()
    lookup = _build_id_lookup(body_ids)

    # per-neuron excitatory(+1)/inhibitory(-1) sign, defaulting to excitatory
    nt_sign_by_index = np.ones(n, dtype=np.float32)
    if nt_path is not None:
        nt = pd.read_feather(nt_path)
        transmitter = nt[COLUMN_MAP["nt_consensus"]].fillna(nt[COLUMN_MAP["nt_predicted"]])
        nt_index, nt_valid = lookup(nt[COLUMN_MAP["nt_body_id"]].to_numpy())
        signs = transmitter.to_numpy()[nt_valid]
        nt_sign_by_index[nt_index[nt_valid]] = np.array([_nt_sign(s) for s in signs], dtype=np.float32)

    # build sparse signed adjacency: rows=pre, cols=post
    pre_idx, pre_valid = lookup(wt[COLUMN_MAP["pre"]].to_numpy())
    post_idx, post_valid = lookup(wt[COLUMN_MAP["post"]].to_numpy())
    valid = pre_valid & post_valid
    pre = pre_idx[valid]
    post = post_idx[valid]
    w = wt[COLUMN_MAP["weight"]].to_numpy(dtype=np.float32)[valid] * weight_scale
    w *= nt_sign_by_index[pre]  # sign the synapse by its pre-synaptic neuron's transmitter

    indices = torch.tensor(np.stack([post, pre]), dtype=torch.long)  # (post, pre) so adj @ r sums inputs per post
    values = torch.tensor(w, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce().to(device)

    return Connectome(
        n_neurons=n,
        body_ids=body_ids,
        id_to_index=id_to_index,
        roles=roles,
        side=side,
        adjacency=adjacency,
    )


def build_curated_subgraph(full: Connectome, hops: int = 2) -> Connectome:
    """CPU fallback: keep the identified looming/escape/steering circuit
    and anything within `hops` synapses of it, then re-index.

    This is intentionally narrower than the old broad visual/motor cut: the
    I/O layer stimulates LC4/LPLC2 and reads DNp01/DNa01/DNa02/DNp09/DNp02/04/11.
    """
    keep = np.zeros(full.n_neurons, dtype=bool)
    for role in ("looming", "escape", "steering", "forward", "escape_wing"):
        keep |= full.roles[role]

    adj = full.adjacency.coalesce()
    idx = adj.indices()
    for _ in range(hops):
        frontier = np.nonzero(keep)[0]
        frontier_set = torch.tensor(frontier)
        mask = torch.isin(idx[1], frontier_set) | torch.isin(idx[0], frontier_set)
        newly = torch.unique(idx[:, mask]).numpy()
        keep[newly] = True

    keep_idx = np.nonzero(keep)[0]
    remap = {old: new for new, old in enumerate(keep_idx)}

    row_ok = torch.isin(idx[0], torch.tensor(keep_idx))
    col_ok = torch.isin(idx[1], torch.tensor(keep_idx))
    both_ok = row_ok & col_ok
    sub_idx = idx[:, both_ok].numpy()
    sub_vals = adj.values()[both_ok]
    remapped = np.vectorize(remap.get)(sub_idx)

    n_sub = len(keep_idx)
    sub_adjacency = torch.sparse_coo_tensor(
        torch.tensor(remapped, dtype=torch.long), sub_vals, size=(n_sub, n_sub)
    ).coalesce()

    return Connectome(
        n_neurons=n_sub,
        body_ids=full.body_ids[keep_idx],
        id_to_index={full.body_ids[old]: new for old, new in remap.items()},
        roles={role: mask[keep_idx] for role, mask in full.roles.items()},
        side=full.side[keep_idx],
        adjacency=sub_adjacency,
    )
