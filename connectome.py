"""
Loads the male-cns annotation + weight tables into a sparse adjacency
matrix, and tags neuron indices by functional role (visual / central
complex / descending / motor) using regex over the annotation columns.

Schema note: I still haven't been able to open the actual feather files
(no network access to storage.googleapis.com from the sandbox this was
written in). COLUMN_MAP below now follows the snake_case naming
confirmed on the syn-partners table (`body_pre`, `body_post`, etc.) on
the download page, but the annotations/weights files may differ. Load
once with `inspect_columns` and fix COLUMN_MAP / ROLE_PATTERNS before
trusting anything downstream.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

# --- adjust these once you've inspected your actual files ---
COLUMN_MAP = {
    "body_id": "body_id",     # or possibly "bodyId" — check with inspect_columns()
    "cls": "class",
    "type": "type",
    "side": "side",           # expects values like 'L'/'R' or 'left'/'right'
    "pre": "body_pre",        # confirmed name on the syn-partners table; weights table may differ
    "post": "body_post",
    "weight": "weight",
}

ROLE_PATTERNS = {
    "visual": re.compile(r"^(R[1-8]|Mi|Tm|T[1-5]|LC|LPLC|LPC|C[23]|Am)\d*", re.I),
    "central_complex": re.compile(r"^(EPG|PEN|PFN|PFL|FB|EB|hDelta|PFR)\d*", re.I),
    "descending": re.compile(r"^DN[a-z]?\d*", re.I),
    "motor": re.compile(r"(^MN|motor)", re.I),
}


def inspect_columns(annotations_path: str, weights_path: str) -> None:
    ann = pd.read_feather(annotations_path)
    wt = pd.read_feather(weights_path)
    print("annotation columns:", list(ann.columns))
    print("weight columns:", list(wt.columns))


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


def load_connectome(annotations_path: str, weights_path: str, device: str = "cpu") -> Connectome:
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

    # build sparse signed adjacency: rows=pre, cols=post
    pre = wt[COLUMN_MAP["pre"]].map(id_to_index)
    post = wt[COLUMN_MAP["post"]].map(id_to_index)
    valid = pre.notna() & post.notna()
    pre = pre[valid].to_numpy(dtype=np.int64)
    post = post[valid].to_numpy(dtype=np.int64)
    w = wt[COLUMN_MAP["weight"]][valid.to_numpy()].to_numpy(dtype=np.float32)

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
    """CPU fallback: keep only visual/CX/descending/motor neurons plus
    anything within `hops` synapses of them, then re-index."""
    keep = np.zeros(full.n_neurons, dtype=bool)
    for role in ("visual", "central_complex", "descending", "motor"):
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
