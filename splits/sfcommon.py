"""Shared helpers for the PocketPoseLibrary corpus-tier leakage-controlled split family.

Every script under /workspace/splits imports from here so that paths, fold
fractions and the fingerprint definition are declared exactly once.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths ----
WORKSPACE = Path(os.environ.get("PPL_WORKSPACE", "/workspace"))
SPLITS = WORKSPACE / "splits"
ART = SPLITS / "artifacts"
FOLDS = SPLITS / "folds"
REPORTS = WORKSPACE / "reports"

DEPOSIT = WORKSPACE / "deposit_v3"
LABELS = DEPOSIT / "labels"
LIGAND_IDENTITY = DEPOSIT / "metadata" / "ligand_identity.json"
PAIRS_ORDER = WORKSPACE / "docking" / "output" / "pairs_order.tsv"
CORPUS_FASTA = WORKSPACE / "corpus_proteins.fasta"
CORPUS_SMILES = WORKSPACE / "corpus_smiles.json"
LIGAND_SCAFFOLD = WORKSPACE / "ligand_scaffold.json"
MMSEQS = WORKSPACE / "mmseqs" / "bin" / "mmseqs"

# Protein clusterings that already exist in the workspace, keyed by min-seq-id.
PRECOMPUTED_PROTCLUST = {
    "0.30": WORKSPACE / "protclust30_cluster.tsv",
    "0.40": WORKSPACE / "protclust40_cluster.tsv",
}

# ------------------------------------------------------------ constants ----
#: Morgan fingerprint definition used for every ligand computation here.
FP_RADIUS = 2
FP_BITS = 2048

#: Protein identity thresholds swept (mmseqs --min-seq-id, -c 0.8, cluster-mode 2).
PROTEIN_THRESHOLDS = ("0.30", "0.40", "0.50")

#: Butina Tanimoto *similarity* cutoffs swept.  A member joins a cluster when its
#: similarity to the cluster centroid is >= the cutoff, i.e. Butina distance
#: cutoff = 1 - cutoff.
BUTINA_CUTOFFS = (0.30, 0.40, 0.50, 0.60)

#: Ligand clustering variants shipped.  ``butina_0.40`` is the headline.
LIGAND_VARIANTS = ("scaffold",) + tuple(f"butina_{c:.2f}" for c in BUTINA_CUTOFFS)
HEADLINE_LIGAND_VARIANT = "butina_0.40"
HEADLINE_PROTEIN_THRESHOLD = "0.30"

FAMILIES = ("C1", "C2p", "C2l", "C3")
FAMILY_DESC = {
    "C1": "both protein and ligand seen in training",
    "C2p": "protein unseen, ligand seen (protein-cold)",
    "C2l": "ligand unseen, protein seen (ligand-cold)",
    "C3": "neither seen (joint-cold)",
}

SEEDS = (0, 1, 2, 3, 4)
FOLD_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
FOLD_NAMES = ("train", "val", "test")
EXCLUDED = "excluded"


# ------------------------------------------------------------- helpers ----
def ensure_dirs() -> None:
    for d in (SPLITS, ART, FOLDS, SPLITS / "logs"):
        d.mkdir(parents=True, exist_ok=True)


def split_tag(family: str, protein_threshold: str, ligand_variant: str, seed: int) -> str:
    """Stable filename stem for one split configuration."""
    return f"{family}__p{protein_threshold}__{ligand_variant}__seed{seed}"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_fasta(path: Path) -> dict:
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def morgan_generator():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def fps_to_packed(fps) -> np.ndarray:
    """Pack a list of RDKit ExplicitBitVect into a (n, FP_BITS//8) uint8 array."""
    from rdkit import DataStructs

    out = np.zeros((len(fps), FP_BITS // 8), dtype=np.uint8)
    tmp = np.zeros((FP_BITS,), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, tmp)
        out[i] = np.packbits(tmp)
    return out


def unpack_bits(packed: np.ndarray) -> np.ndarray:
    """(n, FP_BITS//8) uint8 -> (n, FP_BITS) uint8 0/1 array."""
    return np.unpackbits(packed, axis=1)


def randomized_greedy_partition(sizes: np.ndarray, fractions: dict, rng) -> np.ndarray:
    """Assign indivisible groups (clusters/components) to folds balancing *system* counts.

    Groups are visited largest-first (ties shuffled by ``rng``) and each is placed
    in the fold with the largest remaining deficit relative to its target.  This is
    the standard greedy multiway number-partitioning heuristic; the seed enters only
    through the tie-shuffle and the deficit tie-break, which is exactly the
    "varying only the assignment of clusters to folds" requirement.

    Returns an array of fold indices into ``tuple(fractions)``.
    """
    names = list(fractions)
    total = float(sizes.sum())
    targets = np.array([fractions[n] * total for n in names], dtype=float)
    order = rng.permutation(len(sizes))
    order = order[np.argsort(-sizes[order], kind="stable")]
    assigned = np.zeros(len(names), dtype=float)
    out = np.empty(len(sizes), dtype=np.int64)
    for gi in order:
        deficit = targets - assigned
        best = np.flatnonzero(deficit >= deficit.max() - 1e-9)
        pick = int(best[0]) if len(best) == 1 else int(rng.choice(best))
        out[gi] = pick
        assigned[pick] += sizes[gi]
    return out
