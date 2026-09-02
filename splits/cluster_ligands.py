#!/usr/bin/env python3
"""Task 2 -- two ligand clustering variants over the corpus-tier ligands.

Variant A ``scaffold``      generic (graph-framework) Bemis-Murcko scaffold, taken
                            from ``ligand_scaffold.json`` and recomputed for any
                            ligand missing from it.
Variant B ``butina_<cut>``  Taylor-Butina leader clustering on 2048-bit Morgan
                            (radius 2) fingerprints at several Tanimoto cutoffs.
                            A ligand joins a cluster when its Tanimoto similarity
                            to that cluster's centroid is >= the cutoff.

Both are shipped.  ``butina_0.40`` is the headline variant: scaffold splits are
known to overestimate prospective performance relative to fingerprint clustering,
because two molecules can share a graph framework while differing in every
substituent, and conversely near-identical analogues can sit in different
scaffold classes after a single ring change.

Butina is run over the sparse pair list from ``ligand_similarity.py`` rather than
a dense distance matrix; the semantics are identical to
``rdkit.ML.Cluster.Butina.ClusterData`` with ``reordering=False``.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402


def butina(n: int, pair_i, pair_j, pair_sim, cutoff: float) -> np.ndarray:
    """Taylor-Butina leader clustering. Returns an (n,) array of cluster ids."""
    keep = pair_sim >= cutoff
    ii, jj = pair_i[keep], pair_j[keep]
    # Symmetrise, then build CSR by sorting on the source index.  Do NOT use
    # fancy-index assignment with a running cursor: numpy writes a repeated
    # index only once, which silently corrupts the adjacency.
    src = np.concatenate([ii, jj])
    dst = np.concatenate([jj, ii])
    order = np.argsort(src, kind="stable")
    adj = dst[order].astype(np.int32, copy=False)
    deg = np.bincount(src, minlength=n)
    adj_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=adj_start[1:])

    # Descending neighbour count, ties broken by *descending* index.  This
    # reproduces rdkit.ML.Cluster.Butina.ClusterData(..., reordering=False),
    # whose ``tLists.sort(reverse=True)`` sorts (count, index) pairs descending.
    order = np.lexsort((-np.arange(n), -deg))
    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for c in order:
        if labels[c] != -1:
            continue
        labels[c] = cid
        nbrs = adj[adj_start[c] : adj_start[c + 1]]
        if nbrs.size:
            free = nbrs[labels[nbrs] == -1]
            labels[free] = cid
        cid += 1
    assert (labels >= 0).all()
    return labels


def scaffold_labels(inchikeys, smiles_map, scaffold_map, empty_policy: str):
    """Return (labels, stats). Missing scaffolds are recomputed with RDKit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    stats = {"recomputed": [], "recompute_failed": [], "empty_scaffold": 0}
    raw = []
    for ik in inchikeys:
        s = scaffold_map.get(ik)
        if s is None:
            mol = Chem.MolFromSmiles(smiles_map.get(ik, "")) if smiles_map.get(ik) else None
            if mol is None:
                s = None
            else:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    generic = MurckoScaffold.MakeScaffoldGeneric(core)
                    s = Chem.MolToSmiles(generic)
                    stats["recomputed"].append(ik)
                except Exception:
                    s = None
            if s is None:
                stats["recompute_failed"].append(ik)
        raw.append(s)

    labels_by_key: dict = {}
    out = np.empty(len(inchikeys), dtype=np.int64)
    nxt = 0
    for i, (ik, s) in enumerate(zip(inchikeys, raw)):
        if s is None:
            key = ("__unresolved__", ik)  # never merges with anything else
        elif s == "":
            stats["empty_scaffold"] += 1
            key = "__acyclic__" if empty_policy == "group" else ("__acyclic__", ik)
        else:
            key = s
        if key not in labels_by_key:
            labels_by_key[key] = nxt
            nxt += 1
        out[i] = labels_by_key[key]
    stats["n_clusters"] = nxt
    return out, stats


def cluster_stats(labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    srt = np.sort(sizes)[::-1]
    return {
        "n_clusters": int(sizes.size),
        "n_ligands": int(sizes.sum()),
        "singletons": int((sizes == 1).sum()),
        "singleton_fraction": float((sizes == 1).mean()),
        "singleton_ligand_fraction": float((sizes == 1).sum() / sizes.sum()),
        "largest_cluster": int(srt[0]),
        "top10_cluster_sizes": [int(x) for x in srt[:10]],
        "top10_share_of_ligands": float(srt[:10].sum() / sizes.sum()),
        "median_cluster_size": float(np.median(sizes)),
        "mean_cluster_size": float(sizes.mean()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neighbors", type=Path, default=sf.ART / "ligand_neighbors.npz")
    ap.add_argument("--cutoffs", nargs="*", type=float, default=list(sf.BUTINA_CUTOFFS))
    ap.add_argument(
        "--empty-scaffold-policy", choices=("group", "singleton"), default="group",
        help="acyclic molecules have an empty Murcko scaffold; 'group' follows the "
             "conventional scaffold split and pools them, 'singleton' isolates them",
    )
    ap.add_argument("--out", type=Path, default=sf.ART / "ligand_clusters.parquet")
    ap.add_argument("--report", type=Path, default=sf.ART / "ligand_cluster_report.json")
    ap.add_argument("--smiles-json", type=Path, default=sf.CORPUS_SMILES)
    ap.add_argument("--scaffold-json", type=Path, default=sf.LIGAND_SCAFFOLD)
    args = ap.parse_args(argv)

    import json

    sf.ensure_dirs()
    z = np.load(args.neighbors, allow_pickle=True)
    inchikeys = list(z["inchikey"])
    n = len(inchikeys)
    pi, pj, ps = z["pair_i"], z["pair_j"], z["pair_sim"]
    min_sim = float(z["min_sim"])
    print(f"[lig] {n} ligands, {pi.size} pairs with T >= {min_sim}")

    bad = [c for c in args.cutoffs if c < min_sim - 1e-5]
    if bad:
        raise SystemExit(
            f"cutoffs {bad} are below the precomputed pair floor {min_sim}; "
            f"rerun ligand_similarity.py with --min-sim {min(bad)}"
        )

    smiles_map = json.loads(args.smiles_json.read_text())
    scaffold_map = json.loads(args.scaffold_json.read_text())

    report = {"n_ligands": n, "fp_radius": sf.FP_RADIUS, "fp_bits": sf.FP_BITS,
              "headline_variant": sf.HEADLINE_LIGAND_VARIANT, "variants": {}}
    frames = []

    labs, sstats = scaffold_labels(inchikeys, smiles_map, scaffold_map,
                                   args.empty_scaffold_policy)
    st = cluster_stats(labs)
    st.update({k: v for k, v in sstats.items() if k != "n_clusters"})
    st["empty_scaffold_policy"] = args.empty_scaffold_policy
    report["variants"]["scaffold"] = st
    frames.append(pd.DataFrame({"inchikey": inchikeys, "ligand_variant": "scaffold",
                                "ligand_cluster": ["scaffold:%d" % x for x in labs]}))
    print(f"[lig] scaffold: {st['n_clusters']} clusters, "
          f"singleton frac {st['singleton_fraction']:.3f}, "
          f"top10 share {st['top10_share_of_ligands']:.3f}, "
          f"empty scaffolds {sstats['empty_scaffold']}")

    for c in args.cutoffs:
        name = f"butina_{c:.2f}"
        labs = butina(n, pi, pj, ps, c)
        st = cluster_stats(labs)
        st["cutoff_tanimoto"] = c
        report["variants"][name] = st
        frames.append(pd.DataFrame({"inchikey": inchikeys, "ligand_variant": name,
                                    "ligand_cluster": [f"{name}:{x}" for x in labs]}))
        print(f"[lig] {name}: {st['n_clusters']} clusters, "
              f"singleton frac {st['singleton_fraction']:.3f}, "
              f"top10 share {st['top10_share_of_ligands']:.3f}, "
              f"largest {st['largest_cluster']}")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out, index=False)
    sf.write_json(args.report, report)
    print(f"[lig] wrote {args.out} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
