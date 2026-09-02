#!/usr/bin/env python3
"""Check that the sparse Butina implementation reproduces RDKit's reference exactly.

``cluster_ligands.butina`` runs over a thresholded pair list because a dense
31k x 31k distance matrix is not practical, so its equivalence to
``rdkit.ML.Cluster.Butina.ClusterData(..., reordering=False)`` is verified here on
a subsample small enough for the dense reference.  Run this before trusting a
regenerated clustering.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402
from cluster_ligands import butina  # noqa: E402


def canonical(labels) -> tuple:
    seen: dict = {}
    out = []
    for x in labels:
        if x not in seen:
            seen[x] = len(seen)
        out.append(seen[x])
    return tuple(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="ligands in the subsample")
    ap.add_argument("--cutoffs", nargs="*", type=float, default=list(sf.BUTINA_CUTOFFS))
    args = ap.parse_args(argv)

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.ML.Cluster import Butina

    RDLogger.DisableLog("rdApp.*")
    smiles = json.loads(sf.CORPUS_SMILES.read_text())
    keys = sorted(smiles)[: args.n]
    gen = sf.morgan_generator()
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(smiles[k])) for k in keys]
    n = len(fps)

    floor = min(args.cutoffs) - 0.05
    dists, pairs = [], []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
        pairs.extend((j, i, s) for j, s in enumerate(sims) if s >= floor)
    pi = np.array([p[0] for p in pairs], dtype=np.int32)
    pj = np.array([p[1] for p in pairs], dtype=np.int32)
    ps = np.array([p[2] for p in pairs], dtype=np.float32)

    failures = 0
    for cut in args.cutoffs:
        ref_clusters = Butina.ClusterData(dists, n, 1.0 - cut, isDistData=True,
                                          reordering=False)
        ref = np.empty(n, dtype=np.int64)
        for ci, members in enumerate(ref_clusters):
            for m in members:
                ref[m] = ci
        mine = butina(n, pi, pj, ps, cut)
        ok = canonical(ref) == canonical(mine)
        failures += not ok
        print(f"cutoff={cut:.2f}  rdkit={len(ref_clusters)} clusters  "
              f"sparse={len(set(mine.tolist()))} clusters  identical={ok}")
    if failures:
        raise SystemExit(f"FAIL: {failures} cutoff(s) disagree with the RDKit reference")
    print("OK: sparse Butina matches the RDKit reference at every cutoff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
