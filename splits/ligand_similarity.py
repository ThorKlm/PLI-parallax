#!/usr/bin/env python3
"""All-pairs Morgan/Tanimoto pass over the corpus-tier ligands.

Produces two artefacts consumed by the clustering and the leakage diagnostics:

  * ``ligand_fp.npz``        packed 2048-bit Morgan fingerprints + inchikey order
  * ``ligand_neighbors.npz`` sparse i<j pair list for Tanimoto >= ``--min-sim``
                             plus a dense per-ligand top-K neighbour table

The pair list is what Butina clusters over (a dense 31k x 31k distance matrix is
~3.9 GB in float32 and is not needed).  The top-K table is what turns the
"maximum test-vs-train Tanimoto" diagnostic from an O(n_test x n_train) scan per
split into a lookup, with an exact fallback when a ligand's whole top-K happens
to land in the same fold.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402

_FPS = None  # populated per worker process by fork inheritance


def _init(fps):
    global _FPS
    _FPS = fps


def _row_block(job):
    """Compute one block of rows of the Tanimoto matrix."""
    from rdkit import DataStructs

    lo, hi, min_sim, topk = job
    fps = _FPS
    n = len(fps)
    pi, pj, ps = [], [], []
    tk_idx = np.zeros((hi - lo, topk), dtype=np.int32)
    tk_sim = np.zeros((hi - lo, topk), dtype=np.float32)
    for i in range(lo, hi):
        row = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32)
        row[i] = -1.0  # exclude self
        k = min(topk, n - 1)
        part = np.argpartition(-row, k - 1)[:k]
        part = part[np.argsort(-row[part], kind="stable")]
        tk_idx[i - lo, :k] = part
        tk_sim[i - lo, :k] = row[part]
        if k < topk:
            tk_idx[i - lo, k:] = -1
            tk_sim[i - lo, k:] = -1.0
        hits = np.flatnonzero(row[i + 1 :] >= min_sim) + i + 1
        if hits.size:
            pi.append(np.full(hits.size, i, dtype=np.int32))
            pj.append(hits.astype(np.int32))
            ps.append(row[hits])
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    return (
        cat(pi, np.int32),
        cat(pj, np.int32),
        cat(ps, np.float32),
        lo,
        tk_idx,
        tk_sim,
    )


def build_fingerprints(inchikeys, smiles_map):
    """Return (packed_fps, rdkit_fps, failures)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    gen = sf.morgan_generator()
    fps, keys, failures = [], [], []
    for ik in inchikeys:
        smi = smiles_map.get(ik)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            failures.append(ik)
            continue
        fps.append(gen.GetFingerprint(mol))
        keys.append(ik)
    return keys, fps, failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entities", type=Path, default=sf.ART / "system_entities.parquet")
    ap.add_argument("--min-sim", type=float, default=min(sf.BUTINA_CUTOFFS))
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--fp-out", type=Path, default=sf.ART / "ligand_fp.npz")
    ap.add_argument("--nbr-out", type=Path, default=sf.ART / "ligand_neighbors.npz")
    ap.add_argument("--report", type=Path, default=sf.ART / "ligand_similarity_report.json")
    ap.add_argument("--smiles-json", type=Path, default=sf.CORPUS_SMILES)
    args = ap.parse_args(argv)

    import json

    import pandas as pd

    sf.ensure_dirs()
    ent = pd.read_parquet(args.entities, columns=["inchikey"])
    inchikeys = sorted(ent.inchikey.unique())
    smiles_map = json.loads(args.smiles_json.read_text())
    print(f"[sim] {len(inchikeys)} distinct corpus ligands")

    t0 = time.time()
    keys, fps, failures = build_fingerprints(inchikeys, smiles_map)
    print(f"[sim] fingerprints: {len(keys)} ok, {len(failures)} rdkit failures "
          f"({time.time() - t0:.1f}s)")
    if failures:
        print(f"[sim] WARNING unparsable: {failures[:10]}")

    packed = sf.fps_to_packed(fps)
    np.savez_compressed(args.fp_out, packed=packed, inchikey=np.array(keys, dtype=object))
    print(f"[sim] wrote {args.fp_out} {packed.shape}")

    n = len(fps)
    jobs = [
        (lo, min(lo + args.block, n), args.min_sim, args.topk)
        for lo in range(0, n, args.block)
    ]
    t0 = time.time()
    pi, pj, ps = [], [], []
    tk_idx = np.zeros((n, args.topk), dtype=np.int32)
    tk_sim = np.zeros((n, args.topk), dtype=np.float32)
    done = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(fps,)) as pool:
        for a, b, c, lo, ti, ts in pool.imap_unordered(_row_block, jobs, chunksize=1):
            if a.size:
                pi.append(a)
                pj.append(b)
                ps.append(c)
            tk_idx[lo : lo + ti.shape[0]] = ti
            tk_sim[lo : lo + ts.shape[0]] = ts
            done += 1
            if done % 100 == 0:
                print(f"[sim]   {done}/{len(jobs)} blocks  {time.time() - t0:.0f}s", flush=True)
    pi = np.concatenate(pi) if pi else np.zeros(0, np.int32)
    pj = np.concatenate(pj) if pj else np.zeros(0, np.int32)
    ps = np.concatenate(ps) if ps else np.zeros(0, np.float32)
    order = np.lexsort((pj, pi))
    pi, pj, ps = pi[order], pj[order], ps[order]
    print(f"[sim] all-pairs done in {time.time() - t0:.0f}s; "
          f"{pi.size} pairs with T >= {args.min_sim}")

    np.savez_compressed(
        args.nbr_out,
        pair_i=pi,
        pair_j=pj,
        pair_sim=ps,
        topk_idx=tk_idx,
        topk_sim=tk_sim,
        inchikey=np.array(keys, dtype=object),
        min_sim=np.float32(args.min_sim),
    )
    sf.write_json(
        args.report,
        {
            "n_ligands_requested": len(inchikeys),
            "n_ligands_fingerprinted": n,
            "rdkit_failures": failures,
            "fp_radius": sf.FP_RADIUS,
            "fp_bits": sf.FP_BITS,
            "min_sim": args.min_sim,
            "topk": args.topk,
            "n_pairs_above_min_sim": int(pi.size),
            "pair_density": float(pi.size / (n * (n - 1) / 2)),
            "topk_sim_min_of_last_column": float(np.min(tk_sim[:, -1])),
            "seconds": round(time.time() - t0, 1),
        },
    )
    print(f"[sim] wrote {args.nbr_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
