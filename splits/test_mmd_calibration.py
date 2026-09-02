#!/usr/bin/env python3
"""Calibrate the MMD permutation test on a true null.

Splits a random sample of corpus ligands into two halves that differ only by the
draw, so any non-uniformity in the p-values would be a bug in the estimator or in
the permutation scheme rather than a real distribution shift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402
from measure_shift import mmd_permutation_test, tanimoto_kernel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--n-per-side", type=int, default=1500)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    z = np.load(sf.ART / "ligand_fp.npz", allow_pickle=True)
    bits = np.unpackbits(z["packed"], axis=1)
    rng = np.random.default_rng(args.seed)
    m = args.n_per_side
    ps = []
    for _ in range(args.trials):
        idx = rng.choice(len(bits), 2 * m, replace=False)
        K = tanimoto_kernel(bits[idx], bits[idx])
        _, p = mmd_permutation_test(K, m, args.n_perm, rng)
        ps.append(p)
    ps = np.array(ps)
    rate = float((ps < 0.05).mean())
    print(f"trials={args.trials}  mean p={ps.mean():.3f} (expect ~0.5)  "
          f"frac p<0.05={rate:.3f} (expect ~0.05)")
    if not (0.30 <= ps.mean() <= 0.70) or rate > 0.25:
        raise SystemExit("FAIL: permutation null is not calibrated")
    print("OK: MMD permutation test is calibrated on the null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
