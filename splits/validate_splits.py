#!/usr/bin/env python3
"""Task 8 -- independent re-validation of every emitted split.

``build_splits.py`` asserts before writing; this script asserts *after* writing,
reading only the published Parquet artefacts.  It is the check to run against a
downloaded deposit, and it exits non-zero on the first leak it finds rather than
reporting a leaky split as usable.

Checks per split:
  * fold labels are exactly {train, val, test, excluded}
  * the fold file covers every corpus system exactly once
  * the family's cluster-crossing contract holds on both axes
  * the recorded fold sizes match the summary index
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402

TRAIN, VAL, TEST, EXCL = "train", "val", "test", "excluded"

DISJOINT_AXES = {
    "C1": (),
    "C2p": ("protein_cluster",),
    "C2l": ("ligand_cluster",),
    "C3": ("protein_cluster", "ligand_cluster"),
    "C3comp": ("protein_cluster", "ligand_cluster"),
}
CONTAINED_AXES = {
    "C1": ("protein_cluster", "ligand_cluster"),
    "C2p": ("ligand_cluster",),
    "C2l": ("protein_cluster",),
    "C3": (),
    "C3comp": (),
}


def validate_one(row, audit_by_ctx) -> list:
    problems = []
    ctx = audit_by_ctx[(row.protein_threshold, row.ligand_variant)]
    fold = pd.read_parquet(row.path)

    if set(fold.fold.unique()) - {TRAIN, VAL, TEST, EXCL}:
        problems.append(f"unexpected fold labels {sorted(set(fold.fold.unique()))}")
    if fold.system_id.duplicated().any():
        problems.append("duplicate system_id in fold file")
    if set(fold.system_id) != set(ctx.system_id):
        problems.append(
            f"fold file covers {len(set(fold.system_id))} systems, audit has {len(ctx)}"
        )
        return problems

    df = ctx.merge(fold, on="system_id", validate="one_to_one")
    sets = {f: {ax: set(df.loc[df.fold == f, ax]) for ax in
                ("protein_cluster", "ligand_cluster")} for f in (TRAIN, VAL, TEST)}

    for ax in DISJOINT_AXES[row.family]:
        for a, b in itertools.combinations((TRAIN, VAL, TEST), 2):
            shared = sets[a][ax] & sets[b][ax]
            if shared:
                problems.append(
                    f"{ax} leak: {len(shared)} cluster(s) shared by {a}/{b}, "
                    f"e.g. {sorted(shared)[:3]}"
                )
    for ax in CONTAINED_AXES[row.family]:
        for f in (VAL, TEST):
            missing = sets[f][ax] - sets[TRAIN][ax]
            if missing:
                problems.append(
                    f"{ax} not seen: {len(missing)} cluster(s) in {f} absent from train"
                )
    for f, col in ((TRAIN, "n_train"), (VAL, "n_val"), (TEST, "n_test"),
                   (EXCL, "n_excluded")):
        got = int((df.fold == f).sum())
        if got != int(getattr(row, col)):
            problems.append(f"{col} mismatch: file={got} index={int(row[col])}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=sf.ART / "split_index.parquet")
    ap.add_argument("--audit", type=Path, default=sf.ART / "split_audit.parquet")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args(argv)

    idx = pd.read_parquet(args.index)
    audit = pd.read_parquet(args.audit)
    audit_by_ctx = {k: v for k, v in audit.groupby(["protein_threshold", "ligand_variant"])}

    n_bad = 0
    for row in idx.itertuples():
        problems = validate_one(row, audit_by_ctx)
        if problems:
            n_bad += 1
            print(f"LEAK {row.split_tag}:")
            for p in problems:
                print(f"      {p}")
            if args.fail_fast:
                raise SystemExit(1)
    if n_bad:
        raise SystemExit(f"FAILED: {n_bad}/{len(idx)} splits violate the leakage contract")
    print(f"OK: all {len(idx)} splits pass zero-cluster-crossing validation on every axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
