#!/usr/bin/env python3
"""Tasks 3, 4, 6, 7, 8 -- build the leakage-controlled split family.

Taxonomy (Park & Marcotte pair-input classes)::

    C1      both protein and ligand seen in training
    C2p     protein unseen, ligand seen        (protein-cold)
    C2l     ligand unseen, protein seen        (ligand-cold)
    C3      neither seen                       (joint-cold)

"Seen" / "unseen" are evaluated at *cluster* level, not exact-entity level.  That
is forced by the corpus: 98.0 percent of corpus ligands occur in exactly one
system, so an exact-identity "ligand seen" condition would cap the warm classes
at a few hundred systems.  Cluster-level semantics are also the only definition
consistent with the leakage guarantee, which is itself stated over clusters.  The
summary still reports what fraction of each test set satisfies the strict
exact-entity condition.

Two joint-cold constructions are emitted:

``C3comp``  the literal specification -- bipartite connected components over
            (protein cluster, ligand cluster) nodes with systems as edges, whole
            components assigned to folds.  **This does not balance on this
            corpus.**  Although neither axis has a hub (largest protein cluster
            = 2.0 percent of systems, largest ligand cluster <= 1.1 percent), the
            bipartite graph percolates and a single giant component carries
            77-99 percent of all systems depending on the ligand cutoff.  Absence
            of hub domination on each axis does not imply small components.
            Emitted for completeness and to document the percolation curve.

``C3``      the shipped joint-cold split -- an independent *block* partition of
            the protein-cluster axis and the ligand-cluster axis.  A system is
            kept only when both of its clusters fall in the same fold; the
            off-diagonal blocks are discarded.  Axis fractions are set to
            sqrt(target) / sum(sqrt(target)) so the retained diagonal blocks land
            on the requested 80/10/10 ratio.  Zero cluster crossing holds on both
            axes by construction.

Every emitted split is asserted before it is written; a violation raises
``LeakageError`` and no file is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402

TRAIN, VAL, TEST, EXCL = "train", "val", "test", "excluded"


class LeakageError(AssertionError):
    """Raised when an emitted split would leak clusters across folds."""


def stable_seed(family: str, p_thr: str, l_var: str, seed: int) -> int:
    """Process-independent RNG seed for one split configuration."""
    payload = f"{family}|{p_thr}|{l_var}|{seed}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**32)


# ------------------------------------------------------------- geometry ----
def bipartite_components(pc_codes: np.ndarray, lc_codes: np.ndarray, n_pc: int, n_lc: int):
    """Connected components of the bipartite protein-cluster / ligand-cluster graph.

    Returns (component_of_system, component_of_pc, component_of_lc).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = n_pc + n_lc
    rows = pc_codes
    cols = lc_codes + n_pc
    g = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
    )
    _, labels = connected_components(g, directed=False)
    return labels[pc_codes], labels[:n_pc], labels[n_pc:]


def axis_fractions(fractions: dict) -> dict:
    """Per-axis fractions whose diagonal blocks reproduce ``fractions``."""
    r = {k: np.sqrt(v) for k, v in fractions.items()}
    s = sum(r.values())
    return {k: v / s for k, v in r.items()}


# ------------------------------------------------------------ families ----
def assign_c1(sysdf, rng, fractions):
    """Random system-level split constrained so every fold-out cluster stays seen."""
    n = len(sysdf)
    pc = sysdf.pc_code.to_numpy()
    lc = sysdf.lc_code.to_numpy()
    in_train_p = np.bincount(pc, minlength=pc.max() + 1)
    in_train_l = np.bincount(lc, minlength=lc.max() + 1)

    order = rng.permutation(n)
    n_test = int(round(fractions[TEST] * n))
    n_val = int(round(fractions[VAL] * n))
    desired = np.full(n, TRAIN, dtype=object)
    desired[order[:n_test]] = TEST
    desired[order[n_test : n_test + n_val]] = VAL

    fold = np.full(n, TRAIN, dtype=object)
    for i in order:
        want = desired[i]
        if want == TRAIN:
            continue
        # Moving i out of train is allowed only if both of its clusters keep a
        # representative behind, which is exactly the "seen in training" rule.
        if in_train_p[pc[i]] > 1 and in_train_l[lc[i]] > 1:
            fold[i] = want
            in_train_p[pc[i]] -= 1
            in_train_l[lc[i]] -= 1
    return pd.Series(fold, index=sysdf.index)


def _cold_axis(sysdf, rng, fractions, cold: str, warm: str):
    """Shared body for C2p (cold='pc') and C2l (cold='lc')."""
    cold_code = sysdf[f"{cold}_code"].to_numpy()
    warm_code = sysdf[f"{warm}_code"].to_numpy()
    n_cold = cold_code.max() + 1
    sizes = np.bincount(cold_code, minlength=n_cold)
    part = sf.randomized_greedy_partition(sizes, fractions, rng)
    names = list(fractions)
    cold_fold = np.array(names)[part]

    sys_fold = cold_fold[cold_code]
    train_mask = sys_fold == TRAIN
    warm_seen = np.zeros(warm_code.max() + 1, dtype=bool)
    warm_seen[np.unique(warm_code[train_mask])] = True

    out = np.where(train_mask, TRAIN, EXCL).astype(object)
    for f in (VAL, TEST):
        m = (sys_fold == f) & warm_seen[warm_code]
        out[m] = f
    return pd.Series(out, index=sysdf.index)


def assign_c2p(sysdf, rng, fractions):
    return _cold_axis(sysdf, rng, fractions, cold="pc", warm="lc")


def assign_c2l(sysdf, rng, fractions):
    return _cold_axis(sysdf, rng, fractions, cold="lc", warm="pc")


def assign_c3_block(sysdf, rng, fractions):
    """Independent block partition of both cluster axes; off-diagonal discarded."""
    af = axis_fractions(fractions)
    names = list(fractions)
    pc = sysdf.pc_code.to_numpy()
    lc = sysdf.lc_code.to_numpy()
    p_part = np.array(names)[
        sf.randomized_greedy_partition(np.bincount(pc, minlength=pc.max() + 1), af, rng)
    ]
    l_part = np.array(names)[
        sf.randomized_greedy_partition(np.bincount(lc, minlength=lc.max() + 1), af, rng)
    ]
    pf, lf = p_part[pc], l_part[lc]
    return pd.Series(np.where(pf == lf, pf, EXCL).astype(object), index=sysdf.index)


def assign_c3_component(sysdf, rng, fractions):
    """Literal specification: whole bipartite connected components to folds."""
    comp = sysdf.component.to_numpy()
    sizes = np.bincount(comp, minlength=comp.max() + 1)
    part = sf.randomized_greedy_partition(sizes, fractions, rng)
    names = np.array(list(fractions))
    return pd.Series(names[part][comp], index=sysdf.index)


ASSIGNERS = {
    "C1": assign_c1,
    "C2p": assign_c2p,
    "C2l": assign_c2l,
    "C3": assign_c3_block,
    "C3comp": assign_c3_component,
}


# ---------------------------------------------------------- validation ----
def assert_split(family: str, sysdf: pd.DataFrame, fold: pd.Series) -> dict:
    """Assert the leakage contract for ``family``; raise LeakageError on violation."""
    tag = f"[{family}]"
    if set(fold.unique()) - {TRAIN, VAL, TEST, EXCL}:
        raise LeakageError(f"{tag} unexpected fold labels {set(fold.unique())}")
    if len(fold) != len(sysdf):
        raise LeakageError(f"{tag} fold vector length mismatch")

    sets = {}
    for f in (TRAIN, VAL, TEST):
        m = (fold == f).to_numpy()
        sets[f] = {
            "pc": set(sysdf.pc_code.to_numpy()[m].tolist()),
            "lc": set(sysdf.lc_code.to_numpy()[m].tolist()),
        }
    if not len(sets[TEST]["pc"]):
        raise LeakageError(f"{tag} empty test fold")

    def disjoint(axis):
        for a, b in itertools.combinations((TRAIN, VAL, TEST), 2):
            common = sets[a][axis] & sets[b][axis]
            if common:
                raise LeakageError(
                    f"{tag} {axis} clusters shared between {a} and {b}: "
                    f"{len(common)} clusters, e.g. {sorted(common)[:5]}"
                )

    def contained(axis):
        for f in (VAL, TEST):
            missing = sets[f][axis] - sets[TRAIN][axis]
            if missing:
                raise LeakageError(
                    f"{tag} {len(missing)} {axis} clusters in {f} are absent from train "
                    f"(the '{axis} seen' condition is violated)"
                )

    if family == "C1":
        contained("pc")
        contained("lc")
    elif family == "C2p":
        disjoint("pc")
        contained("lc")
    elif family == "C2l":
        disjoint("lc")
        contained("pc")
    elif family in ("C3", "C3comp"):
        disjoint("pc")
        disjoint("lc")
    else:
        raise LeakageError(f"{tag} unknown family")

    return {
        "n_train": int((fold == TRAIN).sum()),
        "n_val": int((fold == VAL).sum()),
        "n_test": int((fold == TEST).sum()),
        "n_excluded": int((fold == EXCL).sum()),
        "n_train_protein_clusters": len(sets[TRAIN]["pc"]),
        "n_test_protein_clusters": len(sets[TEST]["pc"]),
        "n_train_ligand_clusters": len(sets[TRAIN]["lc"]),
        "n_test_ligand_clusters": len(sets[TEST]["lc"]),
    }


def exact_entity_stats(sysdf: pd.DataFrame, fold: pd.Series) -> dict:
    """How much of the test fold satisfies the *strict* exact-entity conditions."""
    tr = sysdf[(fold == TRAIN).to_numpy()]
    te = sysdf[(fold == TEST).to_numpy()]
    if not len(te):
        return {}
    tp, tl = set(tr.accession), set(tr.inchikey)
    return {
        "test_frac_exact_protein_seen": float(te.accession.isin(tp).mean()),
        "test_frac_exact_ligand_seen": float(te.inchikey.isin(tl).mean()),
    }


# ---------------------------------------------------------------- main ----
def build_context(ent, pcl, lcl, p_thr: str, l_var: str) -> pd.DataFrame:
    pmap = pcl[pcl.protein_threshold == p_thr].set_index("accession").protein_cluster
    lmap = lcl[lcl.ligand_variant == l_var].set_index("inchikey").ligand_cluster
    df = ent[["system_id", "accession", "inchikey"]].copy()
    df["protein_cluster"] = df.accession.map(pmap)
    df["ligand_cluster"] = df.inchikey.map(lmap)
    if df.protein_cluster.isna().any() or df.ligand_cluster.isna().any():
        raise RuntimeError(
            f"unmapped entities for p={p_thr} l={l_var}: "
            f"{int(df.protein_cluster.isna().sum())} proteins, "
            f"{int(df.ligand_cluster.isna().sum())} ligands"
        )
    df["pc_code"], _ = pd.factorize(df.protein_cluster)
    df["lc_code"], _ = pd.factorize(df.ligand_cluster)
    comp, _, _ = bipartite_components(
        df.pc_code.to_numpy(), df.lc_code.to_numpy(),
        int(df.pc_code.max()) + 1, int(df.lc_code.max()) + 1,
    )
    # densify component ids so bincount stays small
    df["component"], _ = pd.factorize(comp)
    return df


def component_stats(df: pd.DataFrame) -> dict:
    sizes = np.bincount(df.component.to_numpy())
    sizes = np.sort(sizes[sizes > 0])[::-1]
    n = int(sizes.sum())
    return {
        "n_components": int(sizes.size),
        "largest_component_systems": int(sizes[0]),
        "largest_component_fraction": float(sizes[0] / n),
        "top5_component_sizes": [int(x) for x in sizes[:5]],
        "singleton_components": int((sizes == 1).sum()),
        "median_component_size": float(np.median(sizes)),
        "components_covering_90pct": int(np.searchsorted(np.cumsum(sizes), 0.9 * n) + 1),
        "balanceable": bool(sizes[0] / n <= 0.85),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=list(ASSIGNERS))
    ap.add_argument("--protein-thresholds", nargs="*", default=list(sf.PROTEIN_THRESHOLDS))
    ap.add_argument("--ligand-variants", nargs="*", default=list(sf.LIGAND_VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(sf.SEEDS))
    ap.add_argument("--train-frac", type=float, default=sf.FOLD_FRACTIONS["train"])
    ap.add_argument("--val-frac", type=float, default=sf.FOLD_FRACTIONS["val"])
    ap.add_argument("--test-frac", type=float, default=sf.FOLD_FRACTIONS["test"])
    ap.add_argument("--folds-dir", type=Path, default=sf.FOLDS)
    ap.add_argument("--audit-out", type=Path, default=sf.ART / "split_audit.parquet")
    ap.add_argument("--index-out", type=Path, default=sf.ART / "split_index.parquet")
    ap.add_argument("--report", type=Path, default=sf.ART / "build_splits_report.json")
    ap.add_argument("--entities", type=Path, default=sf.ART / "system_entities.parquet")
    ap.add_argument("--protein-clusters", type=Path, default=sf.ART / "protein_clusters.parquet")
    ap.add_argument("--ligand-clusters", type=Path, default=sf.ART / "ligand_clusters.parquet")
    args = ap.parse_args(argv)

    fractions = {TRAIN: args.train_frac, VAL: args.val_frac, TEST: args.test_frac}
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SystemExit(f"fold fractions must sum to 1, got {fractions}")

    sf.ensure_dirs()
    args.folds_dir.mkdir(parents=True, exist_ok=True)
    ent = pd.read_parquet(args.entities)
    pcl = pd.read_parquet(args.protein_clusters)
    lcl = pd.read_parquet(args.ligand_clusters)

    audit_frames, index_rows = [], []
    report = {"fold_fractions": fractions, "axis_fractions_C3": axis_fractions(fractions),
              "contexts": {}}

    for p_thr, l_var in itertools.product(args.protein_thresholds, args.ligand_variants):
        ctx = build_context(ent, pcl, lcl, p_thr, l_var)
        cstats = component_stats(ctx)
        key = f"p{p_thr}|{l_var}"
        report["contexts"][key] = {
            "n_protein_clusters": int(ctx.pc_code.nunique()),
            "n_ligand_clusters": int(ctx.lc_code.nunique()),
            "components": cstats,
        }
        print(f"[split] {key}: {ctx.pc_code.nunique()} pc, {ctx.lc_code.nunique()} lc, "
              f"{cstats['n_components']} components, largest "
              f"{cstats['largest_component_fraction']:.1%} of systems", flush=True)

        audit_frames.append(
            ctx.assign(protein_threshold=p_thr, ligand_variant=l_var)[
                ["protein_threshold", "ligand_variant", "system_id", "accession",
                 "inchikey", "protein_cluster", "ligand_cluster", "component"]
            ]
        )

        for family in args.families:
            for seed in args.seeds:
                # Deterministic across processes: Python's str hash is salted
                # per interpreter, so hash() must not be used for a seed that a
                # published deposit has to reproduce.
                rng = np.random.default_rng(stable_seed(family, p_thr, l_var, seed))
                fold = ASSIGNERS[family](ctx, rng, fractions)
                stats = assert_split(family, ctx, fold)  # raises on leakage
                stats.update(exact_entity_stats(ctx, fold))
                tag = sf.split_tag(family, p_thr, l_var, seed)
                out = args.folds_dir / f"{tag}.parquet"
                pd.DataFrame({"system_id": ctx.system_id.to_numpy(),
                              "fold": fold.to_numpy()}).to_parquet(out, index=False)
                row = {"split_tag": tag, "family": family, "family_desc":
                       sf.FAMILY_DESC.get(family, "joint-cold, whole-component assignment"),
                       "protein_threshold": p_thr, "ligand_variant": l_var, "seed": seed,
                       "path": str(out), "n_systems": len(ctx)}
                row.update(stats)
                row["n_components"] = cstats["n_components"]
                row["largest_component_fraction"] = cstats["largest_component_fraction"]
                index_rows.append(row)

    audit = pd.concat(audit_frames, ignore_index=True)
    audit.to_parquet(args.audit_out, index=False)
    idx = pd.DataFrame(index_rows)
    idx.to_parquet(args.index_out, index=False)
    sf.write_json(args.report, report)
    print(f"[split] emitted {len(idx)} splits -> {args.folds_dir}")
    print(f"[split] audit table {audit.shape} -> {args.audit_out}")

    bad = idx[idx.n_test < 0.5 * args.test_frac * idx.n_systems]
    if len(bad):
        print(f"[split] NOTE: {len(bad)} configuration(s) reached under half the "
              f"requested test fraction; see the summary table.")
        print(bad.groupby("family").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
