#!/usr/bin/env python3
"""Fit the per-system reliability field from the deposited contact tables.

Agreement is the mean pairwise Jaccard of the 4 A contact residue sets of chai1,
single-sequence boltz2 and smina; accuracy is the same index against the
experimental set, averaged over the three arms. Isotonic regression of accuracy
on agreement, fitted on half the crystal systems and evaluated on the other half,
then applied to the corpus tier where no experimental set exists.

Was section 4 of repair_smina_crystal_meta.py, which read smina from
smina_v2_b/v3/labels/*.npz. That arm is in the deposit now, so this reads the
deposit only.

    python fit_reliability.py --deposit /path/to/deposit --out .
"""
import argparse, collections, itertools, json, os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold

ARMS = ("chai1", "boltz2", "smina")
FLAG = "contact_4A"


def contacts_path(deposit, arm, tier):
    """Path to one arm's contacts table; the experimental set has its own file name."""
    fn = ("labels_crystal_groundtruth_contacts.parquet" if arm == "groundtruth"
          else f"labels_{arm}_{tier}_contacts.parquet")
    return os.path.join(deposit, "labels", fn)


def contact_sets(path):
    """system_id -> set of res_row asserted at 4 A."""
    t = pq.read_table(path, columns=["system_id", "res_row", FLAG])
    d = t.filter(t[FLAG]).to_pydict()
    out = collections.defaultdict(set)
    for s, r in zip(d["system_id"], d["res_row"]):
        out[s].add(r)
    return dict(out)


def jac(a, b):
    """Jaccard index, undefined rather than zero when both sets are empty."""
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def agreement(sets, ids):
    """Mean pairwise Jaccard over the arms, one value per system."""
    pairs = list(itertools.combinations(sets, 2))
    return np.array([np.mean([jac(sets[x][s], sets[y][s]) for x, y in pairs]) for s in ids])


def accuracy(sets, truth, ids):
    """Mean Jaccard of each arm against the experimental set, one value per system."""
    return np.array([np.mean([jac(sets[n][s], truth[s]) for n in sets]) for s in ids])


def load_tier(deposit, tier, truth=False):
    """Contact sets per arm plus optionally the experimental set, and their common systems."""
    sets = {a: contact_sets(contacts_path(deposit, a, tier)) for a in ARMS}
    cov = [set(v) for v in sets.values()]
    gt = None
    if truth:
        gt = contact_sets(contacts_path(deposit, "groundtruth", tier))
        cov.append(set(gt))
    return sets, gt, sorted(set.intersection(*cov))


def protein_ids(deposit, ids, tier):
    """protein_id per system, from whichever meta tables carry it. None if incomplete."""
    key = {}
    for arm in ARMS:
        p = os.path.join(deposit, "labels", f"labels_{arm}_{tier}_meta.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        if "protein_id" not in t.schema.names:
            continue
        d = t.select(["system_id", "protein_id"]).to_pydict()
        key.update(dict(zip(d["system_id"], d["protein_id"])))
    g = [key.get(s) for s in ids]
    return None if any(v is None for v in g) else np.asarray(g)


def mae(p, y):
    """Mean absolute error."""
    return float(np.mean(np.abs(p - y)))


def cv_mae(x, y, folds, groups=None, seed=0):
    """Out-of-fold MAE of the isotonic map, with folds grouped by protein where given."""
    if groups is None:
        split = KFold(n_splits=folds, shuffle=True, random_state=seed).split(x)
    else:
        split = GroupKFold(n_splits=folds).split(x, groups=groups)
    pred = np.empty_like(y)
    for tr, te in split:
        m = IsotonicRegression(out_of_bounds="clip").fit(x[tr], y[tr])
        pred[te] = m.predict(x[te])
    return mae(pred, y), pred


def check_archive(path, ids, agree):
    """Compare recomputed agreement with the archived intermediate table, where they overlap."""
    ref = {r["system_id"]: r["agree_mean"] for r in json.load(open(path))}
    sh = [i for i, s in enumerate(ids) if s in ref]
    if not sh:
        return None
    d = np.abs(agree[sh] - np.array([ref[ids[i]] for i in sh]))
    return len(sh), float(d.max()), float(np.median(d))


def support(sets, ids, tier):
    """One row per asserted residue, counting how many arms assert it."""
    rows = []
    for s in ids:
        c = collections.Counter()
        for n in sets:
            for r in sets[n][s]:
                c[r] += 1
        rows.extend((s, tier, r, k) for r, k in c.items())
    return pd.DataFrame(rows, columns=["system_id", "tier", "res_row", "n_arms_asserting"])


def main(argv=None):
    """Fit on the crystal tier, transfer to the corpus tier, write the three artefacts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deposit", default=".")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--check", default=None, help="archived agreement json, optional")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    crystal, gt, ik = load_tier(a.deposit, "crystal", truth=True)
    ak, mk = agreement(crystal, ik), accuracy(crystal, gt, ik)
    # Either index is undefined where both sets are empty. Those systems carry no
    # calibration information, so they are dropped rather than imputed.
    keep = np.isfinite(ak) & np.isfinite(mk)
    ids = [s for s, k in zip(ik, keep) if k]
    ak, mk = ak[keep], mk[keep]
    print(f"crystal, three arms and ground truth : {len(ik):,}")
    print(f"  defined on both axes               : {len(ids):,}")

    if a.check:
        r = check_archive(a.check, ids, ak)
        if r:
            print(f"  vs archived table: n={r[0]:,} max {r[1]:.6f} median {r[2]:.6f}")

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ids))
    h = len(ids) // 2
    fi, hi = perm[:h], perm[h:]
    iso = IsotonicRegression(out_of_bounds="clip").fit(ak[fi], mk[fi])
    pred = iso.predict(ak[hi])
    half = float(np.quantile(np.abs(pred - mk[hi]), a.coverage))

    const = float(mk[fi].mean())
    lin = LinearRegression().fit(ak[fi].reshape(-1, 1), mk[fi])
    cvm, cvp = cv_mae(ak, mk, a.folds, seed=a.seed)
    groups = protein_ids(a.deposit, ids, "crystal")
    grouped = cv_mae(ak, mk, a.folds, groups=groups, seed=a.seed)[0] if groups is not None else None
    rho = float(pd.Series(ak).corr(pd.Series(mk), method="spearman"))

    print(f"held-out MAE             {mae(pred, mk[hi]):.4f}")
    print(f"  constant               {mae(np.full(hi.size, const), mk[hi]):.4f}")
    print(f"  linear                 {mae(lin.predict(ak[hi].reshape(-1, 1)), mk[hi]):.4f}")
    print(f"  agreement as accuracy  {mae(ak[hi], mk[hi]):.4f}")
    print(f"cross-validated MAE      {cvm:.4f}" + (f" (grouped {grouped:.4f})" if grouped else ""))
    print(f"spearman                 {rho:.4f}")
    print(f"half-width at {a.coverage:.2f}       {half:.4f} "
          f"(cross-validated {np.quantile(np.abs(cvp - mk), a.coverage):.4f})")

    corpus, _, ic = load_tier(a.deposit, "corpus")
    ac = agreement(corpus, ic)
    ok = np.isfinite(ac)
    ic = [s for s, k in zip(ic, ok) if k]
    ac = ac[ok]
    print(f"corpus, three arms                   : {len(ic):,}")

    rel = pd.concat([
        pd.DataFrame({"system_id": ids, "tier": "crystal", "agreement": ak,
                      "pred_accuracy": iso.predict(ak), "observed_accuracy": mk,
                      "conformal_halfwidth_90": half}),
        pd.DataFrame({"system_id": ic, "tier": "corpus", "agreement": ac,
                      "pred_accuracy": iso.predict(ac), "observed_accuracy": np.nan,
                      "conformal_halfwidth_90": np.nan}),
    ], ignore_index=True)
    rel.to_parquet(f"{a.out}/system_reliability.parquet", index=False)
    print(f"system_reliability.parquet {len(rel):,} rows")

    sup = pd.concat([support(crystal, ids, "crystal"), support(corpus, ic, "corpus")],
                    ignore_index=True)
    sup.to_parquet(f"{a.out}/residue_support.parquet", index=False)
    print(f"residue_support.parquet    {len(sup):,} rows")
    print(sup.groupby(["tier", "n_arms_asserting"]).size().to_string())

    # thresholds too, so the map can be applied without sklearn
    np.savez(os.path.join(a.out, "reliability_fit.npz"),
             fit_agreement=ak[fi], fit_observed=mk[fi],
             cal_agreement=ak[hi], cal_observed=mk[hi],
             iso_x=iso.X_thresholds_, iso_y=iso.y_thresholds_,
             halfwidth=np.array([half]), seed=np.array([a.seed]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
