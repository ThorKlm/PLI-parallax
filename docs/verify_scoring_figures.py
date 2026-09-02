#!/usr/bin/env python
"""Recompute the quoted figures that live in the scoring layer rather than the
deposit: PoseBusters rates, symmetry-corrected RMSD, contact accuracy, the
training-cutoff split and the teacher-independence correlations.

Companion to verify_quoted_figures.py, which covers the deposit. Three outcomes
per claim: PASS or FAIL for anything recomputed against a quoted value, VALUE
for a quantity recomputed and reported without a target, and MANUAL for a number
the manuscript quotes that no computation here reaches, listed with its source.

Read-only. Runs on the primary, C.40516683."""
import glob, json, os, sys, collections
import numpy as np

PB   = "/workspace/posebusters"
BISY = "/workspace/datasets/experimental_expansion/bisy_v2"
REP  = "/workspace/reports"
LAB  = "/workspace/deposit_v3/labels"
fails = []
def check(name, got, want, tol=5e-4, unit=""):
    ok = got is not None and abs(got - want) <= tol
    if not ok: fails.append(name)
    g = "None" if got is None else f"{got:.4f}"
    print(f"{'PASS' if ok else 'FAIL'} {name:38s} recomputed {g:>9s}  quoted {want:9.4f}{unit}")
def value(name, got, unit=""):
    g = "None" if got is None else (f"{got:.4f}" if isinstance(got, float) else f"{got:,}")
    print(f"VAL  {name:38s} {g}{unit}")
def manual(name, where):
    print(f"MAN  {name:38s} not derivable here; source: {where}")

# ---------------------------------------------------------------- PoseBusters
S = {}
p = f"{PB}/pb_summary_teachers.csv"
if os.path.exists(p):
    import csv
    with open(p) as fh:
        for r in csv.DictReader(fh):
            S[r["teacher"]] = r
def fnum(t, col):
    r = S.get(t)
    if not r or not r.get(col): return None
    try: return float(r[col])
    except ValueError: return None

print("\n=== PoseBusters, pb_valid_core")
for t, want in (("smina_v3_depcond", 0.8348), ("crystal_self", 0.6325),
                ("boltz2_msa_mc", 0.5381), ("boltz2_msa", 0.5324),
                ("chai1", 0.3880), ("boltz2", 0.2482), ("smina", 0.0368)):
    check(f"pb_valid_core {t}", fnum(t, "pb_valid_core"), want, tol=1e-3)

print("\n=== PoseBusters, cohort sizes")
for t in ("crystal_self", "smina_v3_depcond", "chai1", "boltz2", "boltz2_msa",
          "boltz2_msa_mc"):
    n = fnum(t, "systems_scored")
    value(f"systems_scored {t}", int(n) if n else None)

print("\n=== PoseBusters, per-check failure fractions")
# The residual on non-metal ligands is one check; sanitization separates
# experimental ligands, which carry a chemical component definition, from
# predicted poses, which carry only coordinates.
for t, chk, want in (("chai1", "sanitization", 0.1675),
                     ("crystal_self", "sanitization", 0.0000)):
    p = f"{PB}/pb_checks_{t}.csv"
    got = None
    if os.path.exists(p):
        import csv
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r["check"] == chk:
                    got = float(r["fail_rate_of_systems"]); break
    check(f"{chk} fail {t}", got, want, tol=2e-3)
for t in ("crystal_self", "smina_v3_depcond", "boltz2_msa_mc"):
    p = f"{PB}/pb_checks_{t}.csv"
    if not os.path.exists(p): continue
    import csv
    with open(p) as fh:
        rows = sorted(csv.DictReader(fh),
                      key=lambda r: -float(r["fail_rate_of_systems"]))
    value(f"worst check {t}",
          None) if not rows else print(
        f"VAL  {'worst check '+t:38s} {rows[0]['check']} "
        f"{float(rows[0]['fail_rate_of_systems']):.4f}")

# --------------------------------------------------------------------- RMSD
def load(paths, meta=None):
    """Union over shards, de-duplicated by system_id, first occurrence wins.
    When meta is given, restrict to systems that also carry distance labels;
    that is the cohort tab:accuracy reports, so RMSD and contact accuracy
    describe the same systems. Some shard sets carry no pocket_resid."""
    sid, rm, pk = [], [], []
    for p in paths:
        if not os.path.exists(p): continue
        z = np.load(p, allow_pickle=True)
        n = len(z["system_id"])
        sid += [str(x) for x in z["system_id"]]
        rm  += list(z["ligand_rmsd"])
        pk  += list(z["pocket_resid"]) if "pocket_resid" in z else [np.nan]*n
    if not sid: return None, None
    sid = np.array(sid); rm = np.array(rm, float); pk = np.array(pk, float)
    _, ix = np.unique(sid, return_index=True)
    sid, rm, pk = sid[ix], rm[ix], pk[ix]
    if meta:
        try:
            import pyarrow.parquet as pq
            ids = set(pq.read_table(f"{LAB}/labels_{meta}_meta.parquet",
                                    columns=["system_id"]).column(0).to_pylist())
            keep = np.array([s in ids for s in sid])
            sid, rm, pk = sid[keep], rm[keep], pk[keep]
        except Exception:
            pass
    return rm, pk

# Explicit patterns only. A glob on bisy_rmsd_v2_boltz2_* also matches
# msa_full, msa2_full, msa_mc_{p,s,smoke} and r3empty_full, which are four
# different arms.
ARMS = {
    "chai1":         [f"{BISY}/bisy_rmsd_v2_chai1_{i}.npz" for i in range(8)],
    "boltz2":        [f"{BISY}/bisy_rmsd_v2_boltz2_{i}.npz" for i in range(8)],
    "boltz2_msa":    [f"{BISY}/bisy_rmsd_v2_boltz2_msa_full.npz",
                      f"{BISY}/bisy_rmsd_v2_boltz2_msa2_full.npz"],
    "smina":         [f"{BISY}/bisy_smina_v2_full{i}.npz" for i in range(8)],
    "boltz2_msa_mc": [f"{BISY}/bisy_rmsd_v2_boltz2_msa_mc_p.npz",
                      f"{BISY}/bisy_rmsd_v2_boltz2_msa_mc_s.npz"],
}
print("\n=== Symmetry-corrected ligand RMSD")
META = {"chai1": "chai1_crystal", "boltz2": "boltz2_crystal",
        "boltz2_msa": "boltz2_msa_crystal", "smina": "smina_crystal",
        "boltz2_msa_mc": None}
QUOTED = {"chai1": (9369, 5.27, 32.1), "boltz2": (8670, 11.45, 4.5),
          "boltz2_msa": (1753, 3.89, 41.9), "smina": (6485, None, None),
          "boltz2_msa_mc": (5569, 2.14, 48.7)}
R = {}
for arm, paths in ARMS.items():
    rm, pk = load(paths, META.get(arm))
    if rm is None: manual(f"rmsd {arm}", "shards absent"); continue
    v = rm[~np.isnan(rm)]; R[arm] = v
    n, med, u2 = QUOTED.get(arm, (None, None, None))
    value(f"rmsd n {arm}", int(len(v)))
    if med is not None:
        check(f"rmsd median {arm}", float(np.median(v)), med, tol=0.02, unit=" A")
        check(f"rmsd under 2A {arm}", 100*float((v < 2).mean()), u2, tol=0.15, unit=" %")
    else:
        value(f"rmsd median {arm}", float(np.median(v)), " A")
        value(f"rmsd under 2A {arm}", 100*float((v < 2).mean()), " %")
    if pk is not None:
        q = pk[~np.isnan(pk)]
        if len(q): value(f"pocket rmsd median {arm}", float(np.median(q)), " A")

# The aligned arm's two shards are disjoint partial scorings of one arm, not
# two arms. Reported separately so the union is auditable.
for f_ in ("boltz2_msa_full", "boltz2_msa2_full"):
    p = f"{BISY}/bisy_rmsd_v2_{f_}.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        value(f"aligned shard {f_}", int(len(z['system_id'])))
a = f"{BISY}/bisy_rmsd_v2_boltz2_msa_full.npz"
b = f"{BISY}/bisy_rmsd_v2_boltz2_msa2_full.npz"
if os.path.exists(a) and os.path.exists(b):
    sa = {str(x) for x in np.load(a, allow_pickle=True)["system_id"]}
    sb = {str(x) for x in np.load(b, allow_pickle=True)["system_id"]}
    value("aligned shard overlap", len(sa & sb))

# --------------------------------------------------- contact-set accuracy
print("\n=== Contact accuracy against ground truth")
# Two cohorts are in circulation: the three-teacher core, on which the arms are
# directly comparable, and each arm's own coverage. Figure 5b and the accuracy
# table have disagreed by roughly 0.014 for this reason. Both are computed.
try:
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    def cset(name, ids):
        t = pq.read_table(f"{LAB}/labels_{name}_contacts.parquet",
                          columns=["system_id", "res_row", "contact_4A"])
        t = t.filter(pc.is_in(t.column("system_id"), pa.array(sorted(ids))))
        t = t.filter(pc.equal(t.column("contact_4A"), True)).to_pydict()
        out = collections.defaultdict(set)
        for s, r in zip(t["system_id"], t["res_row"]): out[s].add(r)
        return out
    def ids(name):
        return set(pq.read_table(f"{LAB}/labels_{name}_meta.parquet",
                                 columns=["system_id"]).column(0).to_pylist())
    gt_ids = ids("crystal_groundtruth")
    arms = {a: ids(f"{a}_crystal") for a in ("smina", "chai1", "boltz2")}
    core = gt_ids & arms["smina"] & arms["chai1"] & arms["boltz2"]
    value("crystal triple core", len(core))
    G = cset("crystal_groundtruth", core)
    # Core cohort, n 7,166. Own-cohort means are reported separately below;
    # the two differ because each arm covers a differently difficult subset.
    for a_, want in (("smina", 0.5592), ("chai1", 0.4442), ("boltz2", 0.1815)):
        P = cset(f"{a_}_crystal", core)
        v = [len(G[s] & P[s]) / max(1, len(G[s] | P[s]))
             for s in core if s in G and s in P]
        check(f"contact acc core {a_}", float(np.mean(v)), want, tol=2e-3)
        own = arms[a_] & gt_ids
        Go = cset("crystal_groundtruth", own); Po = cset(f"{a_}_crystal", own)
        w = [len(Go[s] & Po[s]) / max(1, len(Go[s] | Po[s]))
             for s in own if s in Go and s in Po]
        value(f"contact acc own-cohort {a_}", float(np.mean(w)))
except Exception as e:
    manual("contact accuracy", f"deposit unavailable here: {type(e).__name__}")

# ------------------------------------------------------- training cutoff
print("\n=== Training-cutoff stratification")
p = f"{REP}/pdb_deposit_dates.json"
if os.path.exists(p) and "chai1" in R:
    dates = json.load(open(p))
    value("deposition dates available", len(dates))
    manual("cutoff split by arm",
           "needs the per-arm scored id lists joined to pdb_deposit_dates.json")
else:
    manual("cutoff split by arm", f"{p}")

# --------------------------------------------------- teacher independence
print("\n=== Teacher independence")
for f_ in ("agreement_contact_4A.json", "agreement_contact_5A.json",
           "agreement_contact_8A.json"):
    p = f"{REP}/{f_}"
    if not os.path.exists(p): continue
    d = json.load(open(p))
    print(f"VAL  {f_+' keys':38s} {list(d)[:6]}")
manual("accuracy correlations 0.3656 / 0.2050 / 0.0926",
       "computed ad hoc in the v9 session; no artifact written")
manual("isotonic MAE 0.0814 vs 0.1734",
       "final_five.py; reliability_fit.npz if it was persisted")
manual("conformal half-width 0.1526 at 0.90",
       "deposit metadata/system_reliability.parquet, constant column")
manual("frame control 1.0000 vs 0.2843 (n=199)",
       "one-off receptor-conditioning experiment; no artifact located")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
