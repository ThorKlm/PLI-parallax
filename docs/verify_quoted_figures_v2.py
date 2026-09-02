#!/usr/bin/env python
"""Recompute every manuscript figure that is derivable from the deposit.

Each claim is (section, name, recomputed, quoted). Claims are grouped by the
artifact they need so the large contact tables are read once. Figures that
cannot be recomputed from the deposit are listed in MANUAL at the end with the
artifact that would be needed; they are reported, not silently omitted.

Read-only. Takes the deposit path as an argument.
"""
import collections, glob, json, os, sys
import numpy as np, h5py
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

D = sys.argv[1] if len(sys.argv) > 1 else "/workspace/deposit_v3"
L, M, S, ST = f"{D}/labels", f"{D}/metadata", f"{D}/splits", f"{D}/stores"
claims, notes = [], []

def claim(sec, name, got, want, tol=0):
    """tol=0 is exact; a float tol compares within an absolute tolerance."""
    ok = (got == want) if tol == 0 else (got is not None and abs(got - want) <= tol)
    claims.append((sec, name, got, want, ok))

def note(sec, name, got):
    notes.append((sec, name, got))

def arm_meta(n):
    fn = ("labels_crystal_groundtruth_meta.parquet" if n == "crystal_groundtruth"
          else f"labels_{n}_meta.parquet")
    return set(pq.read_table(f"{L}/{fn}", columns=["system_id"]).column(0).to_pylist())

def nrows(p):
    return pq.read_metadata(p).num_rows

# ------------------------------------------------------- 1. arms and coverage
A = {k: arm_meta(k) for k in
     ("smina_corpus", "boltz2_corpus", "chai1_corpus", "boltz2_msa_corpus",
      "smina_crystal", "boltz2_crystal", "chai1_crystal", "boltz2_msa_crystal",
      "crystal_groundtruth")}
sec = "coverage"
for k, want in (("smina_corpus", 31713), ("boltz2_corpus", 23494), ("chai1_corpus", 23485),
                ("boltz2_msa_corpus", 161), ("crystal_groundtruth", 19350),
                ("smina_crystal", 11543), ("chai1_crystal", 9410),
                ("boltz2_crystal", 8725), ("boltz2_msa_crystal", 1755)):
    claim(sec, f"arm {k}", len(A[k]), want)
claim(sec, "corpus triple core",
      len(A["smina_corpus"] & A["boltz2_corpus"] & A["chai1_corpus"]), 23451)
claim(sec, "corpus union",
      len(A["smina_corpus"] | A["boltz2_corpus"] | A["chai1_corpus"] | A["boltz2_msa_corpus"]), 31746)
claim(sec, "crystal triple core",
      len(A["chai1_crystal"] & A["boltz2_crystal"] & A["smina_crystal"]), 7166)
# the tautology in v1 replaced by the property it was standing in for
claim(sec, "ground truth is a superset of every crystal arm",
      all(A[k] <= A["crystal_groundtruth"] for k in
          ("smina_crystal", "chai1_crystal", "boltz2_crystal", "boltz2_msa_crystal")), True)
claim(sec, "tier id spaces disjoint",
      len((A["smina_corpus"] | A["boltz2_corpus"] | A["chai1_corpus"] | A["boltz2_msa_corpus"])
          & A["crystal_groundtruth"]), 0)
claim(sec, "total meta rows", sum(len(v) for v in A.values()), 129636)

# --------------------------------------------------------- 2. table row counts
sec = "tables"
CONTACTS = {
    "smina_corpus": 70033209, "boltz2_corpus": 67281516, "chai1_corpus": 54348638,
    "crystal_groundtruth": 46881795, "boltz2_crystal": 21637489,
    "chai1_crystal": 21558787, "smina_crystal": 21317135,
    "boltz2_msa_crystal": 3890602, "boltz2_msa_corpus": 365475,
}
tot = 0
for k, want in CONTACTS.items():
    fn = ("labels_crystal_groundtruth_contacts.parquet" if k == "crystal_groundtruth"
          else f"labels_{k}_contacts.parquet")
    n = nrows(f"{L}/{fn}")
    tot += n
    claim(sec, f"contacts {k}", n, want)
claim(sec, "contact rows, all arms", tot, 307314646)
allpq = sorted(glob.glob(f"{D}/*/*.parquet"))
claim(sec, "Parquet files in the deposit", len(allpq), 30)
claim(sec, "Parquet rows, all tables", sum(nrows(p) for p in allpq), 336385168)
claim(sec, "files in the deposit",
      sum(len(f) for _, _, f in os.walk(D)), 45)   # 43 manifest entries plus both manifests
claim(sec, "deposit size, GB",
      round(sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(D) for f in fns) / 1e9, 2), 6.16, tol=0.01)

# ------------------------------------------------------------ 3. field dictionary
sec = "dictionary"
F = json.load(open(f"{D}/FIELDS.json"))
claim(sec, "fields defined", len(F["fields"]), 235)
claim(sec, "record sets", len(F["record_sets"]), 14)
cr = json.load(open(f"{D}/croissant.json"))
claim(sec, "croissant distribution entries", len(cr["distribution"]), 34)
claim(sec, "croissant recordSets", len(cr["recordSet"]), 30)
claim(sec, "croissant fields",
      sum(len(r.get("field", [])) for r in cr["recordSet"]), 467)

# ------------------------------------------------------------------ 4. stores
sec = "store"
for tag, n_want, gb_want in (("corpus", 31617, 3.10), ("crystal", 17368, 1.91)):
    p = f"{ST}/{tag}.h5"
    f = h5py.File(p, "r")
    claim(sec, f"{tag} systems", len(f), n_want)
    claim(sec, f"{tag} size, GB", round(os.path.getsize(p) / 1e9, 2), gb_want, tol=0.01)
    f.close()
fc, fx = h5py.File(f"{ST}/corpus.h5", "r"), h5py.File(f"{ST}/crystal.h5", "r")
claim(sec, "corpus skipped", 31746 - len(fc), 129)
claim(sec, "crystal skipped", 19350 - len(fx), 1982)
g = fx[list(fx)[0]]
claim(sec, "quantisation scale", int(g.attrs["scale"]), 100)
claim(sec, "teacher axis order",
      [x.decode() if isinstance(x, bytes) else x for x in g.attrs["teachers"]],
      ["crystal", "chai1", "boltz2", "boltz2_msa", "smina"])
# crystal systems carrying ground-truth rows but no experimental pose
nopose = sum(1 for s in fx if not fx[s]["ligand_valid"][0].any())
claim(sec, "crystal systems without an experimental pose", nopose, 1341)
claim(sec, "  as a percentage", round(100 * nopose / len(fx), 1), 7.7, tol=0.05)

# ------------------------------------------------------------- 5. metadata tables
sec = "metadata"
claim(sec, "residues_corpus rows", nrows(f"{M}/residues_corpus.parquet"), 10607632)
claim(sec, "pockets_corpus rows", nrows(f"{M}/pockets_corpus.parquet"), 101510)
claim(sec, "system_reliability rows", nrows(f"{M}/system_reliability.parquet"), 30567)
claim(sec, "residue_support rows", nrows(f"{M}/residue_support.parquet"), 1056757)
claim(sec, "teacher_agreement rows", nrows(f"{M}/teacher_agreement.parquet"), 32249)
li = json.load(open(f"{M}/ligand_identity.json"))
claim(sec, "ligand_identity keys", len(li), 31878)
claim(sec, "distinct corpus SMILES in ligand_identity",
      len({v["inchikey"] for v in li.values() if v.get("inchikey")}), 31193)
sat = json.load(open(f"{M}/int16_saturated_systems.json"))
claim(sec, "int16 saturated systems", sum(len(v) for v in sat.values()), 89)

pk = pq.read_table(f"{M}/pockets_corpus.parquet").to_pydict()
claim(sec, "pocket receptors", len(set(pk["protein_id"])), 20404)
for f_, want in (("source_fpocket", 51096), ("source_p2rank", 34187), ("source_vnegnn", 42288)):
    claim(sec, f"pockets found by {f_[7:]}", int(sum(1 for x in pk[f_] if x)), want)
nflag = collections.Counter(
    sum(1 for f_ in ("source_fpocket", "source_p2rank", "source_vnegnn") if pk[f_][i])
    for i in range(len(pk["protein_id"])))
for k, want in ((1, 79800), (2, 17359), (3, 4351)):
    claim(sec, f"pockets with {k} detector flag(s)", nflag[k], want)
claim(sec, "pockets with no flag", nflag[0], 0)

# ---------------------------------------------------------- 6. reliability
sec = "reliability"
rel = pq.read_table(f"{M}/system_reliability.parquet").to_pydict()
p = np.array(rel["pred_accuracy"], float)
a = np.array(rel["agreement"], float)
tier = np.array(rel["tier"])
claim(sec, "corpus rows", int((tier == "corpus").sum()), 23433)
claim(sec, "crystal rows", int((tier == "crystal").sum()), 7134)
claim(sec, "isotonic floor", round(float(p.min()), 4), 0.1604, tol=1e-4)
claim(sec, "systems at the floor", int((p <= p.min() + 1e-9).sum()), 4151)
hw = set(np.round(np.array(rel["conformal_halfwidth_90"], float), 8).tolist())
claim(sec, "conformal half-width is constant", len(hw), 1)
claim(sec, "conformal half-width", round(hw.pop(), 4), 0.1526, tol=1e-4)
claim(sec, "agreement-accuracy Pearson", round(float(np.corrcoef(a, p)[0, 1]), 4), 0.9888, tol=5e-4)
for thr, want in ((0.30, 12937), (0.50, 3943), (0.70, 515)):
    claim(sec, f"systems with pred_accuracy >= {thr:.2f}", int((p >= thr).sum()), want)

sup = pq.read_table(f"{M}/residue_support.parquet", columns=["tier", "n_teachers_asserting"]).to_pydict()
st = np.array(sup["tier"])
claim(sec, "residue_support corpus rows", int((st == "corpus").sum()), 847323)
claim(sec, "residue_support crystal rows", int((st == "crystal").sum()), 209434)
claim(sec, "support count range",
      (int(min(sup["n_teachers_asserting"])), int(max(sup["n_teachers_asserting"]))), (1, 3))

# ------------------------------------------------------ 7. teacher agreement
sec = "agreement"
ta = pq.read_table(f"{M}/teacher_agreement.parquet").to_pydict()
off = np.array(ta["teacher_centroid_offset_median"], float)
mx = np.array(ta["teacher_centroid_offset_max"], float)
tt = np.array(ta["tier"])
ok = np.isfinite(off)
claim(sec, "unmeasured systems", int((~ok).sum()), 61)
for t, want in (("corpus", 18.2), ("crystal", 11.5)):
    m = ok & (tt == t)
    claim(sec, f"median centroid offset, {t}", round(float(np.median(off[m])), 1), want, tol=0.05)
n = int(ok.sum())
for thr, want_n, want_pct in ((1.0, 351, 1.1), (2.0, 939, 2.9),
                              (5.0, 3038, 9.4), (10.0, 6811, 21.2)):
    k = int((mx[ok] <= thr).sum())
    claim(sec, f"all pairs within {thr:g} A", k, want_n)
    claim(sec, f"  as a percentage", round(100 * k / n, 1), want_pct, tol=0.05)

# ----------------------------------------------------------------- 8. splits
sec = "splits"
sc = pq.read_table(f"{S}/splits_corpus.parquet", columns=["split_tag", "system_id"]).to_pydict()
sx = pq.read_table(f"{S}/splits_crystal.parquet", columns=["split_tag", "system_id"]).to_pydict()
tc, tx = set(sc["split_tag"]), set(sx["split_tag"])
tmp = set(pq.read_table(f"{S}/splits_temporal.parquet", columns=["split_tag"]).column(0).to_pylist())
lo = set(pq.read_table(f"{S}/splits_lo_corpus.parquet", columns=["split_tag"]).column(0).to_pylist())
claim(sec, "corpus configurations", len(tc), 375)
claim(sec, "crystal configurations", len(tx), 250)
claim(sec, "temporal configurations", len(tmp), 6)
claim(sec, "lead-optimisation configurations", len(lo), 15)
claim(sec, "configurations, all families", len(tc) + len(tx) + len(tmp) + len(lo), 646)
claim(sec, "corpus systems per configuration", len(set(sc["system_id"])), 31746)
claim(sec, "crystal systems per configuration", len(set(sx["system_id"])), 18458)
claim(sec, "crystal systems omitted from splits", 19350 - len(set(sx["system_id"])), 892)
claim(sec, "splits_corpus rows", nrows(f"{S}/splits_corpus.parquet"), 11904750)
claim(sec, "splits_crystal rows", nrows(f"{S}/splits_crystal.parquet"), 4614500)
claim(sec, "splits_temporal rows", nrows(f"{S}/splits_temporal.parquet"), 116100)
fams = set(pq.read_table(f"{S}/splits_corpus.parquet", columns=["family"]).column(0).to_pylist())
fams |= set(pq.read_table(f"{S}/splits_crystal.parquet", columns=["family"]).column(0).to_pylist())
fams |= {"T1", "LO"}
claim(sec, "split families", sorted(fams), ["C1", "C2l", "C2p", "C3", "C3comp", "LO", "T1"])

for tag, fn, want in (("corpus", "split_summary_corpus.parquet", 174),
                      ("crystal", "split_summary_crystal.parquet", 125)):
    d = pq.read_table(f"{S}/{fn}", columns=["n_test", "n_systems"]).to_pydict()
    claim(sec, f"underpowered configurations, {tag}",
          sum(1 for t, s in zip(d["n_test"], d["n_systems"]) if t < 0.05 * s), want)
    claim(sec, f"summary columns, {tag}",
          len(pq.ParquetFile(f"{S}/{fn}").schema_arrow.names), 49)

ssc = pq.read_table(f"{S}/split_summary_corpus.parquet").to_pydict()
ssx = pq.read_table(f"{S}/split_summary_crystal.parquet").to_pydict()
def famean(d, col, fam):
    v = [x for x, f_ in zip(d[col], d["family"]) if f_ == fam and x is not None]
    return round(float(np.mean(v)), 3) if v else None
claim(sec, "C1 ligand C2ST AUC, corpus", famean(ssc, "c2st_auc", "C1"), 0.508, tol=0.002)
claim(sec, "C3 ligand C2ST AUC, corpus", famean(ssc, "c2st_auc", "C3"), 0.824, tol=0.002)
claim(sec, "C2l ligand C2ST AUC, crystal", famean(ssx, "c2st_auc", "C2l"), 0.962, tol=0.002)
claim(sec, "C2l ligand C2ST AUC, corpus", famean(ssc, "c2st_auc", "C2l"), 0.618, tol=0.002)

# -------------------------------------------------- 9. distances and censoring
sec = "distances"
viol = worst = 0
for p_ in sorted(glob.glob(f"{L}/*_contacts.parquet")):
    t = pq.read_table(p_, columns=["d_ca", "d_min"])
    d = pc.subtract(t.column("d_min"), t.column("d_ca"))
    m = pc.greater(d, 0)
    k = pc.sum(m).as_py() or 0
    if k:
        worst = max(worst, pc.max(pc.filter(d, m)).as_py())
    viol += k
    del t
claim(sec, "rows where d_min exceeds d_ca", viol, 195)
claim(sec, "largest excess, A", round(worst, 5), 0.00781, tol=1e-5)

thr_counts = {}
for thr in (4.0, 5.0, 8.0, 15.0):
    k = 0
    for p_ in sorted(glob.glob(f"{L}/*_contacts.parquet")):
        col = pq.read_table(p_, columns=["d_min"]).column(0)
        k += pc.sum(pc.equal(col, pa.scalar(np.float32(thr)))).as_py() or 0
    thr_counts[thr] = k
for thr, want in ((4.0, 22960), (5.0, 37713), (8.0, 128932), (15.0, 186886)):
    claim(sec, f"rows exactly at {thr:g} A", thr_counts[thr], want)

c = pq.read_table(f"{L}/labels_crystal_groundtruth_contacts.parquet",
                  columns=["system_id", "res_row"]).to_pydict()
per = collections.defaultdict(set)
for s, r in zip(c["system_id"], c["res_row"]):
    per[s].add(r)
mt = pq.read_table(f"{L}/labels_crystal_groundtruth_meta.parquet",
                   columns=["system_id", "n_res"]).to_pydict()
nres = dict(zip(mt["system_id"], mt["n_res"]))
frac = [len(per[s]) / nres[s] for s in per if nres.get(s)]
claim(sec, "mean fraction of residues carrying rows",
      round(100 * float(np.mean(frac)), 1), 23.5, tol=0.05)

# ------------------------------------------------------ 10. meta column claims
sec = "meta columns"
hc_false = 0
for p_ in sorted(glob.glob(f"{L}/*_meta.parquet")):
    t = pq.read_table(p_, columns=["system_id", "has_contacts"]).to_pydict()
    hc_false += sum(1 for h in t["has_contacts"] if h is False)
claim(sec, "systems with has_contacts False", hc_false, 259)

m = pq.read_table(f"{L}/labels_crystal_groundtruth_meta.parquet",
                  columns=["system_id", "n_lig_heavy"]).to_pydict()
pop = {s: v for s, v in zip(m["system_id"], m["n_lig_heavy"]) if v is not None}
claim(sec, "systems with n_lig_heavy populated", len(pop), 9575)
claim(sec, "systems with n_lig_heavy null", 19350 - len(pop), 9775)
t = pq.read_table(f"{L}/labels_crystal_groundtruth_contacts.parquet",
                  columns=["system_id", "atom_idx"])
nat = t.group_by("system_id").aggregate([("atom_idx", "count_distinct")]).to_pydict()
obs = dict(zip(nat["system_id"], nat["atom_idx_count_distinct"]))
eq = hi = lo_ = 0
for s, v in pop.items():
    o = obs.get(s)
    if o is None: continue
    eq += v == o; hi += v > o; lo_ += v < o
n = eq + hi + lo_
for lab, k, want in (("equal", eq, 89.2), ("declared larger", hi, 8.5), ("declared smaller", lo_, 2.3)):
    claim(sec, f"n_lig_heavy {lab}, per cent", round(100 * k / n, 1), want, tol=0.05)

mm = pq.read_table(f"{L}/labels_smina_corpus_meta.parquet",
                   columns=["protein_id", "pocket_rank", "affinity"]).to_pydict()
acc = set()
for p_ in sorted(glob.glob(f"{L}/*_corpus_meta.parquet")):
    acc |= {x for x in pq.read_table(p_, columns=["protein_id"]).column(0).to_pylist() if x}
claim(sec, "distinct corpus accessions, all arms", len(acc), 906)
af = [x for x in mm["affinity"] if x is not None]
claim(sec, "affinity minimum, kcal/mol", round(float(min(af)), 2), -14.18, tol=0.01)
claim(sec, "affinity maximum, kcal/mol", round(float(max(af)), 2), -3.21, tol=0.01)
claim(sec, "smina meta rows with pocket_rank",
      sum(1 for x in mm["pocket_rank"] if x is not None), 31713)

fc.close(); fx.close()

# ------------------------------------------------------------------- report
by_sec = collections.OrderedDict()
for sec_, name, got, want, ok in claims:
    by_sec.setdefault(sec_, []).append((name, got, want, ok))
fails = 0
for sec_, rows_ in by_sec.items():
    print(f"\n===== {sec_}")
    for name, got, want, ok in rows_:
        fails += not ok
        g = f"{got:,}" if isinstance(got, int) else str(got)
        w = f"{want:,}" if isinstance(want, int) else str(want)
        print(f"  {'PASS' if ok else 'FAIL'} {name:52s} recomputed {g:>16s}  quoted {w:>16s}")
if notes:
    print("\n===== reported, not asserted")
    for sec_, name, got in notes:
        print(f"       {sec_}/{name}: {got}")

# Figures quoted in the manuscript that the deposit cannot reproduce, with the
# artifact each would need. Listed so the coverage gap is explicit rather than
# silent; verify_scoring_figures.py covers the scoring half.
MANUAL = [
 ("scoring", "pb_valid_core per arm", "posebusters/tables"),
 ("scoring", "symmetry-corrected ligand RMSD per arm", "smina_v2_b/v3, bisy output"),
 ("scoring", "pocket RMSD per arm", "reports/"),
 ("validation", "atom-order agreement 0.0040 A over 1,754 comparisons", "verify_atom_order.py"),
 ("chain audit", "9,565 crystal systems never folded, 49.4 per cent", "CHAIN_AUDIT.md source"),
 ("chain audit", "8,454 of 9,565 within the per-chain cap, 88.4 per cent", "CHAIN_AUDIT.md source"),
 ("chain audit", "18,239 admissible under a per-chain cap against 9,785 built", "CHAIN_AUDIT.md source"),
 ("constraints", "502 HEM inputs, zero Boltz-2 outputs, 67.3 per cent of 746 dropped",
  "boltz processed-input manifest"),
 ("splits", "886 / 896 / 900 protein clusters at 0.30 / 0.40 / 0.50", "MMseqs2 output"),
 ("splits", "14,852 key-chain sequences, 7,149 and 10,714 clusters", "MMseqs2 output"),
 ("splits", "87,652 against 146 corpus pairs above 0.30 identity", "all-vs-all identity table"),
 ("splits", "9,455,750 against 125,832 crystal pairs", "all-vs-all identity table"),
 ("store", "structural overhead 45 and 57 per cent", "h5py per-dataset inspection"),
 ("environment", "gemmi, pyarrow, pandas, rdkit versions", "the build environment"),
 ("suites", "191 checks over eight suites", "run_all_tests.sh"),
]
print(f"\n===== not derivable from the deposit ({len(MANUAL)})")
for sec_, name, need in MANUAL:
    print(f"       {sec_:12s} {name:60s} needs {need}")

print(f"\n{len(claims)} claims, {fails} FAIL" if fails else
      f"\nall {len(claims)} claims reproduce")
sys.exit(1 if fails else 0)
