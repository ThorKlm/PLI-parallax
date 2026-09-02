#!/usr/bin/env python
"""Validation suite for the PLI-Parallax deposit.

Fifty checks, read-only. Most run against the deposit alone. Seven compare the
coordinate store against the structures it was built from and need those
structures on disk; they report as skipped otherwise.

    python test_deposit.py [deposit_root]

Environment:
    PLIP_STRUCTURES   crystal-tier source mmCIF, one file per PDB entry
    PLIP_AF           AlphaFold receptor PDB files
    PLIP_SYS2ACC      system_id to UniProt accession map, JSON

Prints PASS or FAIL per check. Lines marked with a dash are measurements with no
correct value and never fail the run. Exit status is non-zero on any failure.
"""
import collections, glob, json, os, random, re, subprocess, sys
import numpy as np
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

D = sys.argv[1] if len(sys.argv) > 1 else "."
STRUCTURES = os.environ.get("PLIP_STRUCTURES", "")
AF = os.environ.get("PLIP_AF", "")
SYS2ACC = os.environ.get("PLIP_SYS2ACC", "")

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
         "THR TRP TYR VAL".split())
BACKBONE = ("N", "CA", "C", "O", "OXT")
TEACHERS = ["crystal", "chai1", "boltz2", "boltz2_msa", "smina"]
SLOT = {t: i for i, t in enumerate(TEACHERS)}
random.seed(0)

fails, skipped = [], []
def check(n, name, ok, msg=""):
    if not ok: fails.append(f"{n:02d} {name}")
    print(f"{'PASS' if ok else 'FAIL'} {n:02d} {name}: {msg}")
def skip(n, name, why):
    skipped.append(f"{n:02d} {name}")
    print(f"SKIP {n:02d} {name}: {why}")
def note(name, msg):
    print(f"       - {name}: {msg}")

L = f"{D}/labels"
labels = sorted(glob.glob(f"{L}/*.parquet"))
cont_f = [p for p in labels if p.endswith("_contacts.parquet")]
meta_f = [p for p in labels if p.endswith("_meta.parquet")]
mdata = sorted(glob.glob(f"{D}/metadata/*"))
splits = sorted(glob.glob(f"{D}/splits/*.parquet"))
stores = sorted(glob.glob(f"{D}/stores/*.h5"))
fx = h5py.File(f"{D}/stores/crystal.h5", "r")
fc = h5py.File(f"{D}/stores/corpus.h5", "r")
README = open(f"{D}/README.md").read()

def meta(name): return pq.read_table(f"{L}/labels_{name}_meta.parquet")
def cont(name): return pq.read_table(f"{L}/labels_{name}_contacts.parquet")

def residue_gate(st):
    """Standard residue, alpha carbon present, at least one heavy atom."""
    out = []
    for ch in st[0]:
        for r in ch:
            if r.name not in AA: continue
            if r.find_atom("CA", "*") is None: continue
            if not any(a.element.name not in ("H", "D") for a in r): continue
            out.append(r)
    return out

def dequant(g, teacher, key):
    return g[f"protein/{teacher}"][key][:] / g.attrs["scale"] + g.attrs["centroid_xyz"]

try:
    import gemmi
    HAVE_GEMMI = True
except ImportError:
    HAVE_GEMMI = False

# ---------------------------------------------------------------- inventory --

check(1, "file inventory",
      len(cont_f) == 9 and len(meta_f) == 9 and len(splits) == 7 and len(stores) == 2,
      f"{len(cont_f)} contacts, {len(meta_f)} meta, {len(splits)} splits, "
      f"{len(stores)} stores, {len(mdata)} metadata")

c_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in cont_f)
m_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in meta_f)
all_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in glob.glob(f"{D}/*/*.parquet"))
check(2, "row totals", c_rows == 307_314_646 and m_rows == 129_636,
      f"contacts {c_rows:,}, meta {m_rows:,}, tabular layer {all_rows:,}")

tabs = [pq.read_table(p) for p in meta_f]
uniform = all(tabs[0].schema.equals(t.schema) for t in tabs)
cat = pa.concat_tables(tabs)
check(3, "meta schema uniform", uniform and cat.num_rows == m_rows,
      f"{cat.num_columns} columns, {cat.num_rows:,} rows, no coercion")
del tabs, cat

odd = {os.path.basename(p): pq.ParquetFile(p).metadata.row_group(0).column(0).compression
       for p in cont_f}
odd = {k: v for k, v in odd.items() if v != "ZSTD"}
check(4, "compression", not odd, f"contacts ZSTD, exceptions {odd or 'none'}")

typed = {}
for fld in json.load(open(f"{D}/FIELDS.json"))["fields"]:
    typed.setdefault(fld["record_set"], {})[fld["name"]] = fld["parquet_type"]
mismatch = []
for p in labels:
    rs = "meta" if p.endswith("_meta.parquet") else "contacts"
    s = pq.ParquetFile(p).schema_arrow
    mismatch += [f"{os.path.basename(p)}:{n}" for n, t in zip(s.names, map(str, s.types))
                 if typed.get(rs, {}).get(n) != t]
check(5, "declared types", not mismatch, f"{len(mismatch)} mismatches {mismatch[:2]}")

# ---------------------------------------------------------------- integrity --

r = subprocess.run(["sha256sum", "-c", "MANIFEST.sha256"], cwd=D,
                   capture_output=True, text=True)
n_ok = r.stdout.count(": OK")
n_bad = len([l for l in r.stdout.splitlines() if l.strip() and ": OK" not in l])
check(6, "manifest verifies", n_bad == 0 and n_ok > 0, f"{n_ok} OK, {n_bad} failed")

man = {x["path"]: x["sha256"] for x in json.load(open(f"{D}/MANIFEST.json"))["files"]}
on_disk = {os.path.relpath(p, D) for p in glob.glob(f"{D}/**/*", recursive=True)
           if os.path.isfile(p)} - {"MANIFEST.sha256", "MANIFEST.json"}
check(7, "manifest complete", on_disk == set(man),
      f"{len(on_disk)} files, unlisted {sorted(on_disk - set(man)) or 'none'}")

cr = json.load(open(f"{D}/croissant.json"))
dist = {e["contentUrl"]: e.get("sha256") for e in cr.get("distribution", [])
        if e.get("contentUrl")}
dis = [u for u, h in dist.items() if u in man and h != man[u]]
check(8, "croissant hashes", not dis,
      f"{len(set(dist) & set(man))} shared, {len(dis)} disagree")

shipping = {os.path.relpath(p, D) for p in labels + mdata + splits + stores}
check(9, "croissant coverage", not (shipping - set(dist)),
      f"{len(shipping)} data files, undeclared {sorted(shipping - set(dist)) or 'none'}")

inv = {x["path"]: x for x in json.load(open(f"{D}/_inventory.json"))}
bad = []
for p in glob.glob(f"{D}/labels/*.parquet") + glob.glob(f"{D}/metadata/*.parquet") \
        + glob.glob(f"{D}/splits/*.parquet"):
    rel = os.path.relpath(p, D)
    if rel not in inv: bad.append(rel)
    elif inv[rel]["rows"] != pq.ParquetFile(p).metadata.num_rows: bad.append(rel + " rows")
    elif inv[rel]["bytes"] != os.path.getsize(p): bad.append(rel + " bytes")
check(10, "inventory", not bad, f"{len(bad)} discrepancies {bad[:3]}")

PATHPAT = re.compile(r"(/workspace/|/home/[a-z]|/mnt/|/tmp/|/root/)")
leaks = collections.defaultdict(int)
for rel in ("README.md", "FIELDS.json", "FIELDS.md", "croissant.json",
            "_inventory.csv", "_inventory.json", "MANIFEST.json"):
    p = f"{D}/{rel}"
    if not os.path.exists(p): continue
    leaks[rel] = sum(1 for line in open(p, errors="replace") if PATHPAT.search(line))
    if not leaks[rel]: del leaks[rel]
for p in glob.glob(f"{D}/*/*.parquet"):
    kv = pq.ParquetFile(p).metadata.metadata or {}
    blob = b" ".join(list(kv.keys()) + list(kv.values())).decode("utf8", "replace")
    if PATHPAT.search(blob): leaks[os.path.basename(p)] += 1
check(11, "no build paths", not leaks, dict(leaks) or "text and Parquet metadata clean")

# -------------------------------------------------------------- identifiers --

orphan = {}
for cp in cont_f:
    mp = cp.replace("_contacts.parquet", "_meta.parquet")
    cs = set(pq.read_table(cp, columns=["system_id"]).column(0).to_pylist())
    ms = set(pq.read_table(mp, columns=["system_id"]).column(0).to_pylist())
    if cs - ms: orphan[os.path.basename(cp)] = len(cs - ms)
check(12, "contacts join to meta", not orphan, f"orphans {orphan or 'none'}")

dup = []
for p in meta_f:
    ids = pq.read_table(p, columns=["system_id"]).column(0).to_pylist()
    if len(ids) != len(set(ids)): dup.append(os.path.basename(p))
check(13, "system_id unique", not dup, f"{len(dup)} files with duplicates {dup}")

corpus_ids, crystal_ids = set(), set()
for p in glob.glob(f"{L}/*_corpus_meta.parquet"):
    corpus_ids |= set(pq.read_table(p, columns=["system_id"]).column(0).to_pylist())
for p in glob.glob(f"{L}/*_crystal_meta.parquet") + [f"{L}/labels_crystal_groundtruth_meta.parquet"]:
    crystal_ids |= set(pq.read_table(p, columns=["system_id"]).column(0).to_pylist())
check(14, "tiers disjoint", not (corpus_ids & crystal_ids),
      f"corpus {len(corpus_ids):,}, crystal {len(crystal_ids):,}, "
      f"overlap {len(corpus_ids & crystal_ids)}")

check(15, "store ids known", set(fc) <= corpus_ids and set(fx) <= crystal_ids,
      f"{len(set(fc) - corpus_ids)} corpus and {len(set(fx) - crystal_ids)} "
      f"crystal store ids not in any meta table")

known = corpus_ids | crystal_ids
orph = {n: len(set(pq.read_table(f"{D}/metadata/{n}.parquet",
                                 columns=["system_id"]).column(0).to_pylist()) - known)
        for n in ("system_reliability", "residue_support")}
check(16, "metadata joins", not any(orph.values()), orph)

res_acc = set(pq.read_table(f"{D}/metadata/residues_corpus.parquet",
                            columns=["accession"]).column(0).to_pylist())
acc = set()
for p in glob.glob(f"{L}/*_corpus_meta.parquet"):
    acc |= {x for x in pq.read_table(p, columns=["protein_id"]).column(0).to_pylist() if x}
check(17, "residue table coverage", not (acc - res_acc),
      f"{len(acc):,} accessions, {len(acc - res_acc)} without residue rows")

# ------------------------------------------------------------------- labels --

# d_min cannot exceed d_ca in exact arithmetic. Both are stored at half
# precision, so a residue whose nearest heavy atom is its own alpha carbon can
# round to adjacent grid points.
viol, worst, per = 0, 0.0, {}
for cp in cont_f:
    t = pq.read_table(cp, columns=["d_ca", "d_min"])
    d = pc.subtract(t.column("d_min"), t.column("d_ca"))
    m = pc.greater(d, 0)
    n = pc.sum(m).as_py() or 0
    if n:
        per[os.path.basename(cp)] = n
        worst = max(worst, pc.max(pc.filter(d, m)).as_py())
    viol += n
    del t
check(18, "d_min at most d_ca", viol <= 200 and worst <= 0.01,
      f"{viol} rows exceed by at most {worst:.2e} A, one half-precision step; {per}")

# The cutoff applies to d_min. d_ca reaches further, since a long side chain can
# contact the ligand while its alpha carbon sits well outside.
prob = []
for cp in cont_f:
    t = pq.read_table(cp, columns=["d_ca", "d_min"])
    for col, cap in (("d_min", 15.01), ("d_ca", 40.0)):
        c = t.column(col)
        if pc.sum(pc.is_nan(c)).as_py(): prob.append(f"{os.path.basename(cp)}.{col} nan")
        if pc.sum(pc.less(c, 0)).as_py(): prob.append(f"{os.path.basename(cp)}.{col} negative")
        if pc.sum(pc.greater(c, cap)).as_py(): prob.append(f"{os.path.basename(cp)}.{col} over cap")
    del t
check(19, "distance domain", not prob,
      prob[:3] or "d_min in [0, 15.01], d_ca in [0, 40]")

mism = 0
for cp in cont_f:
    t = pq.read_table(cp, columns=["d_min", "contact_4A", "contact_5A", "contact_8A"])
    for thr, col in ((4.0, "contact_4A"), (5.0, "contact_5A"), (8.0, "contact_8A")):
        mism += pc.sum(pc.not_equal(pc.less_equal(t.column("d_min"), thr),
                                    t.column(col))).as_py() or 0
    del t
check(20, "flags match d_min", mism == 0, f"{mism} disagreeing rows")

t = pq.read_table(f"{L}/labels_chai1_crystal_contacts.parquet",
                  columns=["contact_4A", "contact_5A", "contact_8A"]).to_pydict()
n4, n5, n8 = (np.array(t[k]) for k in ("contact_4A", "contact_5A", "contact_8A"))
check(21, "flags nested", bool((~n4 | n5).all() and (~n5 | n8).all()),
      f"counts {n4.sum():,} {n5.sum():,} {n8.sum():,}")
del t, n4, n5, n8

off = tot = 0
for cp in sorted(cont_f)[:4]:
    a = np.asarray(pq.read_table(cp, columns=["d_min"]).column(0).slice(0, 200_000)).astype(np.float32)
    tot += a.size
    off += int((a != a.astype(np.float16).astype(np.float32)).sum())
check(22, "half-precision grid", off == 0, f"{tot:,} values sampled, {off} off grid")

arms = ["smina_corpus", "chai1_corpus", "boltz2_corpus", "boltz2_msa_corpus",
        "crystal_groundtruth", "smina_crystal", "chai1_crystal", "boltz2_crystal",
        "boltz2_msa_crystal"]
bad = []
for k in arms:
    m = meta(k)
    with_rows = set(cont(k).column("system_id").to_pylist())
    n = sum(1 for s, h in zip(m.column("system_id").to_pylist(),
                              m.column("has_contacts").to_pylist())
            if h is not None and h != (s in with_rows))
    if n: bad.append((k, n))
check(23, "has_contacts", not bad, f"{len(bad)} arms disagree {bad[:3]}")

bad = []
for p in meta_f:
    t = pq.read_table(p, columns=["teacher", "tier"])
    for c in ("teacher", "tier"):
        if len(pc.unique(t.column(c))) != 1: bad.append(f"{os.path.basename(p)}:{c}")
check(24, "teacher and tier constant", not bad, f"{len(bad)} files vary {bad[:2]}")

# n_lig_heavy is sometimes the full chemical component count rather than the
# modelled atom count, which the field dictionary states. The test is the rate.
bad = done = 0
for k in ("chai1_crystal", "boltz2_crystal", "crystal_groundtruth"):
    m = dict(zip(meta(k).column("system_id").to_pylist(),
                 meta(k).column("n_lig_heavy").to_pylist()))
    mx = cont(k).group_by("system_id").aggregate([("atom_idx", "max")])
    for s, a in zip(mx.column(0).to_pylist(), mx.column(1).to_pylist()):
        if m.get(s) is None: continue
        done += 1
        if a >= m[s]: bad += 1
check(25, "n_lig_heavy rate", done > 0 and bad / done < 0.05,
      f"{done:,} systems, {bad} exceed ({100*bad/max(1,done):.2f}%)")

ccd = pq.read_table(f"{L}/labels_crystal_groundtruth_meta.parquet",
                    columns=["ligand_ccd"]).column(0).to_pylist()
odd = sorted({c for c in ccd if c and not (1 <= len(c) <= 5 and c.isalnum())})
check(26, "CCD codes", not odd,
      f"{len({c for c in ccd if c}):,} distinct, malformed {odd[:5] or 'none'}")

ik = pq.read_table(f"{L}/labels_smina_corpus_meta.parquet",
                   columns=["inchikey"]).column(0).to_pylist()
badk = [k for k in ik if k is not None and not
        (len(k) == 27 and k[14] == "-" and k[25] == "-" and k.replace("-", "").isalnum())]
check(27, "InChIKeys", not badk,
      f"{len({k for k in ik if k}):,} distinct, malformed {badk[:3] or 'none'}")

li = json.load(open(f"{D}/metadata/ligand_identity.json"))
m = pq.read_table(f"{L}/labels_smina_corpus_meta.parquet",
                  columns=["system_id", "protein_id", "inchikey"]).to_pydict()
dis = n = 0
for s, a_, k in zip(m["system_id"], m["protein_id"], m["inchikey"]):
    e = li.get(s)
    if not e: continue
    n += 1
    if e.get("accession") != a_ or e.get("inchikey") != k: dis += 1
check(28, "ligand identity map", dis == 0, f"{n:,} compared, {dis} disagree")

# -------------------------------------------------------------------- store --

sid0 = list(fx)[0]
g0 = fx[sid0]
xyz = g0["ligand_coords"][0, 0] / g0.attrs["scale"] + g0.attrs["centroid_xyz"]
check(29, "store self-describing",
      {"ligand_coords", "ligand_valid", "pose_score", "protein"} <= set(g0)
      and {"smiles", "ccd", "centroid_xyz", "scale", "teachers", "score_kind"} <= set(g0.attrs)
      and np.isfinite(xyz).all(),
      f"{sid0}: ligand spans {np.ptp(xyz, axis=0).round(1).tolist()} A")

check(30, "store contents", len(fc) == 31_617 and len(fx) == 17_368,
      f"corpus {len(fc):,} at {os.path.getsize(f'{D}/stores/corpus.h5')/1e9:.2f} GB, "
      f"crystal {len(fx):,} at {os.path.getsize(f'{D}/stores/crystal.h5')/1e9:.2f} GB")

scales, teach, kinds = set(), set(), set()
for store in (fx, fc):
    for sid in random.sample(list(store), 300):
        a = store[sid].attrs
        scales.add(int(a["scale"]))
        teach.add(tuple(x.decode() if isinstance(x, bytes) else x for x in a["teachers"]))
        kinds.add(tuple(x.decode() if isinstance(x, bytes) else x for x in a["score_kind"]))
check(31, "store attributes", scales == {100} and teach == {tuple(TEACHERS)} and len(kinds) == 1,
      f"scale {scales}, teacher axis in documented order, one score_kind tuple")

missing = misaligned = 0
for sid in list(fx)[:300]:
    for t in fx[sid].get("protein", {}):
        g = fx[sid][f"protein/{t}"]
        need = ("ca", "sc_centroid", "sc_valid", "pep_c", "res_types")
        if any(k not in g for k in need): missing += 1; continue
        n = g["ca"].shape[0]
        if not all(g[k].shape[0] == n for k in need): misaligned += 1
check(32, "per-residue arrays", missing == 0 and misaligned == 0,
      f"{missing} incomplete, {misaligned} misaligned")

gly_n = gly_bad = 0
for sid in list(fx)[:300]:
    for t in fx[sid].get("protein", {}):
        g = fx[sid][f"protein/{t}"]
        is_gly = np.array([x.decode() == "GLY" for x in g["res_types"][:]])
        gly_n += int(is_gly.sum())
        gly_bad += int(g["sc_valid"][:][is_gly].sum())
check(33, "glycine side chain", gly_bad == 0, f"{gly_bad} of {gly_n:,} marked valid")

worst_pep = worst_sc = 0.0
ngroups = 0
for store in (fc, fx):
    for sid in random.sample(list(store), 400):
        g = store[sid]
        for t in g.get("protein", {}):
            gg = g[f"protein/{t}"]
            if "pep_c" not in gg: continue
            s = g.attrs["scale"]
            ca = gg["ca"][:] / s
            worst_pep = max(worst_pep, float(np.linalg.norm(gg["pep_c"][:] / s - ca, axis=1).max()))
            worst_sc = max(worst_sc, float(np.linalg.norm(gg["sc_centroid"][:] / s - ca, axis=1).max()))
            ngroups += 1
check(34, "bead geometry", worst_pep < 25.0 and worst_sc < 12.0,
      f"{ngroups} groups, max pep_c offset {worst_pep:.2f} A, "
      f"max side-chain offset {worst_sc:.2f} A")

ns = collections.Counter()
for sid in list(fx)[:300]:
    for t in fx[sid].get("protein", {}):
        for r in fx[sid][f"protein/{t}"]["res_types"][:]:
            if r.decode() not in AA: ns[r.decode()] += 1
check(35, "residue alphabet", not ns, f"non-standard {ns.most_common(4) or 'none'}")

sid = next(s for s in fx if "protein/crystal" in fx[s]
           and "shell_coords" in fx[s]["protein/crystal"])
g = fx[sid]["protein/crystal"]
n = g["ca"].shape[0]
three = np.stack([g["ca"][:], g["sc_centroid"][:], g["pep_c"][:]], axis=1)
check(36, "reduced views",
      three.shape == (n, 3, 3) and np.array_equal(three[:, 0], g["ca"][:])
      and g["shell_coords"][:].ndim == 2,
      f"{sid}: {n} residues, three-bead {three.shape}, "
      f"all-atom {g['shell_coords'].shape}")

bad = done = 0
worst = 0.0
for store in (fx, fc):
    for sid in random.sample(list(store), 120):
        g = store[sid]
        for t in g.get("protein", {}):
            gg = g[f"protein/{t}"]
            if "shell_res_index" not in gg: continue
            done += 1
            ri = gg["shell_res_index"][:]
            if ri.min() < 0 or ri.max() >= gg["ca"].shape[0]: bad += 1
            sl = SLOT.get(t)
            if sl is None or not g["ligand_valid"][sl].any(): continue
            s = g.attrs["scale"]
            lig = g["ligand_coords"][sl, 0] / s
            sh = gg["shell_coords"][:] / s
            worst = max(worst, float(np.linalg.norm(sh[:, None, :] - lig[None, :, :],
                                                    axis=2).min(1).max()))
check(37, "shell", bad == 0 and worst <= 15.05,
      f"{done} groups, {bad} indexing outside the residue arrays, "
      f"furthest shell atom {worst:.2f} A")

nz = done = 0
for store in (fx, fc):
    for sid in random.sample(list(store), 200):
        g = store[sid]
        done += 1
        if g["ligand_coords"][:][~g["ligand_valid"][:]].any(): nz += 1
check(38, "invalid slots", nz == 0, f"{done} systems, {nz} with stray coordinates")

# --------------------------------------------------------------- derivation --

m = pq.read_table(f"{L}/labels_crystal_groundtruth_meta.parquet",
                  columns=["system_id", "n_res"])
nres = dict(zip(m.column(0).to_pylist(), m.column(1).to_pylist()))
ok = bad = 0
for s in list(fx)[:500]:
    if "protein/crystal" in fx[s] and s in nres:
        if len(fx[s]["protein/crystal"]["ca"]) == nres[s]: ok += 1
        else: bad += 1
check(39, "residue count", bad == 0, f"{ok} agree with meta, {bad} differ")

t = pq.read_table(f"{L}/labels_crystal_groundtruth_contacts.parquet",
                  columns=["system_id", "res_row"])
mx = t.group_by("system_id").aggregate([("res_row", "max")])
bad = seen = 0
for sid, rmax in zip(mx.column(0).to_pylist(), mx.column(1).to_pylist()):
    if sid in fx and "protein/crystal" in fx[sid]:
        seen += 1
        if rmax >= len(fx[sid]["protein/crystal"]["ca"]): bad += 1
sup = pq.read_table(f"{D}/metadata/residue_support.parquet")
sup = sup.filter(pc.equal(sup.column("tier"), "crystal"))
mx2 = sup.group_by("system_id").aggregate([("res_row", "max")])
bad2 = sum(1 for s, v in zip(mx2.column(0).to_pylist(), mx2.column(1).to_pylist())
           if s in nres and v >= nres[s])
check(40, "res_row range", bad == 0 and bad2 == 0,
      f"{seen:,} systems in labels and {mx2.num_rows:,} in support, "
      f"{bad + bad2} out of range")

# The store resolves symmetry-equivalent atom matches against a per-system
# anchor while the label extractor took the first substructure match, so on a
# symmetric ligand the two legitimately disagree. Compare the median.
sid = next(s for s in list(fx)[:800]
           if "protein/chai1" in fx[s] and fx[s]["ligand_valid"][1].any())
g = fx[sid]
s = g.attrs["scale"]
cen = g.attrs["centroid_xyz"]
lig = g["ligand_coords"][1, 0] / s + cen
ca = g["protein/chai1"]["ca"][:] / s + cen
tt = cont("chai1_crystal")
sub = tt.filter(pc.equal(tt.column("system_id"), sid)).to_pydict()
d = np.linalg.norm(lig[np.array(sub["atom_idx"])] - ca[np.array(sub["res_row"])], axis=1)
err = float(np.median(np.abs(d - np.array(sub["d_ca"]))))
check(41, "tables derive from store", err < 0.05,
      f"{sid}: {len(d):,} pairs, median difference {err:.4f} A")

# -------------------------------------------------------------- reliability --

rel = pq.read_table(f"{D}/metadata/system_reliability.parquet").to_pandas()
hw = rel.conformal_halfwidth_90.unique()
check(42, "reliability field",
      rel.pred_accuracy.between(0, 1).all() and len(hw) == 1 and hw[0] > 0,
      f"n={len(rel):,} {dict(collections.Counter(rel.tier))}, "
      f"range [{rel.pred_accuracy.min():.3f}, {rel.pred_accuracy.max():.3f}], "
      f"half-width {hw[0]:.4f}")

# Monotone within each tier, not only in aggregate: the fit is made on crystal
# and applied to corpus, so the two need checking separately.
out = []
for tier, gg in rel.groupby("tier"):
    a = gg.agreement.to_numpy()
    p = gg.pred_accuracy.to_numpy()
    q = np.quantile(a, np.linspace(0, 1, 6))
    mu = [float(p[(a >= q[i]) & (a <= q[i + 1])].mean()) for i in range(5)]
    out.append((tier, all(mu[i] <= mu[i + 1] for i in range(4)), [round(x, 3) for x in mu]))
check(43, "reliability monotone", all(o[1] for o in out),
      "; ".join(f"{t} {m}" for t, _, m in out))

sc = pq.read_table(f"{D}/metadata/residue_support.parquet",
                   columns=["n_teachers_asserting"]).column(0)
check(44, "support count", pc.min(sc).as_py() >= 1 and pc.max(sc).as_py() <= 3,
      f"[{pc.min(sc).as_py()}, {pc.max(sc).as_py()}]")

gt = cont("crystal_groundtruth")
def contact_sets(tbl, ids):
    d = tbl.filter(pc.is_in(tbl.column("system_id"), pa.array(sorted(ids))))
    d = d.filter(pc.equal(d.column("contact_4A"), True)).to_pydict()
    out = collections.defaultdict(set)
    for s, r in zip(d["system_id"], d["res_row"]): out[s].add(r)
    return out
core = (set(meta("smina_crystal").column("system_id").to_pylist())
        & set(meta("boltz2_crystal").column("system_id").to_pylist())
        & set(meta("chai1_crystal").column("system_id").to_pylist())
        & set(meta("crystal_groundtruth").column("system_id").to_pylist()))
sample = set(random.sample(sorted(core), min(400, len(core))))
G = contact_sets(gt, sample)
supd = pq.read_table(f"{D}/metadata/residue_support.parquet")
supd = supd.filter(pc.equal(supd.column("tier"), "crystal")).to_pydict()
byn = collections.defaultdict(lambda: [0, 0])
for s, r, k in zip(supd["system_id"], supd["res_row"], supd["n_teachers_asserting"]):
    if s not in sample or s not in G: continue
    byn[k][0] += 1
    if r in G[s]: byn[k][1] += 1
prec = {k: (v[1] / v[0] if v[0] else 0.0) for k, v in sorted(byn.items())}
check(45, "support precision",
      len(prec) >= 3 and prec.get(1, 1) < prec.get(2, 0) < prec.get(3, 0),
      ", ".join(f"{k} teacher(s) {v:.3f}" for k, v in prec.items()))

# ------------------------------------------------------------------- splits --

bad = {}
for tier, kn in (("corpus", corpus_ids), ("crystal", crystal_ids)):
    p = f"{D}/splits/splits_{tier}.parquet"
    if not os.path.exists(p): continue
    s = set(pq.read_table(p, columns=["system_id"]).column(0).to_pylist())
    if s - kn: bad[tier] = len(s - kn)
check(46, "split ids", not bad, f"unknown ids {bad or 'none'}")

prob = []
for tier in ("corpus", "crystal"):
    p = f"{D}/splits/splits_{tier}.parquet"
    if not os.path.exists(p): continue
    d = pq.read_table(p, columns=["split_tag", "system_id"]).to_pydict()
    per_tag = collections.Counter(d["split_tag"])
    seen = collections.defaultdict(set)
    dups = 0
    for tg, s in zip(d["split_tag"], d["system_id"]):
        if s in seen[tg]: dups += 1
        seen[tg].add(s)
    if len(set(per_tag.values())) != 1 or dups:
        prob.append(f"{tier}: {len(per_tag)} tags, sizes {sorted(set(per_tag.values()))}, "
                    f"{dups} duplicates")
check(47, "split partitions", not prob, prob or "each tag covers its tier exactly once")

vals = set()
for p in splits:
    if "fold" in pq.ParquetFile(p).schema_arrow.names:
        vals |= set(pq.read_table(p, columns=["fold"]).column(0).to_pylist())
check(48, "fold vocabulary", vals <= {"train", "val", "test", "excluded"},
      f"observed {sorted(vals)}")

prob = []
for tier in ("corpus", "crystal"):
    fp, sp = f"{D}/splits/splits_{tier}.parquet", f"{D}/splits/split_summary_{tier}.parquet"
    if not (os.path.exists(fp) and os.path.exists(sp)): continue
    ft = set(pq.read_table(fp, columns=["split_tag"]).column(0).to_pylist())
    st = set(pq.read_table(sp, columns=["split_tag"]).column(0).to_pylist())
    if ft != st: prob.append(f"{tier}: {len(ft ^ st)} tags not in both")
check(49, "split summaries", not prob, prob or "every configuration has a summary row")

fam = set()
for p in splits:
    if "family" in pq.ParquetFile(p).schema_arrow.names:
        fam |= set(pq.read_table(p, columns=["family"]).column(0).to_pylist())
undoc = sorted(f for f in fam if f not in README)
check(50, "split families", not undoc,
      f"{sorted(fam)}, undocumented {undoc or 'none'}")

# ------------------------------------------------- against source structures --

s2a = json.load(open(SYS2ACC)) if os.path.exists(SYS2ACC) else {}

if not (HAVE_GEMMI and os.path.isdir(STRUCTURES)):
    for n, name in ((51, "res_types against source"), (52, "ca against source"),
                    (53, "side-chain centroid against source"),
                    (54, "peptide carbon against source"),
                    (55, "sc_valid explained")):
        skip(n, name, "needs gemmi and PLIP_STRUCTURES")
else:
    picked = random.sample(list(fx), 25)
    parsed = {}
    for sid in picked:
        if "protein/crystal" not in fx[sid]: continue
        p = f"{STRUCTURES}/{sid.split('_')[0]}.cif"
        if not os.path.exists(p): continue
        try: parsed[sid] = residue_gate(gemmi.read_structure(p))
        except Exception: pass

    bad = 0
    for sid, rs in parsed.items():
        got = [x.decode() for x in fx[sid]["protein/crystal"]["res_types"][:]]
        if [r.name for r in rs] != got: bad += 1
    check(51, "res_types against source", bad == 0 and parsed,
          f"{len(parsed)} systems, {bad} differ")

    bad, worst = 0, 0.0
    for sid, rs in parsed.items():
        got = dequant(fx[sid], "crystal", "ca")
        if len(rs) != len(got): bad += 1; continue
        src = np.array([[r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y,
                         r.find_atom("CA", "*").pos.z] for r in rs])
        dmax = float(np.abs(src - got).max())
        worst = max(worst, dmax)
        if dmax > 0.02: bad += 1
    check(52, "ca against source", bad == 0 and parsed,
          f"{len(parsed)} systems, worst {worst:.4f} A")

    bad, worst = 0, 0.0
    for sid, rs in parsed.items():
        got = dequant(fx[sid], "crystal", "sc_centroid")
        if len(rs) != len(got): bad += 1; continue
        for r, want in zip(rs, got):
            sc = [a for a in r if a.name not in BACKBONE and a.element.name not in ("H", "D")]
            pts = sc if sc else [a for a in r if a.element.name not in ("H", "D")]
            c = np.mean([[a.pos.x, a.pos.y, a.pos.z] for a in pts], axis=0)
            dd = float(np.linalg.norm(c - want))
            worst = max(worst, dd)
            if dd > 0.02: bad += 1
    check(53, "side-chain centroid against source", bad == 0 and parsed,
          f"{len(parsed)} systems, worst {worst:.4f} A")

    bad, worst = 0, 0.0
    for sid, rs in parsed.items():
        got = dequant(fx[sid], "crystal", "pep_c")
        if len(rs) != len(got): bad += 1; continue
        for r, want in zip(rs, got):
            c = r.find_atom("C", "*")
            if c is None: continue
            dd = float(np.linalg.norm(np.array([c.pos.x, c.pos.y, c.pos.z]) - want))
            worst = max(worst, dd)
            if dd > 0.02: bad += 1
    check(54, "peptide carbon against source", bad == 0 and parsed,
          f"{len(parsed)} systems, worst {worst:.4f} A")

    wrong = nfalse = 0
    for sid, rs in parsed.items():
        sv = fx[sid]["protein/crystal"]["sc_valid"][:]
        if len(rs) != len(sv): continue
        for r, v in zip(rs, sv):
            if v: continue
            nfalse += 1
            sc = [a for a in r if a.name not in BACKBONE and a.element.name not in ("H", "D")]
            if r.name != "GLY" and sc: wrong += 1
    check(55, "sc_valid explained", wrong == 0,
          f"{nfalse} marked invalid, {wrong} with a modelled side chain")

if not (HAVE_GEMMI and os.path.isdir(AF) and s2a):
    skip(56, "docking receptor against AlphaFold", "needs gemmi, PLIP_AF and PLIP_SYS2ACC")
else:
    cands = [s for s in list(fc)[:6000]
             if "protein" in fc[s] and list(fc[s]["protein"]) == ["smina"]]
    bad = done = 0
    worst = 0.0
    for sid in random.sample(cands, min(15, len(cands))):
        acc = s2a.get(sid)
        p = f"{AF}/AF-{acc}-F1-model_v6.pdb" if acc else None
        if not p or not os.path.exists(p): continue
        try: rs = residue_gate(gemmi.read_structure(p))
        except Exception: continue
        got = dequant(fc[sid], "smina", "ca")
        if len(rs) != len(got): bad += 1; continue
        done += 1
        src = np.array([[r.find_atom("CA", "*").pos.x, r.find_atom("CA", "*").pos.y,
                         r.find_atom("CA", "*").pos.z] for r in rs])
        dmax = float(np.abs(src - got).max())
        worst = max(worst, dmax)
        if dmax > 0.02: bad += 1
    check(56, "docking receptor against AlphaFold", bad == 0 and done > 0,
          f"{done} docking-only systems, worst {worst:.4f} A")

if not s2a:
    skip(57, "corpus residue count", "needs PLIP_SYS2ACC")
else:
    counts = collections.Counter(pq.read_table(f"{D}/metadata/residues_corpus.parquet",
                                               columns=["accession"]).column(0).to_pylist())
    bad = done = 0
    for sid in random.sample(list(fc), 300):
        acc = s2a.get(sid)
        if not acc or acc not in counts: continue
        if "protein/smina" not in fc[sid]: continue
        done += 1
        if fc[sid]["protein/smina"]["ca"].shape[0] != counts[acc]: bad += 1
    check(57, "corpus residue count", bad == 0,
          f"{done} docking receptors compared, {bad} differ")

# ------------------------------------------------------------- measurements --

print()
t = pq.read_table(f"{L}/labels_crystal_groundtruth_contacts.parquet",
                  columns=["system_id", "res_row"]).to_pydict()
per_sys = collections.defaultdict(set)
for s, r in zip(t["system_id"], t["res_row"]): per_sys[s].add(r)
frac = [len(per_sys[s]) / nres[s] for s in list(per_sys)[:3000] if nres.get(s)]
note("censoring", f"a system carries rows for {100*np.mean(frac):.1f}% of its residues; "
     f"the rest lie beyond the cutoff and are not measured non-contacts, so an "
     f"outer join filling zero records contacts that were never observed")

err = []
for sid in random.sample(list(fx), 60):
    g = fx[sid]
    if "protein/crystal" not in g: continue
    q = g["protein/crystal"]["ca"][:]
    err.append(float(np.abs(g.attrs["centroid_xyz"]).max()))
note("decode", f"dropping the centroid from the decode shifts coordinates by a median "
     f"{np.median(err):.1f} A and raises nothing; dropping the scale inflates every "
     f"distance a hundredfold and is obvious")

diffs = []
for sid in random.sample(list(fx), 400):
    g = fx[sid]
    if "protein/crystal" not in g or "protein/boltz2" not in g: continue
    a, b = g["protein/crystal"]["ca"][:], g["protein/boltz2"]["ca"][:]
    if a.shape != b.shape: continue
    diffs.append(float(np.median(np.linalg.norm((a - b) / g.attrs["scale"], axis=1))))
if diffs:
    note("frames", f"crystal against boltz2 over {len(diffs)} paired systems, median "
         f"per-residue offset {np.median(diffs):.1f} A; superpose before comparing")

tp = {}
for thr in (4.0, 5.0, 8.0, 15.0):
    tp[thr] = sum(pc.sum(pc.equal(pq.read_table(cp, columns=["d_min"]).column(0),
                                  pa.scalar(np.float32(thr)))).as_py() or 0
                  for cp in cont_f)
note("threshold proximity", ", ".join(f"{k:g} A {v:,}" for k, v in tp.items()))

lo = float(rel.pred_accuracy.min())
note("isotonic floor", f"{int((rel.pred_accuracy <= lo + 1e-9).sum()):,} of {len(rel):,} "
     f"systems sit at {lo:.4f}, the lower bound of the fit rather than a measurement")

sp = f"{D}/splits/split_summary_crystal.parquet"
if os.path.exists(sp):
    d = pq.read_table(sp).to_pandas()
    cols = [c for c in ("c2st_auc", "c2st_auc_protein", "c2st_auc_protein_length",
                        "max_protein_identity_cov80", "max_protein_identity_local")
            if c in d.columns]
    if cols:
        note("crystal split diagnostics", "\n" + d.groupby("family")[cols].mean().round(4).to_string())

fx.close(); fc.close()
print()
print(f"failures: {fails if fails else 'none'}")
if skipped: print(f"skipped: {len(skipped)}")
sys.exit(1 if fails else 0)
