#!/usr/bin/env python3
"""Verify that the coordinate store's ligand atom axis is the deposited one.

For each (tier, teacher) cell, take the deposited contacts table, index the
store's ligand coordinates with `atom_idx` DIRECTLY, recompute d_ca and d_min,
and compare. A store whose atom axis agrees with the tables differs only by
quantisation: the store holds int16 at 0.01 A (+-0.005 A per coordinate) and the
tables hold float16 distances (~0.002 A at 10 A), so the expected mean |delta| is
~0.004-0.006 A and one int16 LSB, 0.01 A, is the pass threshold.

Where a system exceeds it, we ask whether the residual is a graph AUTOMORPHISM of
the reference molecule. The store resolves symmetry-equivalent matches against a
per-system anchor pose so one atom slot is the same physical atom across all
teachers; the label extractors took the first substructure match independently
per teacher. On a symmetric ligand those two rules legitimately disagree, and the
atoms involved are interchangeable. That is symmetry, not a correspondence error,
and is counted separately from genuine mismatch.

Protein sources for the recompute, chosen to be the structure the labels were
extracted from:
  corpus chai1/boltz2/boltz2_msa : the store's own protein/<teacher> group
  corpus smina                   : the AlphaFold receptor smina was docked into
  crystal chai1/boltz2/boltz2_msa: the store's own protein/<teacher> group
  crystal smina, crystal GT      : the store's protein/crystal group (the mmCIF)
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np, pandas as pd
import h5py, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_coord_store as B

TEACHERS = B.TEACHERS
FLOOR = 0.01            # one int16 LSB of the store's 0.01 A grid
MAX_AUTO = 4096         # cap on automorphisms enumerated per ligand
AF_PDBS = "/workspace/docking/input/pdbs/AF-{acc}-F1-model_v6.pdb"

CELLS = [
    ("corpus",  "chai1",      "labels_chai1_corpus_contacts.parquet"),
    ("corpus",  "boltz2",     "labels_boltz2_corpus_contacts.parquet"),
    ("corpus",  "boltz2_msa", "labels_boltz2_msa_corpus_contacts.parquet"),
    ("corpus",  "smina",      "labels_smina_corpus_contacts.parquet"),
    ("crystal", "crystal",    "labels_crystal_groundtruth_contacts.parquet"),
    ("crystal", "chai1",      "labels_chai1_crystal_contacts.parquet"),
    ("crystal", "boltz2",     "labels_boltz2_crystal_contacts.parquet"),
    ("crystal", "boltz2_msa", "labels_boltz2_msa_crystal_contacts.parquet"),
    ("crystal", "smina",      "labels_smina_crystal_contacts.parquet"),
]
PROT_GROUP = {("corpus", "smina"): None, ("crystal", "smina"): "crystal",
              ("crystal", "crystal"): "crystal"}


def scan(path, sids):
    """Rows of `path` for `sids`. The tables are not sorted by system_id, so this
    is a full pass; one pass per file serves every sampled system of that tier."""
    pf = pq.ParquetFile(path); vs = pa.array(sorted(sids)); out = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["system_id", "res_row", "atom_idx", "d_ca", "d_min"])
        m = pc.is_in(t.column("system_id"), value_set=vs)
        if pc.sum(m).as_py():
            out.append(t.filter(m).to_pandas())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def store_index(shard_glob):
    idx = {}
    for p in sorted(glob.glob(shard_glob)):
        with h5py.File(p, "r") as f:
            for sid in f:
                idx[sid] = p
    return idx


def load_af_protein(acc):
    """CA and per-residue heavy atoms exactly as run_smina_labels.parse_protein
    built them; the corpus smina labels were computed against this receptor."""
    p = AF_PDBS.format(acc=acc)
    if not os.path.exists(p):
        return None, None
    res, seen = [], {}
    for ln in open(p):
        if not ln.startswith("ATOM"):
            continue
        if ln[16] not in (" ", "A"):
            continue
        name = ln[12:16].strip(); elem = (ln[76:78].strip() or name[:1]).upper()
        if elem == "H":
            continue
        key = (ln[21], ln[22:26].strip())
        if key not in seen:
            seen[key] = len(res); res.append({"ca": None, "heavy": []})
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        r = res[seen[key]]; r["heavy"].append(xyz)
        if name == "CA":
            r["ca"] = xyz
    ca = np.array([r["ca"] if r["ca"] else r["heavy"][0] for r in res], np.float64)
    return ca, [np.array(r["heavy"], np.float64) for r in res]


def automorphisms(ref):
    if ref is None:
        return None
    try:
        ms = ref.GetSubstructMatches(ref, uniquify=False, maxMatches=MAX_AUTO)
    except Exception:
        return None
    return [np.asarray(m, int) for m in ms] or None


def check_system(g, teacher, tab, ref, prot):
    """Recompute d_ca/d_min from the store, indexing ligand coords with atom_idx
    directly. Returns a per-system record."""
    ti = TEACHERS.index(teacher)
    if not g["ligand_valid"][ti].any():
        return dict(status="no_pose")
    cen = g.attrs["centroid_xyz"].astype(np.float64)
    lig = g["ligand_coords"][ti, 0].astype(np.float64) / B.SCALE + cen
    n = lig.shape[0]

    rr = tab["res_row"].to_numpy(); aa = tab["atom_idx"].to_numpy().astype(int)
    dca = tab["d_ca"].to_numpy(np.float64); dmin = tab["d_min"].to_numpy(np.float64)
    if aa.max() >= n:
        return dict(status="atom_count", n_store=n, n_tab=int(aa.max()) + 1)

    ca, heavy_by_res, shell_xyz, shell_ri = prot
    if ca is None or rr.max() >= len(ca):
        return dict(status="no_protein" if ca is None else "res_range",
                    n_store=n, n_ca=(0 if ca is None else len(ca)), rr_max=int(rr.max()))

    def dca_err(perm):
        x = lig[perm[aa]] if perm is not None else lig[aa]
        return np.abs(np.linalg.norm(ca[rr] - x, axis=1) - dca)

    e = dca_err(None)
    rec = dict(status="ok", n_store=n, n_rows=len(rr),
               dca_mean=float(e.mean()), dca_max=float(e.max()))

    # d_min: needs the residue's heavy atoms. Use the shell where the store has
    # one (it always contains the minimising atom for a row with d_min <= 15 A),
    # else the receptor parsed above.
    if shell_xyz is not None:
        order = np.argsort(shell_ri, kind="stable")
        sri = shell_ri[order]; sxyz = shell_xyz[order]
        bounds = np.searchsorted(sri, np.arange(len(ca) + 1))
        get = lambda r: sxyz[bounds[r]:bounds[r + 1]]
    else:
        get = lambda r: heavy_by_res[r]
    em, nm = [], 0
    for r in np.unique(rr):
        hv = get(r)
        if len(hv) == 0:
            continue
        sel = rr == r
        d = np.linalg.norm(hv[:, None, :] - lig[None, aa[sel], :], axis=2).min(0)
        em.append(np.abs(d - dmin[sel])); nm += int(sel.sum())
    if em:
        em = np.concatenate(em)
        rec.update(dmin_mean=float(em.mean()), dmin_max=float(em.max()), dmin_rows=nm)

    # Above the floor: is the residual an automorphism of the ligand graph?
    if rec["dca_mean"] > FLOOR:
        autos = automorphisms(ref)
        if autos and len(autos) > 1 and len(autos[0]) == n:
            best = min((float(dca_err(s).mean()), i) for i, s in enumerate(autos))
            rec["dca_mean_best_auto"] = best[0]
            rec["n_auto"] = len(autos)
            rec["symmetry_explained"] = bool(best[0] <= FLOOR)
        else:
            rec["n_auto"] = 0 if not autos else len(autos)
            rec["symmetry_explained"] = False
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="dir holding corpus/ and crystal/ shards")
    ap.add_argument("--labels", default="/workspace/deposit_v3/labels")
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    smiles = {}
    smiles.update(B.load_smiles("/workspace/docking/output/boltz_in_index.tsv"))
    smiles.update(B.load_smiles("/workspace/datasets/experimental_expansion/exp_fold_index.tsv"))
    acc = {r["name"]: r["accession"] for r in
           csv.DictReader(open("/workspace/docking/output/boltz_in_index.tsv"), delimiter="\t")}
    # ~7.8k corpus store ids resolve their SMILES through the inchikey map and are
    # absent from boltz_in_index; the deposit's own meta carries their receptor.
    mt = pq.read_table(os.path.join(A.labels, "labels_smina_corpus_meta.parquet"),
                       columns=["system_id", "protein_id"]).to_pandas()
    for sid_, pid_ in zip(mt.system_id, mt.protein_id):
        acc.setdefault(sid_, pid_)

    stores = {t: store_index(f"{A.store}/{t}/*.h5") for t in ("corpus", "crystal")}
    print({t: len(v) for t, v in stores.items()})

    # sample: for each cell, the first --per-cell store systems that the table covers
    want = {t: set() for t in stores}
    tabs = {}
    for tier, teacher, fn in CELLS:
        pool = sorted(stores[tier])
        got = scan(os.path.join(A.labels, fn), pool)
        if got.empty:
            tabs[(tier, teacher)] = {}
            continue
        by = {s: g for s, g in got.groupby("system_id")}
        # prefer systems that actually carry a pose for this teacher, so a cell
        # is not spent on rows the store cannot answer
        ti = TEACHERS.index(teacher)
        with_pose = []
        for sid in sorted(by):
            with h5py.File(stores[tier][sid], "r") as f:
                if f[sid]["ligand_valid"][ti].any():
                    with_pose.append(sid)
            if len(with_pose) >= A.per_cell:
                break
        keep = with_pose if with_pose else sorted(by)[: A.per_cell]
        tabs[(tier, teacher)] = {s: by[s] for s in keep}
        want[tier] |= set(keep)
        print(f"  {tier}/{teacher}: table covers {len(by)} sampled store systems, using {len(keep)}")

    rows = []
    for tier, teacher, fn in CELLS:
        sel = tabs[(tier, teacher)]
        for sid, tab in sel.items():
            path = stores[tier][sid]
            with h5py.File(path, "r") as f:
                g = f[sid]
                cen = g.attrs["centroid_xyz"].astype(np.float64)
                pg = PROT_GROUP.get((tier, teacher), teacher)
                ca = shell_xyz = shell_ri = None; heavy = None
                if pg is not None and "protein" in g and pg in g["protein"]:
                    gp = g["protein"][pg]
                    ca = gp["ca"][:].astype(np.float64) / B.SCALE + cen
                    if "shell_coords" in gp:
                        shell_xyz = gp["shell_coords"][:].astype(np.float64) / B.SCALE + cen
                        shell_ri = gp["shell_res_index"][:]
                elif pg is None:
                    ca, heavy = load_af_protein(acc.get(sid, ""))
                smi = g.attrs["smiles"] or smiles.get(sid, "")
                ref = B.ref_from_smiles(smi) if smi else None
                rec = check_system(g, teacher, tab, ref,
                                   (ca, heavy, shell_xyz, shell_ri))
            rec.update(tier=tier, teacher=teacher, system_id=sid)
            rows.append(rec)
        print(f"  done {tier}/{teacher}")

    df = pd.DataFrame(rows)
    df.to_csv(A.out, index=False)
    print("\nwrote", A.out, len(df), "rows,", df.system_id.nunique(), "distinct systems")
    ok = df[df.status == "ok"]
    print("\nstatus:", df.status.value_counts().to_dict())
    if len(ok):
        print("\nper-cell d_ca mean |delta| (A):")
        piv = ok.groupby(["tier", "teacher"]).agg(
            n=("system_id", "size"), mean=("dca_mean", "mean"),
            p50=("dca_mean", "median"), p95=("dca_mean", lambda s: s.quantile(.95)),
            mx=("dca_mean", "max"), pass_floor=("dca_mean", lambda s: int((s <= FLOOR).sum())))
        print(piv.to_string())
        if "dmin_mean" in ok:
            print("\nper-cell d_min mean |delta| (A):")
            print(ok.dropna(subset=["dmin_mean"]).groupby(["tier", "teacher"]).agg(
                n=("system_id", "size"), mean=("dmin_mean", "mean"),
                mx=("dmin_mean", "max"),
                pass_floor=("dmin_mean", lambda s: int((s <= FLOOR).sum()))).to_string())
        bad = ok[ok.dca_mean > FLOOR]
        print(f"\nabove floor: {len(bad)} of {len(ok)}")
        if len(bad) and "symmetry_explained" in bad:
            se = bad.symmetry_explained.fillna(False).astype(bool)
            print("  symmetry-explained:", int(se.sum()), " unexplained:", int((~se).sum()))
            if (~se).any():
                cols = [c for c in ["tier", "teacher", "system_id", "n_store", "n_rows",
                                    "dca_mean", "dca_mean_best_auto", "n_auto"] if c in bad]
                print(bad[~se][cols].sort_values("dca_mean", ascending=False).head(25).to_string(index=False))
    print("\nstatus by cell:")
    print(df.groupby(["tier", "teacher", "status"]).size().to_string())


if __name__ == "__main__":
    main()
