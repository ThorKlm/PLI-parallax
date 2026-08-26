"""Recompute d_ca and d_min from coord_shards_v2 and compare to the deposited
pair tables.

The store keeps int16 fixed-point at 0.01 A about a per-system centroid, so a
recomputed distance carries up to ~2*sqrt(3)*0.005 = 0.017 A of quantisation
error; the deposited values are float16 (spacing 0.0078 A at 14 A, so +-0.0039).
The combined budget is ~0.021 A.

The deposit includes exactly those (res_row, atom_idx) pairs with d_min <= 15 A,
which is the same 15 A shell the store retains, so the pair set is recoverable.
Ligand atom_idx ordering is NOT assumed to agree (the deposit builds its
reference from the largest SMILES fragment, the store from the whole ligand and
-- on the crystal tier -- from the crystal copy), so the primary comparison is
permutation-invariant: the sorted multiset of distances per system. Direct
element-wise agreement is reported separately where the orderings do coincide.
"""
import argparse, glob, json
import numpy as np, h5py, pyarrow.parquet as pq

SHELL = 15.0

def store_index(d):
    ix = {}
    for f in sorted(glob.glob(f"{d}/*.h5")):
        with h5py.File(f, "r") as h:
            for s in h:
                ix[s] = f
    return ix

def recompute(g, teacher):
    """(D_ca, D_min) over all residues x all ligand atoms, from the store."""
    c = g.attrs["centroid_xyz"].astype(np.float64); sc = float(g.attrs["scale"])
    lig = g["ligand_coords"][:][:, 0] / sc + c          # (nteach, natom, 3), pose 0
    ti = [t.decode() for t in g.attrs["teachers"]].index(teacher)
    L = lig[ti]
    p = g[f"protein/{teacher}"]
    ca = p["ca"][:] / sc + c
    D_ca = np.linalg.norm(ca[:, None, :] - L[None, :, :], axis=2)
    D_min = np.full_like(D_ca, np.inf)
    if "shell_coords" in p:
        sx = p["shell_coords"][:] / sc + c
        sr = p["shell_res_index"][:]
        d = np.linalg.norm(sx[:, None, :] - L[None, :, :], axis=2)   # (natoms, nlig)
        np.minimum.at(D_min, sr, d)
    return D_ca, D_min

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/workspace/coord_shards_v2/crystal")
    ap.add_argument("--contacts", default="/workspace/deposit_v3/labels/labels_boltz2_crystal_contacts.parquet")
    ap.add_argument("--teacher", default="boltz2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tol", type=float, default=0.021)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="shrink the 15 A inclusion threshold by this much so pairs "
                         "sitting within quantisation error of it cannot flip side")
    ap.add_argument("--out", default="/workspace/reports/coord_store_v2_verify.json")
    A = ap.parse_args()

    ix = store_index(A.store)
    dep = set(pq.read_table(A.contacts, columns=["system_id"])["system_id"].to_pylist())
    cand = sorted(dep & set(ix))
    rng = np.random.default_rng(0)
    pick, i = [], 0
    while len(pick) < A.n and i < len(cand):
        s = cand[rng.integers(0, len(cand))] if False else cand[i]
        i += 1
        with h5py.File(ix[s], "r") as h:
            if f"protein/{A.teacher}" in h[s] and h[s]["ligand_valid"][:][
                    [t.decode() for t in h[s].attrs["teachers"]].index(A.teacher)].any():
                pick.append(s)
    print(f"{len(pick)} systems selected from {len(cand)} with both store and deposit")

    rows = []
    for s in pick:
        t = pq.read_table(A.contacts, filters=[("system_id", "=", s)])
        rr = np.asarray(t["res_row"]); dca = np.asarray(t["d_ca"], dtype=np.float64)
        dmn = np.asarray(t["d_min"], dtype=np.float64)
        with h5py.File(ix[s], "r") as h:
            D_ca, D_min = recompute(h[s], A.teacher)
        if rr.max() >= D_ca.shape[0]:
            rows.append(dict(system_id=s, status="res_row_out_of_range",
                             dep_max_res_row=int(rr.max()), store_n_res=int(D_ca.shape[0])))
            continue
        # A pair whose true d_min sits within quantisation error of the 15 A
        # inclusion threshold can fall on either side of it, so compare on a
        # margin-shrunk threshold where membership is unambiguous.  d_ca is
        # clipped at 20 A in the deposit, so clipped pairs are excluded.
        TH = SHELL - A.margin
        keep = (dmn <= TH) & (dca < 20.0)
        sel = D_min <= TH
        r = dict(system_id=s, status="ok",
                 n_pairs_dep=int(keep.sum()), n_pairs_store=int(sel.sum()),
                 n_pairs_dep_all=len(dmn), n_pairs_store_all=int((D_min <= SHELL).sum()))
        # Residue-anchored: sorting runs only over the ligand-atom axis, so the
        # residue axis (res_row) has to agree exactly for this to pass.  Ligand
        # atom_idx ordering is not assumed (see module docstring).
        rrk, dcak, dmnk = rr[keep], dca[keep], dmn[keep]
        worst_ca = worst_min = 0.0; matched = mism = 0
        for res in np.unique(rrk):
            dv_ca = np.sort(dcak[rrk == res]); dv_mn = np.sort(dmnk[rrk == res])
            m = sel[res].copy()
            sv_ca_all = D_ca[res][m]
            m2 = sv_ca_all < 20.0
            sv_ca = np.sort(sv_ca_all[m2]); sv_mn = np.sort(D_min[res][m][m2])
            if len(dv_ca) != len(sv_ca):
                mism += 1; continue
            matched += 1
            worst_ca = max(worst_ca, float(np.abs(dv_ca - sv_ca).max()))
            worst_min = max(worst_min, float(np.abs(dv_mn - sv_mn).max()))
        r["n_res_matched"] = matched
        r["n_res_count_mismatch"] = mism
        r["dca_per_res_max_abs"] = worst_ca
        r["dmin_per_res_max_abs"] = worst_min
        if matched == 0:
            r["status"] = "no_matched_residues"
        # direct element-wise, only meaningful if atom_idx orderings coincide
        ai = np.asarray(t["atom_idx"])
        if ai.max() < D_ca.shape[1]:
            r["dca_direct_max_abs"] = float(np.abs(D_ca[rr, ai] - dca).max())
            r["dmin_direct_max_abs"] = float(np.abs(D_min[rr, ai] - dmn).max())
        rows.append(r)

    ok = [r for r in rows if r["status"] == "ok"]
    res = dict(store=A.store, contacts=A.contacts, teacher=A.teacher,
               tolerance_angstrom=A.tol, n_selected=len(pick),
               n_ok=len(ok),
               n_res_row_out_of_range=sum(r["status"] == "res_row_out_of_range" for r in rows),
               n_no_matched_residues=sum(r["status"] == "no_matched_residues" for r in rows))
    if ok:
        res["dca_per_res_max_abs_over_systems"] = max(r["dca_per_res_max_abs"] for r in ok)
        res["dmin_per_res_max_abs_over_systems"] = max(r["dmin_per_res_max_abs"] for r in ok)
        res["n_within_tolerance"] = sum(
            r["dca_per_res_max_abs"] <= A.tol and r["dmin_per_res_max_abs"] <= A.tol for r in ok)
        tot_m = sum(r["n_res_matched"] for r in ok); tot_x = sum(r["n_res_count_mismatch"] for r in ok)
        res["residues_matched"] = tot_m
        res["residues_count_mismatch"] = tot_x
        res["residue_match_rate"] = tot_m / (tot_m + tot_x) if tot_m + tot_x else 0.0
        d = [r["dca_direct_max_abs"] for r in ok if "dca_direct_max_abs" in r]
        if d:
            res["n_direct_within_tolerance"] = sum(x <= A.tol for x in d)
            res["dca_direct_max_abs_over_systems"] = max(d)
    res["systems"] = rows
    json.dump(res, open(A.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "systems"}, indent=1))

if __name__ == "__main__":
    main()
