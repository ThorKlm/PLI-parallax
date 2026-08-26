#!/usr/bin/env python
"""Validation for ``bisy_rmsd_v2.py``: does the new number agree with an independent
implementation, and what exactly did the old script do with the systems it lost?

Four checks, each printed as its own block:

  agreement   -- ``spyrmsd.symmrmsd`` on the *same* correspondence problem but given real
                 bond adjacency instead of the 1.9 A cutoff, on a sample of resolved
                 systems.  If v2's candidate ranking is right this should agree exactly.
  invariants  -- rmsd >= rmsd_superposed, rmsd >= centroid_offset, no duplicate ids,
                 mapping bijective on full-scope matches.
  old_failures-- replays ``bisy_rmsd.py``'s inner loop on the systems it produced no
                 number for, and classifies *why*: the point is to show the losses are
                 graph/count failures rather than anything intrinsic to those systems.
  fusion      -- the crystal sequence handed to the cofolders is the concatenation of
                 every protein chain, so a multi-chain entry was folded as one chain.
                 Measures how much of the pocket-superposition residual that accounts for.

Usage:
    python bisy_rmsd_v2_validate.py --teacher chai1 --sample 400
"""

import argparse
import json
import os
import sys
from collections import Counter

import gemmi
import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/datasets")

import bisy_rmsd_v2 as V  # noqa: E402
import bisy_smina_v2 as S  # noqa: E402
from cryst_lig_helper import crystal_ligand  # noqa: E402

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
Z = {'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'F': 9, 'CL': 17, 'BR': 35, 'I': 53,
     'FE': 26, 'ZN': 30, 'MG': 12, 'MN': 25, 'CA': 20, 'NA': 11, 'K': 19, 'CU': 29,
     'NI': 28, 'CO': 27, 'SE': 34}


def bond_adjacency(mol):
    """Adjacency from the perceived bond graph -- what the 1.9 A cutoff was standing in for."""
    n = mol.GetNumAtoms()
    a = np.zeros((n, n), int)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a[i, j] = a[j, i] = 1
    return a


def check_agreement(rows, ccd_smiles, teacher, sample):
    from spyrmsd import rmsd as srmsd

    ok = [r for r in rows if r["status"] == "ok" and r["match_scope"] == "full"]
    ok = ok[:: max(1, len(ok) // sample)][:sample]
    deltas, errors = [], Counter()
    pockets = V.Pockets()
    spec = V.TEACHERS[teacher]
    for r in ok:
        sid = r["system_id"]
        try:
            ref, _ = S.load_reference(sid, ccd_smiles)
            pose, _src, model = V.load_pose(spec["path"].format(sid=sid), spec["ligand"],
                                            sid.split("_")[1], ccd_smiles)
            to_crystal, _resid, _n, _mode = V.pocket_superposition(pockets, sid, model)
            ref_xyz = ref.GetConformer().GetPositions()
            pose_xyz = to_crystal(pose.GetConformer().GetPositions())
            zr = np.array([a.GetAtomicNum() for a in ref.GetAtoms()])
            zp = np.array([a.GetAtomicNum() for a in pose.GetAtoms()])
            val = srmsd.symmrmsd(ref_xyz, pose_xyz, zr, zp,
                                 bond_adjacency(ref), bond_adjacency(pose))
            deltas.append(abs(float(val) - float(r["rmsd"])))
        except Exception as exc:
            errors[type(exc).__name__] += 1
    print(f"[agreement] spyrmsd with real bond graphs, {len(deltas)} systems")
    if deltas:
        d = np.array(deltas)
        print(f"  |delta| mean {d.mean():.4f}  p95 {np.percentile(d, 95):.4f}  "
              f"max {d.max():.4f} A")
        print(f"  within 0.01 A: {(d < 0.01).sum()}/{len(d)}")
    if errors:
        print(f"  spyrmsd could not run on: {dict(errors)}")


def check_invariants(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    ids = [r["system_id"] for r in rows]
    bad_sup = sum(1 for r in ok if float(r["rmsd"]) + 1e-6 < float(r["rmsd_superposed"]))
    bad_cen = sum(1 for r in ok if float(r["rmsd"]) + 1e-6 < float(r["centroid_offset"]))
    print("[invariants]")
    print(f"  rows {len(rows)}  resolved {len(ok)}  duplicate ids {len(ids) - len(set(ids))}")
    print(f"  rmsd < rmsd_superposed : {bad_sup}")
    print(f"  rmsd < centroid_offset : {bad_cen}")
    scopes = Counter(r["match_scope"] for r in ok)
    print(f"  match scope: {dict(scopes)}")
    print(f"  n_matched != n_ref_heavy: "
          f"{sum(1 for r in ok if r['n_matched'] != r['n_ref_heavy'])}")


def replay_old(rows, teacher, sample):
    """Classify why ``bisy_rmsd.py`` produced no number for the systems it lost."""
    from spyrmsd import rmsd as srmsd

    old = V.old_rmsd(teacher) or {}
    pockets = V.Pockets()
    unreached = V.old_unattempted(teacher, pockets)
    lost = [r for r in rows
            if r["system_id"] not in old and r["system_id"] not in unreached]
    lost = lost[:: max(1, len(lost) // sample)][:sample]
    spec = V.TEACHERS[teacher]

    def adj19(xyz):
        d = np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1))
        return ((d < 1.9) & (d > 0.1)).astype(int)

    def ca_all(model):
        out = []
        for chain in model:
            for res in chain:
                if res.name in AA:
                    ca = None
                    for atom in res:
                        if atom.name == 'CA':
                            ca = (atom.pos.x, atom.pos.y, atom.pos.z)
                    out.append(ca if ca else (np.nan, np.nan, np.nan))
        return np.array(out, float)

    def lig(model, resname):
        for chain in model:
            for res in chain:
                if res.name == resname:
                    xs, el = [], []
                    for atom in res:
                        if atom.element.name not in ('H', 'D'):
                            xs.append([atom.pos.x, atom.pos.y, atom.pos.z])
                            el.append(atom.element.name.upper())
                    return np.array(xs, float), el
        return None, None

    why = Counter()
    v2_ok = Counter()
    for r in lost:
        sid = r["system_id"]
        pdb = sid.split("_")[0]
        cif = f"{V.STRUCT_DIR}/{pdb}.cif"
        path = spec["path"].format(sid=sid)
        tag = None
        try:
            if not (os.path.exists(path) and os.path.exists(cif)):
                tag = "missing_file"
            else:
                mc = gemmi.read_structure(cif)[0]
                mp = gemmi.read_structure(path)[0]
                cca, pca = ca_all(mc), ca_all(mp)
                clig, cel = crystal_ligand(sid, cif, sid.split("_")[1])
                plig, pel = lig(mp, spec["ligand"])
                if clig is None or plig is None or len(clig) == 0:
                    tag = "no_ligand"
                elif len(clig) != len(plig):
                    tag = "atom_count_mismatch"
                else:
                    info = pockets.get(sid)
                    con = info["contacts"]
                    con = con[(con < len(cca)) & (con < len(pca))]
                    P, Q = pca[con], cca[con]
                    good = ~(np.isnan(P).any(1) | np.isnan(Q).any(1))
                    P, Q = P[good], Q[good]
                    if len(P) < 4:
                        tag = "too_few_pocket_residues"
                    else:
                        R, Pc, Qc = V.kabsch(P, Q)
                        aligned = (R @ (plig - Pc).T).T + Qc
                        zc = np.array([Z.get(e, 6) for e in cel])
                        zp = np.array([Z.get(e, 6) for e in pel])
                        srmsd.symmrmsd(clig, aligned, zc, zp, adj19(clig), adj19(plig))
                        tag = "would_have_worked"
        except Exception as exc:
            tag = f"exception:{type(exc).__name__}"
        why[tag] += 1
        if r["status"] == "ok":
            v2_ok[tag] += 1

    print(f"[old_failures] replayed bisy_rmsd.py on {len(lost)} systems it lost "
          f"(sampled from {sum(1 for r in rows if r['system_id'] not in old and r['system_id'] not in unreached)})")
    for tag, n in why.most_common():
        print(f"  {tag:<26s} {n:5d}   v2 resolves {v2_ok[tag]:5d}")


def check_fusion(rows, sample):
    """How much of the pocket residual is the single-chain fusion of a multi-chain entry."""
    ok = [r for r in rows if r["status"] == "ok"]
    ok = ok[:: max(1, len(ok) // sample)][:sample]
    single, multi = [], []
    for r in ok:
        cif = f"{V.STRUCT_DIR}/{r['pdb']}.cif"
        if not os.path.exists(cif):
            continue
        try:
            model = gemmi.read_structure(cif)[0]
        except Exception:
            continue
        n_chains = sum(1 for c in model if any(res.name in AA for res in c))
        (single if n_chains <= 1 else multi).append(
            (float(r["pocket_rmsd"]), float(r["rmsd"])))
    print(f"[fusion] crystal protein chains vs pocket superposition, {len(single) + len(multi)} sampled")
    for label, group in (("single-chain", single), ("multi-chain", multi)):
        if not group:
            continue
        pocket = np.array([g[0] for g in group])
        rmsd = np.array([g[1] for g in group])
        print(f"  {label:<13s} n {len(group):5d}   pocket residual median "
              f"{np.median(pocket):6.2f} A   ligand RMSD median {np.median(rmsd):6.2f} A   "
              f"< 2 A {100.0 * (rmsd < 2).mean():5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", default="chai1", choices=list(V.TEACHERS))
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--checks", default="agreement,invariants,old_failures,fusion")
    args = ap.parse_args()

    rows = V.read_rows(args.teacher)
    if not rows:
        raise SystemExit(f"no v2 CSVs for {args.teacher}")
    with open(V.CCD_SMILES) as fh:
        ccd_smiles = json.load(fh)

    print(f"### {args.teacher}  ({len(rows)} rows)")
    checks = set(args.checks.split(","))
    if "agreement" in checks:
        check_agreement(rows, ccd_smiles, args.teacher, args.sample)
    if "invariants" in checks:
        check_invariants(rows)
    if "old_failures" in checks:
        replay_old(rows, args.teacher, args.sample)
    if "fusion" in checks:
        check_fusion(rows, args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
