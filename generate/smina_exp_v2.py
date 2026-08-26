#!/usr/bin/env python
"""Re-run of the smina crystal arm with a holo receptor and deeper sampling.

Adapted from ``datasets/smina_exp.py``, which generated the deposited arm.  The
docking box, the ligand preparation and the emitted label schema are byte-for-byte
the same code path, so the two arms differ only in the intended variables:

  1. the receptor **retains metals and cofactors** instead of being reduced to
     standard amino acids (``build_receptor`` below vs ``clean_receptor`` there);
  2. ``--exhaustiveness 8`` instead of 4;
  3. a ``__main__`` guard and argparse (``--shard --nshards --systems --out``).

Three further deviations are forced rather than chosen, and are called out because
they are not free:

  4. the receptor PDBQT is written to a **per-system** path.  ``run_smina_labels.prep_receptor``
     returns early when its output already exists, and both ``smina_exp.py`` and
     ``smina_targeted.py`` hand it one fixed ``<cache>/rec.pdbqt`` per shard.  The
     freshly written ``rec.pdb`` is therefore discarded for every system after the
     first, and the whole shard docks into the first system's protein.  Reproducing
     that would make this run measure nothing.
  5. the smina timeout is 600 s rather than 120 s.  A 120 s cap set for
     exhaustiveness 4 against a stripped receptor would truncate exhaustiveness 8
     against a larger one, i.e. it would silently confound the variable under test.
  6. per-system failure reasons are appended to a sidecar CSV.  The NPZ schema is
     untouched; the original's bare ``except Exception: fail += 1`` simply cannot
     report why a system dropped out.

Nothing here writes to ``experimental_expansion/smina_out``; output goes to
``--out`` (default ``smina_out_v2``).

Usage:
    python smina_exp_v2.py --systems /workspace/reports/smina_v2_systems.txt \
                           --shard 0 --nshards 32
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/workspace")
from run_smina_labels import prep_receptor, prep_ligand, contacts, parse_protein  # noqa: E402,F401
from reextract_labels import strip_h  # noqa: E402
from atom_match import map_pose_to_ref  # noqa: E402
from rdkit import Chem  # noqa: E402
import gemmi  # noqa: E402

AA_ = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())

STR = "/workspace/datasets/experimental_expansion/structures"
DEFAULT_OUT = "/workspace/datasets/experimental_expansion/smina_out_v2"
CCD_SMILES = "/workspace/datasets/experimental_expansion/ccd_smiles.json"
CRYSTAL_LABELS = "/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz"

EXHAUSTIVENESS = 8
BOX = "22"          # identical to smina_exp.py: 22 A cube on the crystal ligand centroid
TIMEOUT = 600


def build_receptor(cif, out_pdb, ligand_ccd):
    """The deposited receptor minus water and minus the ligand being docked.

    ``smina_exp.py``'s ``clean_receptor`` keeps *only* residues whose name is one of the
    20 standard amino acids.  That silently deletes catalytic Zn/Mg/Fe, every heme and
    Fe-S cluster, NAD/FAD/PLP cofactors, glycans and any nucleic acid chain -- so a
    ligand whose real pocket is defined by a metal is docked into a hole where the metal
    used to be.  Here everything is retained except:

      * water, which docking convention removes and which would otherwise fill the site;
      * every copy of the ligand's own CCD, which must go or the pocket is occupied by
        the answer.

    Contact labels are still computed from ``parse_cif_protein`` (standard AAs only), so
    the label definition is unchanged and the two arms stay directly comparable -- the
    retention affects what the pose is docked *against*, not what is recorded.
    """
    model = gemmi.read_structure(cif)[0]
    st = gemmi.Structure()
    md = gemmi.Model("1")
    kept_het = 0
    for ch in model:
        nc = gemmi.Chain(ch.name)
        keep = False
        for res in ch:
            if res.is_water():
                continue
            if res.name == ligand_ccd:
                continue
            nc.add_residue(res)
            keep = True
            if res.name not in AA_:
                kept_het += 1
        if keep:
            md.add_chain(nc)
    st.add_model(md)
    st.write_pdb(out_pdb)
    return out_pdb, kept_het


def parse_cif_protein(path):
    """Unchanged from smina_exp.py -- the contact-label definition must not move."""
    m = gemmi.read_structure(path)[0]
    ca = []; heavy = []
    for ch in m:
        for res in ch:
            if res.name not in AA_: continue
            hv = []; cx = None
            for a in res:
                if a.element.name in ("H", "D"): continue
                p = (a.pos.x, a.pos.y, a.pos.z); hv.append(p)
                if a.name == "CA": cx = p
            if cx is None or not hv: continue
            ca.append(cx); heavy.append(np.array(hv, np.float32))
    return np.array(ca, np.float32), heavy


def systems():
    """Stream system_id and crystal ligand centroid (pocket box center) from shards.

    Identical to smina_exp.py, so the box centre is the same number for both arms.
    """
    for f in sorted(glob.glob(CRYSTAL_LABELS)):
        d = np.load(f, allow_pickle=True)
        sid, lo, lig = d["system_id"], d["lig_offsets"], d["lig_xyz"]
        for i in range(len(sid)):
            yield str(sid[i]), lig[lo[i]:lo[i+1]].mean(0)
        del d


def wanted_ids(spec):
    if not spec:
        return None
    if os.path.exists(spec):
        return {l.strip() for l in open(spec) if l.strip()}
    return {s.strip() for s in spec.split(",") if s.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--systems", default="",
                    help="comma-separated system_ids, or a file with one per line")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    outd = args.out
    os.makedirs(outd, exist_ok=True)
    ccd_smiles = json.load(open(CCD_SMILES))

    wanted = wanted_ids(args.systems)
    pool = [(s, c) for s, c in systems() if wanted is None or s in wanted]
    mine = [(s, c) for i, (s, c) in enumerate(pool) if i % args.nshards == args.shard]

    cache = f"{outd}/work_{args.shard}"; os.makedirs(cache, exist_ok=True)
    posedir = f"{outd}/poses"; os.makedirs(posedir, exist_ok=True)

    SID = []; off = [0]; C_res = []; C_atom = []; C_dca = []; C_dmin = []
    failures = []
    done = miss = fail = 0

    for s, center in mine:
        safe_s = s.replace(".", "_")
        if os.path.exists(f"{posedir}/{safe_s}.sdf"):
            done += 1; continue
        pdb, ccd = s.split("_")[0], s.split("_")[1]
        smi = ccd_smiles.get(ccd); cif = f"{STR}/{pdb}.cif"
        if not smi or not os.path.exists(cif):
            miss += 1
            failures.append((s, "missing_input",
                             "no CCD SMILES" if not smi else f"no structure {pdb}.cif"))
            continue
        stage = "receptor"
        try:
            recp, kept_het = build_receptor(cif, f"{cache}/{safe_s}_rec.pdb", ccd)
            # Per-system PDBQT: prep_receptor short-circuits on an existing path, so a
            # shared filename would pin the whole shard to the first system's protein.
            recq = prep_receptor(recp, f"{cache}/{safe_s}_rec.pdbqt")
            stage = "ligand"
            ligf, ref = prep_ligand(smi, f"{cache}/{safe_s}_lig.sdf")
            stage = "dock"
            outp = f"{cache}/{safe_s}_pose.sdf"
            subprocess.run(["smina", "--receptor", recq, "--ligand", ligf,
                "--center_x", f"{center[0]}", "--center_y", f"{center[1]}", "--center_z", f"{center[2]}",
                "--size_x", BOX, "--size_y", BOX, "--size_z", BOX,
                "--exhaustiveness", str(EXHAUSTIVENESS), "--num_modes", "1", "--cpu", "1",
                "--seed", "0", "--out", outp, "--quiet"], check=True, timeout=TIMEOUT)
            stage = "pose"
            shutil.copy(outp, f"{posedir}/{safe_s}.sdf")
            pose = strip_h(next(iter(Chem.SDMolSupplier(outp, removeHs=False, sanitize=False))))
            stage = "match"
            perm, _ = map_pose_to_ref(ref, pose)
            conf = pose.GetConformer()
            ligc = np.zeros((ref.GetNumAtoms(), 3), np.float32)
            for pi, ri in enumerate(perm):
                p = conf.GetAtomPosition(pi); ligc[ri] = (p.x, p.y, p.z)
            stage = "contacts"
            ca, heavy = parse_cif_protein(cif)
            rr, ai, dca, dmn = contacts(ligc, ca, heavy)
            if not rr:
                fail += 1
                failures.append((s, "contacts", "no residue within 15 A of the pose"))
                continue
            C_res += rr; C_atom += ai; C_dca += dca; C_dmin += dmn
            SID.append(s); off.append(len(C_res)); done += 1
            # keep the shard dir bounded: the pose is already copied out
            for tmp in (recp, recq, ligf, outp):
                try: os.remove(tmp)
                except OSError: pass
        except subprocess.TimeoutExpired:
            fail += 1
            failures.append((s, stage, f"smina exceeded {TIMEOUT}s"))
        except Exception as exc:
            fail += 1
            failures.append((s, stage, f"{type(exc).__name__}: {exc}"))

    np.savez_compressed(f"{outd}/smina_exp_v2_{args.shard}.npz",
        system_id=np.array(SID), contact_offsets=np.array(off, np.int64),
        res_row=np.array(C_res, np.int32), atom_idx=np.array(C_atom, np.int16),
        d_ca=np.array(C_dca, np.float16), d_min=np.array(C_dmin, np.float16),
        cutoff=np.float32(15.0), source=np.array("smina"))

    with open(f"{outd}/failures_{args.shard}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["system_id", "stage", "reason"])
        w.writerows(failures)

    print(f"shard {args.shard}: done={done} miss={miss} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
