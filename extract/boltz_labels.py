"""Parse Boltz predicted complexes into source=boltz2 residue-atom distance labels.

For each predicted structure: protein residues are read in chain order (matching
proteins_v10 rows), the ligand is rebuilt from its predicted atoms, remapped to
the canonical head ordering via atom_match, and per-residue CA and min-heavy
distances are computed in Boltz's own frame. Confidence (iptm, ligand_iptm,
confidence_score) is carried per pair. Output is the same CSR npz as smina.
Run: python boltz_labels.py /workspace/docking/output/boltz_in_index.tsv \
         /workspace/docking/output/boltz_out /workspace/docking/output/boltz
"""
import csv, glob, json, os, sys
import numpy as np
import gemmi
from rdkit import Chem
from atom_match import map_pose_to_ref, ref_mol_from_smiles

CUTOFF = 15.0  # Angstrom, min-heavy residue-atom distance to record a contact

def contacts(lig, ca, heavy):
    """(res_row, atom_idx, d_ca, d_min) for residues within CUTOFF of the ligand."""
    rr, ai, dca, dmn = [], [], [], []
    for r in range(len(heavy)):
        dm = np.sqrt(((heavy[r][:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
        hit = np.where(dm <= CUTOFF)[0]
        if hit.size == 0: continue
        dc = np.sqrt(((ca[r] - lig[hit]) ** 2).sum(-1))
        rr.extend([r] * hit.size); ai.extend(hit.tolist())
        dca.extend(dc.tolist()); dmn.extend(dm[hit].tolist())
    return rr, ai, dca, dmn

def read_structure(model_path):
    """Return (ca[R,3], heavy[list of R arrays], ligand_atoms[(elem,xyz)])."""
    st = gemmi.read_structure(model_path)
    st.setup_entities()
    m = st[0]; ca, heavy, lig = [], [], []
    for chain in m:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            is_poly = info is not None and info.is_amino_acid()
            if is_poly:
                hv, cap = [], None
                for at in res:
                    if at.element == gemmi.Element("H"): continue
                    p = (at.pos.x, at.pos.y, at.pos.z); hv.append(p)
                    if at.name == "CA": cap = p
                if hv:
                    heavy.append(np.array(hv, np.float32))
                    ca.append(cap if cap else hv[0])
            else:
                for at in res:
                    if at.element == gemmi.Element("H"): continue
                    lig.append((at.element.name, (at.pos.x, at.pos.y, at.pos.z)))
    return np.array(ca, np.float32), heavy, lig

def mol_from_atoms(lig):
    """Fallback only: rebuild connectivity from geometry when the pkl mol is absent."""
    from rdkit.Chem import rdDetermineBonds
    rw = Chem.RWMol(); conf = Chem.Conformer(len(lig))
    for i, (el, xyz) in enumerate(lig):
        rw.AddAtom(Chem.Atom(el)); conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*xyz))
    m = rw.GetMol(); m.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(m)
    return m

def pose_from_pkl(lig, pkl_mol):
    """Heavy-atom mol with real bonds from the Boltz mol and predicted cif coords.
    Requires the pkl heavy-atom order to equal the cif ligand order (verified by
    element sequence). Returns None on mismatch so the caller can fall back.
    """
    ph = Chem.RemoveHs(pkl_mol, sanitize=False)
    sp = [a.GetSymbol().upper() for a in ph.GetAtoms()]
    sl = [e.upper() for e, _ in lig]
    if sp != sl: return None
    conf = Chem.Conformer(ph.GetNumAtoms())
    for i, (_, xyz) in enumerate(lig):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(c) for c in xyz]))
    ph = Chem.Mol(ph); ph.RemoveAllConformers(); ph.AddConformer(conf, assignId=True)
    return ph

def ligand_canonical(lig, ref, pkl_mol):
    """Predicted ligand coords reordered into reference (head) atom ordering."""
    pose = pose_from_pkl(lig, pkl_mol) if pkl_mol is not None else None
    if pose is None: pose = mol_from_atoms(lig)
    perm, _ = map_pose_to_ref(ref, pose)
    conf = Chem.RemoveHs(pose, sanitize=False).GetConformer()
    coords = np.zeros((ref.GetNumAtoms(), 3), np.float32)
    for pose_i, ref_i in enumerate(perm):
        p = conf.GetAtomPosition(pose_i); coords[ref_i] = (p.x, p.y, p.z)
    return coords

def load_ligand_mol(pkl_path, n_ref):
    """Pick the ligand RDKit mol whose heavy-atom count matches the reference."""
    import pickle
    try:
        d = pickle.load(open(pkl_path, "rb"))
    except Exception:
        return None
    cands = d.values() if isinstance(d, dict) else [d]
    for v in cands:
        if isinstance(v, Chem.Mol) and Chem.RemoveHs(v, sanitize=False).GetNumAtoms() == n_ref:
            return v
    return None

def main():
    index, out_dir, save_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(save_dir, exist_ok=True)
    meta = {r["name"]: r for r in csv.DictReader(open(index), delimiter="\t")}
    models = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "**", "predictions", "*", "*_model_0.*"),
                              recursive=True)):
        if not p.endswith((".pdb", ".cif")): continue
        name = os.path.basename(os.path.dirname(p))
        if name not in models or p.endswith(".pdb"): models[name] = p  # prefer pdb

    P_idx, P_acc, P_conf, P_iptm, P_ligi, P_off = [], [], [], [], [], [0]
    C_res, C_atom, C_dca, C_dmin = [], [], [], []
    done, fail = 0, 0
    for name, mpath in models.items():
        if name not in meta: continue
        r = meta[name]; cdir = os.path.dirname(mpath)
        cj = glob.glob(os.path.join(cdir, "confidence_*_model_0.json"))
        root = os.path.dirname(os.path.dirname(cdir))  # boltz_results_*
        pkl = os.path.join(root, "processed", "mols", f"{name}.pkl")
        try:
            ref = ref_mol_from_smiles(r["smiles"])
            ca, heavy, lig = read_structure(mpath)
            if not lig: fail += 1; continue
            pkl_mol = load_ligand_mol(pkl, ref.GetNumAtoms())
            ligc = ligand_canonical(lig, ref, pkl_mol)
            rr, ai, dca, dmn = contacts(ligc, ca, heavy)
        except Exception:
            fail += 1; continue
        conf = json.load(open(cj[0])) if cj else {}
        C_res += rr; C_atom += ai; C_dca += dca; C_dmin += dmn
        P_idx.append(int(r["pair_idx"])); P_acc.append(r["accession"])
        P_conf.append(conf.get("confidence_score", float("nan")))
        P_iptm.append(conf.get("iptm", float("nan")))
        P_ligi.append(conf.get("ligand_iptm", float("nan")))
        P_off.append(len(C_res)); done += 1

    out = os.path.join(save_dir, "boltz_labels.npz")
    np.savez_compressed(out,
        pair_idx=np.array(P_idx, np.int32), accession=np.array(P_acc),
        confidence=np.array(P_conf, np.float32), iptm=np.array(P_iptm, np.float32),
        ligand_iptm=np.array(P_ligi, np.float32),
        contact_offsets=np.array(P_off, np.int64),
        res_row=np.array(C_res, np.int32), atom_idx=np.array(C_atom, np.int16),
        d_ca=np.array(C_dca, np.float16), d_min=np.array(C_dmin, np.float16),
        cutoff=np.float32(CUTOFF), source=np.array("boltz2"))
    print(f"parsed={done} fail={fail} contacts={len(C_res)} -> {out}")

if __name__ == "__main__":
    main()
