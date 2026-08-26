import sys, os, glob, json, subprocess
import numpy as np
sys.path.insert(0, "/workspace")
from run_smina_labels import prep_receptor, prep_ligand, contacts, parse_protein
from reextract_labels import strip_h
from atom_match import map_pose_to_ref
from rdkit import Chem
import gemmi
AA_ = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
def clean_receptor(cif, out_pdb):
    # write standard-AA protein atoms only, dropping metals/HETATM that break obabel pdbqt
    m = gemmi.read_structure(cif)[0]
    st = gemmi.Structure(); md = gemmi.Model("1")
    for ch in m:
        nc = gemmi.Chain(ch.name); keep = False
        for res in ch:
            if res.name not in AA_: continue
            nc.add_residue(res); keep = True
        if keep: md.add_chain(nc)
    st.add_model(md)
    st.write_pdb(out_pdb)
    return out_pdb
def parse_cif_protein(path):
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

STR = "/workspace/datasets/experimental_expansion/structures"
OUTD = "/workspace/datasets/experimental_expansion/smina_out"

def systems():
    # stream system_id and crystal ligand centroid (pocket box center) from shards
    for f in sorted(glob.glob("/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz")):
        d = np.load(f, allow_pickle=True)
        sid, lo, lig = d["system_id"], d["lig_offsets"], d["lig_xyz"]
        for i in range(len(sid)):
            yield str(sid[i]), lig[lo[i]:lo[i+1]].mean(0)
        del d


def main():
    os.makedirs(OUTD, exist_ok=True)
    ccd_smiles = json.load(open("/workspace/datasets/experimental_expansion/ccd_smiles.json"))
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    mine = [(s, c) for i, (s, c) in enumerate(systems()) if i % nshard == shard]
    cache = f"{OUTD}/work_{shard}"; os.makedirs(cache, exist_ok=True)
    posedir = f"{OUTD}/poses"; os.makedirs(posedir, exist_ok=True)
    SID = []; off = [0]; C_res = []; C_atom = []; C_dca = []; C_dmin = []
    done = miss = fail = 0
    for s, center in mine:
        if os.path.exists(f"{OUTD}/poses/{s.replace('.','_')}.sdf"):
            done += 1; continue
        pdb, ccd = s.split("_")[0], s.split("_")[1]
        smi = ccd_smiles.get(ccd); cif = f"{STR}/{pdb}.cif"
        if not smi or not os.path.exists(cif):
            miss += 1; continue
        try:
            recq = prep_receptor(clean_receptor(cif, f"{cache}/rec.pdb"), f"{cache}/rec.pdbqt")
            ligf, ref = prep_ligand(smi, f"{cache}/lig.sdf")
            safe_s = s.replace(".","_"); outp = f"{cache}/pose.sdf"
            subprocess.run(["smina", "--receptor", recq, "--ligand", ligf,
                "--center_x", f"{center[0]}", "--center_y", f"{center[1]}", "--center_z", f"{center[2]}",
                "--size_x", "22", "--size_y", "22", "--size_z", "22",
                "--exhaustiveness", "4", "--num_modes", "1", "--cpu", "1",
                "--seed", "0", "--out", outp, "--quiet"], check=True, timeout=120)
            import shutil
            shutil.copy(outp, f"{posedir}/{safe_s}.sdf")
            pose = strip_h(next(iter(Chem.SDMolSupplier(outp, removeHs=False, sanitize=False))))
            perm, _ = map_pose_to_ref(ref, pose)
            conf = pose.GetConformer()
            ligc = np.zeros((ref.GetNumAtoms(), 3), np.float32)
            for pi, ri in enumerate(perm):
                p = conf.GetAtomPosition(pi); ligc[ri] = (p.x, p.y, p.z)
            ca, heavy = parse_cif_protein(cif)
            rr, ai, dca, dmn = contacts(ligc, ca, heavy)
            if not rr:
                fail += 1; continue
            C_res += rr; C_atom += ai; C_dca += dca; C_dmin += dmn
            SID.append(s); off.append(len(C_res)); done += 1
        except Exception:
            fail += 1; continue
    np.savez_compressed(f"{OUTD}/smina_exp_{shard}.npz",
        system_id=np.array(SID), contact_offsets=np.array(off, np.int64),
        res_row=np.array(C_res, np.int32), atom_idx=np.array(C_atom, np.int16),
        d_ca=np.array(C_dca, np.float16), d_min=np.array(C_dmin, np.float16),
        cutoff=np.float32(15.0), source=np.array("smina"))
    print(f"shard {shard}: done={done} miss={miss} fail={fail}")

if __name__ == "__main__":
    main()
