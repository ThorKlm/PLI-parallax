import sys, os, glob, json
import numpy as np
sys.path.insert(0, "/workspace")
from run_smina_labels import contacts
from reextract_labels import strip_h
from atom_match import map_pose_to_ref, ref_mol_from_smiles
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import gemmi
AA_ = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
def parse_cif_protein(path):
    m = gemmi.read_structure(path)[0]
    ca = []; heavy = []
    for ch in m:
        for res in ch:
            if res.name not in AA_: continue
            hv = []; cx = None
            for a in res:
                if a.element.name in ("H","D"): continue
                q = (a.pos.x, a.pos.y, a.pos.z); hv.append(q)
                if a.name == "CA": cx = q
            if cx is None or not hv: continue
            ca.append(cx); heavy.append(np.array(hv, np.float32))
    return np.array(ca, np.float32), heavy
STR = "/workspace/datasets/experimental_expansion/structures"
OUTD = "/workspace/datasets/experimental_expansion/smina_labels"
os.makedirs(OUTD, exist_ok=True)
ccd_smiles = json.load(open("/workspace/datasets/experimental_expansion/ccd_smiles.json"))
shard, nshard = int(sys.argv[1]), int(sys.argv[2])
poses = sorted(glob.glob("/workspace/datasets/experimental_expansion/smina_out/poses/*.sdf"))
mine = [p for i, p in enumerate(poses) if i % nshard == shard]
protc, refc = {}, {}
SID=[]; off=[0]; C_res=[]; C_atom=[]; C_dca=[]; C_dmin=[]
done=miss=fail=0
for p in mine:
    sid = os.path.basename(p)[:-4]
    parts = sid.split("_")
    pdb, ccd = parts[0], parts[1]
    smi = ccd_smiles.get(ccd); cif = f"{STR}/{pdb}.cif"
    if not smi or not os.path.exists(cif):
        miss += 1; continue
    try:
        if ccd not in refc:
            refc[ccd] = ref_mol_from_smiles(smi)
        ref = refc[ccd]
        pose = strip_h(next(iter(Chem.SDMolSupplier(p, removeHs=False, sanitize=False))))
        perm, _ = map_pose_to_ref(ref, pose)
        conf = pose.GetConformer()
        lig = np.zeros((ref.GetNumAtoms(), 3), np.float32)
        for pi, ri in enumerate(perm):
            q = conf.GetAtomPosition(pi); lig[ri] = (q.x, q.y, q.z)
        if pdb not in protc:
            if len(protc) > 300: protc.clear()
            protc[pdb] = parse_cif_protein(cif)
        ca, heavy = protc[pdb]
        rr, ai, dca, dmn = contacts(lig, ca, heavy)
        if not rr: fail += 1; continue
        C_res += rr; C_atom += ai; C_dca += dca; C_dmin += dmn
        SID.append(sid); off.append(len(C_res)); done += 1
    except Exception:
        fail += 1; continue
np.savez_compressed(f"{OUTD}/smina_lab_{shard}.npz",
    system_id=np.array(SID), contact_offsets=np.array(off, np.int64),
    res_row=np.array(C_res, np.int32), atom_idx=np.array(C_atom, np.int16),
    d_ca=np.array(C_dca, np.float16), d_min=np.array(C_dmin, np.float16),
    cutoff=np.float32(15.0), source=np.array("smina"))
print(f"shard {shard}: done={done} miss={miss} fail={fail} contacts={len(C_res)}")
