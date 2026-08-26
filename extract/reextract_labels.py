import os, sys, glob, csv, importlib.util
import numpy as np
from rdkit import Chem

R = importlib.util.spec_from_file_location("rs", "/workspace/run_smina_labels.py")
rs = importlib.util.module_from_spec(R); R.loader.exec_module(rs)
A = importlib.util.spec_from_file_location("am", "/workspace/atom_match.py")
am = importlib.util.module_from_spec(A); A.loader.exec_module(am)

CUTOFF = 15.0

def strip_h(m):
    em = Chem.RWMol(m)
    for i in sorted([a.GetIdx() for a in em.GetAtoms() if a.GetAtomicNum()==1], reverse=True):
        em.RemoveAtom(i)
    return em.GetMol()

def coords_from_pose(pose_raw, ref, perm):
    ph = strip_h(pose_raw); conf = ph.GetConformer()
    coords = np.zeros((ref.GetNumAtoms(), 3), np.float32)
    for pose_i, ref_i in enumerate(perm):
        p = conf.GetAtomPosition(pose_i); coords[ref_i] = (p.x, p.y, p.z)
    return coords

def load_smiles(manifest):
    d = {}
    for r in csv.DictReader(open(manifest), delimiter="\t"):
        d[int(r["pair_idx"])] = (r["accession"], r["smiles"])
    return d

def reextract(poses_dir, manifest, root, out_npz):
    smap = load_smiles(manifest)
    bypair = {}
    for f in glob.glob(os.path.join(poses_dir, "*.sdf")):
        pair = int(os.path.basename(f).split("_")[0]); bypair.setdefault(pair, []).append(f)
    P_idx,P_acc,P_rank,P_aff,P_amb = [],[],[],[],[]
    off,C_res,C_atom,C_dca,C_dmin = [0],[],[],[],[]
    prot,refc = {},{}
    ok=fail=0
    for pair,files in bypair.items():
        if pair not in smap: fail+=1; continue
        acc,smi = smap[pair]
        try: ref = refc.get(smi) or rs.ref_mol_from_smiles(smi); refc[smi]=ref
        except Exception: fail+=1; continue
        best=None
        for f in files:
            rank=int(os.path.basename(f).split("_")[1].split(".")[0])
            try:
                sup=list(Chem.SDMolSupplier(f, removeHs=False, sanitize=False))
            except Exception:
                continue
            for p in sup:
                if p is None: continue
                try: aff=float(p.GetProp("minimizedAffinity"))
                except Exception: aff=0.0
                if best is None or aff<best[0]: best=(aff,p,rank)
        if best is None: fail+=1; continue
        try:
            perm,amb = am.map_pose_to_ref(ref, strip_h(best[1]))
            lig = coords_from_pose(best[1], ref, perm)
        except Exception: fail+=1; continue
        pdb=os.path.join(root,"pdbs",f"AF-{acc}-F1-model_v6.pdb")
        if acc not in prot:
            try: prot[acc]=rs.parse_protein(pdb)
            except Exception: prot[acc]=None
        if prot[acc] is None: fail+=1; continue
        ca,heavy = prot[acc]
        rr,ai,dca,dmn = rs.contacts(lig, ca, heavy)
        C_res+=rr; C_atom+=ai; C_dca+=dca; C_dmin+=dmn
        P_idx.append(pair); P_acc.append(acc); P_rank.append(best[2])
        P_aff.append(best[0]); P_amb.append(bool(amb)); off.append(len(C_res)); ok+=1
    np.savez_compressed(out_npz,
        pair_idx=np.array(P_idx,np.int32), accession=np.array(P_acc),
        pocket_rank=np.array(P_rank,np.int16), affinity=np.array(P_aff,np.float32),
        ambiguous=np.array(P_amb,bool), contact_offsets=np.array(off,np.int64),
        res_row=np.array(C_res,np.int32), atom_idx=np.array(C_atom,np.int16),
        d_ca=np.array(C_dca,np.float16), d_min=np.array(C_dmin,np.float16),
        cutoff=np.float32(CUTOFF), source=np.array("smina"))
    print(f"{out_npz}: recovered {ok}, failed {fail}, unique {len(set(P_idx))}")

if __name__=="__main__":
    root="/workspace/docking/input"
    streams=[
        ("/workspace/docking/output/smina/poses","/workspace/docking/output/manifest_shuf.tsv","/workspace/docking/output/smina/smina_labels_re.npz"),
        ("/workspace/smina_from_B/smina_B/poses","/workspace/docking/output/manifest_shuf.tsv","/workspace/smina_from_B/smina_B/smina_labels_re.npz"),
        ("/workspace/smina_from_C/smina_C/poses","/workspace/docking/output/manifest_shuf.tsv","/workspace/smina_from_C/smina_C/smina_labels_re.npz"),
        ("/workspace/docking/output/smina_gap_rem/poses","/workspace/docking/output/manifest_gap_rem.tsv","/workspace/docking/output/smina_gap_rem/smina_labels_re.npz"),
    ]+[(f"/workspace/docking/output/smina_fg_{i}/poses",f"/workspace/docking/output/manifest_fg_{i}.tsv",f"/workspace/docking/output/smina_fg_{i}/smina_labels_re.npz") for i in range(8)]+[(f"/workspace/docking/output/smina_rem_{i}/poses",f"/workspace/docking/output/manifest_rem_{i}.tsv",f"/workspace/docking/output/smina_rem_{i}/smina_labels_re.npz") for i in range(8)]
    only=sys.argv[1] if len(sys.argv)>1 else None
    for pd,mf,outp in streams:
        if only and only not in pd: continue
        if not os.path.isdir(pd): print("skip",pd); continue
        reextract(pd,mf,root,outp)
