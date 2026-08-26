import sys, os, csv, glob
import numpy as np, gemmi
from rdkit import Chem
sys.path.insert(0,"/workspace")
from run_smina_labels import contacts
from atom_match import map_pose_to_ref
AA=set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())

def parse_pred(path, is_cif):
    # predicted complex: protein chain(s) + ligand as the non-AA residue. Returns ca, heavy, lig_xyz, elements
    m=gemmi.read_structure(path)[0]; ca=[]; heavy=[]; lig=[]; el=[]
    for ch in m:
        for res in ch:
            if res.name in AA:
                hv=[]; cx=None
                for at in res:
                    if at.element.name in ("H","D"): continue
                    p=(at.pos.x,at.pos.y,at.pos.z); hv.append(p)
                    if at.name=="CA": cx=p
                if cx is None or not hv: continue
                ca.append(cx); heavy.append(np.array(hv,np.float32))
            else:
                for at in res:
                    if at.element.name in ("H","D"): continue
                    lig.append([at.pos.x,at.pos.y,at.pos.z]); el.append(at.element.name)
    return np.array(ca,np.float32), heavy, np.array(lig,np.float32), el

def canon(lig, el, smi):
    frag=max(smi.split('.'),key=len); ref=Chem.MolFromSmiles(frag)
    if ref is None: raise ValueError("bad smiles")
    rel=[a.GetSymbol() for a in ref.GetAtoms()]
    if el==rel and len(lig)==len(rel): return lig, ref.GetNumAtoms()
    if sorted(el)!=sorted(rel): raise ValueError(f"elem mismatch pose={len(el)} ref={len(rel)}")
    rw=Chem.RWMol()
    for e in el: rw.AddAtom(Chem.Atom(e))
    conf=Chem.Conformer(len(lig))
    for i,(x,y,z) in enumerate(lig): conf.SetAtomPosition(i,(float(x),float(y),float(z)))
    pose=rw.GetMol(); pose.AddConformer(conf)
    perm,_=map_pose_to_ref(ref,pose)
    out=np.zeros((ref.GetNumAtoms(),3),np.float32)
    for pi,ri in enumerate(perm): out[ri]=lig[pi]
    return out, ref.GetNumAtoms()

def main():
    teacher=sys.argv[1]  # boltz or chai
    shard=int(sys.argv[2]); nshard=int(sys.argv[3])
    idx={}
    for r in csv.DictReader(open("/workspace/datasets/experimental_expansion/exp_fold_index.tsv"),delimiter="\t"):
        idx[r["system_id"]]=r["smiles"]
    sids=sorted(idx.keys())
    mine=[s for i,s in enumerate(sids) if i%nshard==shard]
    def locate(sid):
        safe=sid.replace('.','_')
        if teacher=="boltz":
            p=f"/workspace/datasets/experimental_expansion/boltz_out/boltz_results_boltz_in/predictions/{safe}/{safe}_model_0.pdb"
            return p if os.path.exists(p) else None
        else:
            p=f"/workspace/docking/output/chai_out_exp/{safe}/pred.model_idx_0.cif"
            return p if os.path.exists(p) else None
    SID=[]; off=[0]; C_res=[]; C_atom=[]; C_dca=[]; C_dmin=[]; NLIG=[]
    done=miss=fail=0
    for sid in mine:
        path=locate(sid)
        if path is None: miss+=1; continue
        smi=idx[sid]
        try:
            ca,heavy,lig,el=parse_pred(path, teacher!="boltz")
            ligc,n_lig=canon(lig,el,smi)
            rr,ai,dca,dmn=contacts(ligc,ca,heavy)
        except Exception as e:
            fail+=1; continue
        C_res+=list(rr); C_atom+=list(ai); C_dca+=list(dca); C_dmin+=list(dmn)
        SID.append(sid); NLIG.append(n_lig); off.append(len(C_res)); done+=1
    src="boltz2" if teacher=="boltz" else "chai1"
    os.makedirs("/workspace/datasets/experimental_expansion/teacher_labels",exist_ok=True)
    np.savez_compressed(f"/workspace/datasets/experimental_expansion/teacher_labels/exp_{src}_{shard}.npz",
        system_id=np.array(SID), n_lig_atoms=np.array(NLIG,np.int32),
        contact_offsets=np.array(off,np.int64), res_row=np.array(C_res,np.int32),
        atom_idx=np.array(C_atom,np.int16), d_ca=np.array(C_dca,np.float16),
        d_min=np.array(C_dmin,np.float16), cutoff=np.float32(15.0), source=np.array(src))
    print(f"{src} shard {shard}: done={done} miss={miss} fail={fail}")

if __name__=="__main__": main()
