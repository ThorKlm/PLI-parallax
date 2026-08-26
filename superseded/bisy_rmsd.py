"""SUPERSEDED. This script re-enumerates residues through ca_all(), which
emits a NaN row for standard-amino-acid residues lacking a C-alpha while the
label generator skips them, shifting every subsequent res_row by one. It is
retained for provenance only. Use datasets/bisy_rmsd_v2.py, which takes its
correspondence from the label generator rather than re-deriving it."""

import sys, os, glob, csv
import numpy as np, gemmi
from spyrmsd import rmsd as srmsd
import sys as _sys; _sys.path.insert(0,"/workspace/datasets")
from cryst_lig_helper import crystal_ligand
AA=set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
Z={'C':6,'N':7,'O':8,'S':16,'P':15,'F':9,'CL':17,'BR':35,'I':53,'FE':26,'ZN':30,'MG':12,'MN':25,'CA':20,'NA':11,'K':19,'CU':29,'NI':28,'CO':27,'SE':34}

def ca_all(m):
    out=[]
    for ch in m:
        for res in ch:
            if res.name in AA:
                cx=None
                for a in res:
                    if a.name=='CA': cx=(a.pos.x,a.pos.y,a.pos.z)
                out.append(cx if cx else (np.nan,np.nan,np.nan))
    return np.array(out,float)
def lig(m, ccd):
    for ch in m:
        for res in ch:
            if res.name==ccd:
                xs=[]; el=[]
                for a in res:
                    if a.element.name not in ('H','D'):
                        xs.append([a.pos.x,a.pos.y,a.pos.z]); el.append(a.element.name.upper())
                return np.array(xs,float), el
    return None,None
def adj(xyz):
    d=np.sqrt(((xyz[:,None]-xyz[None])**2).sum(-1))
    return ((d<1.9)&(d>0.1)).astype(int)

def main():
    teacher=sys.argv[1]; shard=int(sys.argv[2]); nshard=int(sys.argv[3])
    cryst_con={}
    for f in sorted(glob.glob('/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz')):
        d=np.load(f,allow_pickle=True); o=d['contact_offsets']; sids=d['system_id']
        for i in range(len(sids)):
            cryst_con[str(sids[i])]=np.unique(d['res_row'][int(o[i]):int(o[i+1])])
        del d
    sids=sorted(cryst_con); mine=[s for j,s in enumerate(sids) if j%nshard==shard]
    def pred(s):
        safe=s.replace('.','_')
        if teacher=="boltz":
            return f"/workspace/datasets/experimental_expansion/boltz_out/boltz_results_boltz_in/predictions/{safe}/{safe}_model_0.pdb","LIG"
        return f"/workspace/docking/output/chai_out_exp/{safe}/pred.model_idx_0.cif","LIG2"
    SID=[]; RMSD=[]
    done=miss=fail=0
    for s in mine:
        pp,plr=pred(s); cp=f"/workspace/datasets/experimental_expansion/structures/{s.split('_')[0]}.cif"; clr=s.split('_')[1]
        if not os.path.exists(pp) or not os.path.exists(cp): miss+=1; continue
        try:
            mc=gemmi.read_structure(cp)[0]; mp=gemmi.read_structure(pp)[0]
            cca=ca_all(mc); pca=ca_all(mp)
            clig,cel=crystal_ligand(s, cp, clr); plig,pel=lig(mp,plr)
            if clig is None or plig is None or len(clig)!=len(plig) or len(clig)==0: fail+=1; continue
            con=cryst_con[s]; con=con[(con<len(cca))&(con<len(pca))]
            P=pca[con]; Q=cca[con]; g=~(np.isnan(P).any(1)|np.isnan(Q).any(1)); P,Q=P[g],Q[g]
            if len(P)<4: fail+=1; continue
            Pc,Qc=P.mean(0),Q.mean(0); H=(P-Pc).T@(Q-Qc); U,S,Vt=np.linalg.svd(H); dt=np.sign(np.linalg.det(Vt.T@U.T))
            R=Vt.T@np.diag([1,1,dt])@U.T
            paln=(R@(plig-Pc).T).T+Qc
            zc=np.array([Z.get(e,6) for e in cel]); zp=np.array([Z.get(e,6) for e in pel])
            v=srmsd.symmrmsd(clig, paln, zc, zp, adj(clig), adj(plig))
            SID.append(s); RMSD.append(float(v)); done+=1
        except Exception:
            fail+=1; continue
    src={"boltz":"boltz2","chai":"chai1"}[teacher]
    os.makedirs("/workspace/datasets/experimental_expansion/bisy",exist_ok=True)
    np.savez_compressed(f"/workspace/datasets/experimental_expansion/bisy/bisy_{src}_{shard}.npz",
        system_id=np.array(SID), ligand_rmsd=np.array(RMSD,np.float32))
    print(f"{src} shard {shard}: done={done} miss={miss} fail={fail}")
if __name__=="__main__": main()
