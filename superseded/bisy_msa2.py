"""SUPERSEDED. This script re-enumerates residues through ca_all(), which
emits a NaN row for standard-amino-acid residues lacking a C-alpha while the
label generator skips them, shifting every subsequent res_row by one. It is
retained for provenance only. Use datasets/bisy_rmsd_v2.py, which takes its
correspondence from the label generator rather than re-deriving it."""

import sys, os, glob, numpy as np, gemmi
from spyrmsd import rmsd as srmsd
sys.path.insert(0,"/workspace/datasets")
from cryst_lig_helper import crystal_ligand
D="/workspace/datasets/experimental_expansion"
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
def lig(m,ccd):
    for ch in m:
        for res in ch:
            if res.name==ccd:
                xs=[];el=[]
                for a in res:
                    if a.element.name not in ('H','D'): xs.append([a.pos.x,a.pos.y,a.pos.z]); el.append(a.element.name.upper())
                return np.array(xs,float),el
    return None,None
def adj(x):
    d=np.sqrt(((x[:,None]-x[None])**2).sum(-1)); return ((d<1.9)&(d>0.1)).astype(int)
con={}
for f in sorted(glob.glob(f"{D}/crystal_labels/crystal_labels_*.npz")):
    d=np.load(f,allow_pickle=True);o=d['contact_offsets'];sids=d['system_id']
    for i in range(len(sids)): con[str(sids[i])]=np.unique(d['res_row'][int(o[i]):int(o[i+1])])
    del d
ids=[l.strip() for l in open(f"{D}/msa_sample_ids2.txt") if l.strip()]
SID=[];RM=[];POC=[]
done=miss=fail=0
for s in ids:
    pp=f"{D}/boltz_out_msa2/boltz_results_boltz_in_msa2/predictions/{s}/{s}_model_0.pdb"
    if not os.path.exists(pp): pp=f"{D}/boltz_out_msa2/predictions/{s}/{s}_model_0.pdb"
    cp=f"{D}/structures/{s.split('_')[0]}.cif"; clr=s.split('_')[1]
    if not (os.path.exists(pp) and os.path.exists(cp)): miss+=1; continue
    try:
        mc=gemmi.read_structure(cp)[0]; mp=gemmi.read_structure(pp)[0]
        cca=ca_all(mc); pca=ca_all(mp)
        clig,cel=crystal_ligand(s,cp,clr); plig,pel=lig(mp,"LIG")
        if clig is None or plig is None or len(clig)!=len(plig) or len(clig)==0: fail+=1; continue
        c=con[s]; c=c[(c<len(cca))&(c<len(pca))]
        P=pca[c];Q=cca[c];g=~(np.isnan(P).any(1)|np.isnan(Q).any(1));P,Q=P[g],Q[g]
        if len(P)<4: fail+=1; continue
        Pc,Qc=P.mean(0),Q.mean(0);H=(P-Pc).T@(Q-Qc);U,S,Vt=np.linalg.svd(H);dt=np.sign(np.linalg.det(Vt.T@U.T))
        R=Vt.T@np.diag([1,1,dt])@U.T
        poc=float(np.sqrt((((R@(P-Pc).T).T+Qc-Q)**2).sum(1).mean()))
        paln=(R@(plig-Pc).T).T+Qc
        v=srmsd.symmrmsd(clig,paln,np.array([Z.get(e,6) for e in cel]),np.array([Z.get(e,6) for e in pel]),adj(clig),adj(plig))
        SID.append(s);RM.append(float(v));POC.append(poc);done+=1
    except Exception: fail+=1; continue
np.savez_compressed(f"{D}/bisy/bisy_boltz2_msa2.npz",system_id=np.array(SID),ligand_rmsd=np.array(RM,np.float32),pocket_resid=np.array(POC,np.float32))
r=np.array(RM); p=np.array(POC)
print(f"MSA-Boltz n={done} miss={miss} fail={fail}")
print(f"  ligand RMSD  median={np.median(r):.2f}  <2A={100*(r<2).mean():.0f}%  <5A={100*(r<5).mean():.0f}%")
print(f"  pocket resid median={np.median(p):.2f}  frac>5A={100*(p>5).mean():.0f}%")
