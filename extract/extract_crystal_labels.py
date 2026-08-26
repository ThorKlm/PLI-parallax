"""Extract experimental crystal distance labels from fetched mmCIFs.
Self-contained per system: stores protein sequence, pocket residue coords,
ligand atom coords, and residue-ligand-atom distance labels (CSR schema).
Pocket = residues with any heavy atom within CUTOFF of ligand, plus a 50%
radius extension. Sharded for parallel CPU. Keyed by system_id (pdb_ligand_chain).
"""
import sys, os, csv, glob
import numpy as np, gemmi
AA=set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
T={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
CUTOFF=15.0; EXT=1.5  # 50% radius extension for pocket definition
STR="/workspace/datasets/experimental_expansion/structures"
OUT="/workspace/datasets/experimental_expansion/crystal_labels"
os.makedirs(OUT, exist_ok=True)

def heavy_atoms(res):
    return [(a.pos.x,a.pos.y,a.pos.z) for a in res if a.element.name not in ('H','D')]

def extract(pdb, ccd, chain_hint):
    st=gemmi.read_structure(f"{STR}/{pdb}.cif"); m=st[0]
    # collect protein residues (CA-bearing standard AA) and the ligand instance
    prot=[]  # list of (resname, ca_xyz, [heavy_xyz])
    lig=None
    for ch in m:
        for res in ch:
            if res.name in AA:
                ca=None; hv=[]
                for a in res:
                    if a.element.name in ('H','D'): continue
                    p=(a.pos.x,a.pos.y,a.pos.z); hv.append(p)
                    if a.name=='CA': ca=p
                if ca is None or not hv: continue
                prot.append((res.name, ca, np.array(hv,np.float32)))
            elif res.name==ccd and lig is None:
                la=heavy_atoms(res)
                if la: lig=np.array(la,np.float32)
    if lig is None or not prot: return None
    ca=np.array([p[1] for p in prot],np.float32)
    seq=''.join(T[p[0]] for p in prot)
    # contacts: per residue, min dist to any ligand atom
    rr=[]; ai=[]; dca=[]; dmn=[]
    pocket=set()
    for ri,(rn,ca_xyz,hv) in enumerate(prot):
        dm=np.sqrt(((hv[:,None,:]-lig[None,:,:])**2).sum(-1)).min(0)  # per-lig-atom min over residue heavies
        hit=np.where(dm<=CUTOFF)[0]
        if hit.size==0: continue
        pocket.add(ri)
        dc=np.sqrt(((np.array(ca_xyz)-lig[hit])**2).sum(-1))
        rr.extend([ri]*hit.size); ai.extend(hit.tolist())
        dca.extend(dc.tolist()); dmn.extend(dm[hit].tolist())
    if not rr: return None
    # pocket+50%: include residues whose CA is within (max pocket CA-lig dist * 1.5)
    pc=ca[sorted(pocket)]
    ligcen=lig.mean(0)
    maxr=np.sqrt(((pc-ligcen)**2).sum(-1)).max()*EXT
    pocket_ext=sorted(set(ri for ri in range(len(prot)) if np.sqrt(((ca[ri]-ligcen)**2).sum())<=maxr))
    return dict(seq=seq, n_res=len(prot),
        pocket_res=np.array(pocket_ext,np.int32),
        pocket_ca=ca[pocket_ext].astype(np.float32),
        lig_xyz=lig.astype(np.float32),
        res_row=np.array(rr,np.int32), atom_idx=np.array(ai,np.int16),
        d_ca=np.array(dca,np.float16), d_min=np.array(dmn,np.float16))

def main():
    shard=int(sys.argv[1]); nshard=int(sys.argv[2])
    rows=list(csv.DictReader(open('/workspace/datasets/experimental_expansion/candidates_resolved.tsv'),delimiter='\t'))
    # dedup to one (pdb,ccd,chain) system; key system_id
    NUC=set("dna rna DA DT DG DC DU A U G C I N PSU UR3 OMG OMU 5CM 5NC 1MA 5MC 7MG 2MG H2U".split())
    seen=set(); systems=[]
    for r in rows:
        lig=r['ligand_ccd']
        if lig in NUC or lig.lower() in ('dna','rna'): continue
        if len(lig)>3 or len(lig)<2: continue
        sid=f"{r['pdb']}_{lig}_{r['recep_chain']}"
        if sid in seen: continue
        seen.add(sid); systems.append((sid,r['pdb'],lig,r['recep_chain'],r['uniprot']))
    mine=[s for i,s in enumerate(systems) if i%nshard==shard]
    done=ok=fail=0
    SID=[];UNI=[];SEQ=[];NRES=[]; poff=[0]; pres=[]; pca=[]
    coff=[0]; rr=[];ai=[];dca=[];dmn=[]; loff=[0]; lig=[]
    for sid,pdb,ccd,chain,uni in mine:
        if not os.path.exists(f"{STR}/{pdb}.cif"): fail+=1; continue
        try: r=extract(pdb,ccd,chain)
        except Exception: r=None
        if r is None: fail+=1; continue
        SID.append(sid); UNI.append(uni); SEQ.append(r['seq']); NRES.append(r['n_res'])
        pres.extend(r['pocket_res'].tolist()); pca.append(r['pocket_ca']); poff.append(len(pres))
        rr.extend(r['res_row'].tolist()); ai.extend(r['atom_idx'].tolist())
        dca.extend(r['d_ca'].tolist()); dmn.extend(r['d_min'].tolist()); coff.append(len(rr))
        lig.append(r['lig_xyz']); loff.append(loff[-1]+len(r['lig_xyz'])); ok+=1
    np.savez_compressed(f"{OUT}/crystal_labels_{shard}.npz",
        system_id=np.array(SID), uniprot=np.array(UNI), seq=np.array(SEQ), n_res=np.array(NRES,np.int32),
        pocket_offsets=np.array(poff,np.int64), pocket_res=np.array(pres,np.int32),
        pocket_ca=np.concatenate(pca).astype(np.float32) if pca else np.zeros((0,3),np.float32),
        contact_offsets=np.array(coff,np.int64), res_row=np.array(rr,np.int32), atom_idx=np.array(ai,np.int16),
        d_ca=np.array(dca,np.float16), d_min=np.array(dmn,np.float16),
        lig_offsets=np.array(loff,np.int64), lig_xyz=np.concatenate(lig).astype(np.float32) if lig else np.zeros((0,3),np.float32),
        cutoff=np.float32(CUTOFF), source=np.array("experimental_crystal"))
    print(f"shard {shard}: ok={ok} fail={fail}")

if __name__=="__main__": main()
