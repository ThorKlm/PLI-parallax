"""Build Boltz YAML + Chai FASTA inputs for the 494 PLINDER calibration systems.
Protein sequence reconstructed from proteins_v10 aa_idx (CA order == proteins_v10
rows), ligand from experimental SMILES. Emits an index TSV keyed by system_id.
"""
import os, json, numpy as np
ALPHA="ACDEFGHIKLMNPQRSTVWY"
d=np.load('/workspace/datasets/plinder_experimental_labels_aligned.npz',allow_pickle=True)
meta=json.load(open('/workspace/docking/input/meta.json'))
pidx={u:i for i,u in enumerate(meta['protein_ids'])}
P=np.load('/workspace/docking/input/proteins_v10.npz',allow_pickle=True,mmap_mode='r')
off=P['offsets']; aa=P['aa_idx']
MAXLEN=800
outb="/workspace/docking/output/calib_boltz_in"; outc="/workspace/docking/output/calib_chai_in"
os.makedirs(outb,exist_ok=True); os.makedirs(outc,exist_ok=True)
def seq_of(u):
    i=pidx[u]; return ''.join(ALPHA[int(v)] if int(v)<20 else 'X' for v in aa[off[i]:off[i+1]])
idx=open("/workspace/docking/output/calib_index.tsv","w")
idx.write("system_id\tuniprot\tpdb_id\ttier\tseq_len\tsmiles\n")
n=skip=0
for k in range(len(d['system_id'])):
    sid=str(d['system_id'][k]); u=str(d['uniprot'][k]); smi=d['ligand_smiles'][k]
    if not smi: skip+=1; continue
    s=seq_of(u)
    if len(s)>MAXLEN: skip+=1; continue
    safe=sid.replace('.','_').replace('__','_')
    # boltz yaml
    with open(os.path.join(outb,f"{safe}.yaml"),"w") as f:
        f.write("version: 1\nsequences:\n")
        f.write(f"  - protein:\n      id: A\n      sequence: {s}\n      msa: empty\n")
        f.write(f"  - ligand:\n      id: B\n      smiles: '{smi}'\n")
    # chai fasta
    with open(os.path.join(outc,f"{safe}.fasta"),"w") as f:
        f.write(f">protein|A\n{s}\n>ligand|B\n{smi}\n")
    idx.write(f"{sid}\t{u}\t{d['pdb_id'][k]}\t{d['tier'][k]}\t{len(s)}\t{smi}\n")
    n+=1
idx.close()
print(f"built {n} systems, skipped {skip} (no smiles or >{MAXLEN} res)")
