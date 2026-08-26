import numpy as np, json, os, glob
# build boltz yaml + chai fasta per crystal system, streaming the 12 shards
shards = sorted(glob.glob('/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz'))
ccd_smiles = json.load(open('/workspace/datasets/experimental_expansion/ccd_smiles.json'))
outb = '/workspace/datasets/experimental_expansion/boltz_in'
outc = '/workspace/datasets/experimental_expansion/chai_in'
os.makedirs(outb, exist_ok=True); os.makedirs(outc, exist_ok=True)
MAXLEN = 800
idx = open('/workspace/datasets/experimental_expansion/exp_fold_index.tsv', 'w')
idx.write("system_id\tccd\tseq_len\tsmiles\n")
q = open('/workspace/docking/output/chai_queue_exp.txt', 'w')
n = skip = 0
for f in shards:
    d = np.load(f, allow_pickle=True)
    sid, seq = d['system_id'], d['seq']
    for i in range(len(sid)):
        s = str(sid[i]); ccd = s.split('_')[1]; sq = str(seq[i])
        smi = ccd_smiles.get(ccd)
        if not smi or not (10 <= len(sq) <= MAXLEN): skip += 1; continue
        safe = s.replace('.', '_')
        with open(f"{outb}/{safe}.yaml", 'w') as fh:
            fh.write("version: 1\nsequences:\n")
            fh.write(f"  - protein:\n      id: A\n      sequence: {sq}\n      msa: empty\n")
            fh.write(f"  - ligand:\n      id: B\n      smiles: '{smi}'\n")
        with open(f"{outc}/{safe}.fasta", 'w') as fh:
            fh.write(f">protein|name=P1\n{sq}\n>ligand|name=L1\n{smi}\n")
        idx.write(f"{s}\t{ccd}\t{len(sq)}\t{smi}\n"); q.write(safe + '\n'); n += 1
    del d
idx.close(); q.close()
print("built", n, "fold inputs, skipped", skip)
