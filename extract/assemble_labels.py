#!/usr/bin/env python3
"""Assemble per-zip checkpoints into the final CSR npz matching smina_labels.npz,
in chosen_systems order, with per-system metadata. Then emit stats."""
import os, json, collections
import numpy as np

CKPT_DIR = 'plinder_cache/ckpt'
CUTOFF = 15.0
OUT = '/workspace/datasets/plinder_experimental_labels.npz'

chosen = json.load(open('plinder_cache/chosen_systems.json'))
meta_by_sid = {c['system_id']: c for c in chosen}

# load all checkpoints -> per-system arrays
sys_arrays = {}   # sid -> (rr, aa, dca, dmin)
fail_total = collections.Counter()
nbytes_total = 0
for f in sorted(os.listdir(CKPT_DIR)):
    if not f.endswith('.npz'): continue
    z = np.load(f"{CKPT_DIR}/{f}", allow_pickle=True)
    nbytes_total += int(z['nbytes'])
    fr = json.loads(str(z['fail']))
    for k, v in fr.items(): fail_total[k] += v
    sysids = list(z['sysids']); counts = z['counts']
    rr, aa, dca, dmin = z['rr'], z['aa'], z['dca'], z['dmin']
    off = 0
    for sid, n in zip(sysids, counts):
        n = int(n)
        sys_arrays[sid] = (rr[off:off+n], aa[off:off+n],
                           dca[off:off+n], dmin[off:off+n])
        off += n

# assemble in chosen order (only systems that produced contacts)
offsets = [0]
res_row, atom_idx, d_ca, d_min = [], [], [], []
sys_meta = []
for c in chosen:
    sid = c['system_id']
    if sid not in sys_arrays: continue
    rr, aa, dca, dmin = sys_arrays[sid]
    res_row.append(rr); atom_idx.append(aa); d_ca.append(dca); d_min.append(dmin)
    offsets.append(offsets[-1] + rr.size); sys_meta.append(c)

cat = lambda L, dt: (np.concatenate(L) if L else np.array([], dt))
out = dict(
    contact_offsets=np.asarray(offsets, np.int64),
    res_row=cat(res_row, np.int32), atom_idx=cat(atom_idx, np.int16),
    d_ca=cat(d_ca, np.float16), d_min=cat(d_min, np.float16),
    cutoff=np.float32(CUTOFF), source=np.str_('experimental'),
    system_id=np.array([c['system_id'] for c in sys_meta]),
    uniprot=np.array([c['uniprot'] for c in sys_meta]),
    pdb_id=np.array([c['pdb_id'] for c in sys_meta]),
    tier=np.array([c['tier'] for c in sys_meta]),
    ligand_ccd=np.array([c['ccd'] for c in sys_meta]),
    ligand_smiles=np.array([c['smiles'] for c in sys_meta], dtype=object),
    ligand_canon_smiles=np.array([c['canon_smiles'] for c in sys_meta], dtype=object),
)
np.savez(OUT, **out)

stats = dict(
    systems_chosen=len(chosen),
    systems_with_contacts=len(sys_meta),
    distinct_proteins=len(set(c['uniprot'] for c in sys_meta)),
    distinct_proteins_tierA=len(set(c['uniprot'] for c in sys_meta if c['tier']=='A')),
    distinct_proteins_tierB=len(set(c['uniprot'] for c in sys_meta if c['tier']=='B')),
    tierA_systems=sum(c['tier']=='A' for c in sys_meta),
    tierB_systems=sum(c['tier']=='B' for c in sys_meta),
    total_contacts=int(offsets[-1]),
    downloaded_bytes=int(nbytes_total),
    downloaded_MB=round(nbytes_total/1e6, 2),
    cap_bytes=1_000_000_000,
    npz_bytes=os.path.getsize(OUT),
    fail_reasons=dict(fail_total),
)
json.dump(stats, open('plinder_cache/assemble_stats.json','w'), indent=2)
print(json.dumps(stats, indent=2))
