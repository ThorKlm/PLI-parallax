import numpy as np, glob, os
parts = sorted(glob.glob('/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz'))
OUT = '/workspace/datasets/experimental_expansion/crystal_labels_final.npz'
TMP = '/workspace/datasets/experimental_expansion/_merge_tmp'
os.makedirs(TMP, exist_ok=True)
# pass 1: totals for preallocation
n_sys = n_pres = n_con = n_lig = 0
for f in parts:
    d = np.load(f, allow_pickle=True)
    n_sys += d['system_id'].size
    n_pres += d['pocket_res'].size
    n_con += d['res_row'].size
    n_lig += d['lig_xyz'].shape[0]
# memmaps for the large flat arrays
res_row = np.memmap(f'{TMP}/res_row.dat', np.int32, 'w+', shape=(n_con,))
atom_idx = np.memmap(f'{TMP}/atom_idx.dat', np.int16, 'w+', shape=(n_con,))
d_ca = np.memmap(f'{TMP}/d_ca.dat', np.float16, 'w+', shape=(n_con,))
d_min = np.memmap(f'{TMP}/d_min.dat', np.float16, 'w+', shape=(n_con,))
pocket_res = np.memmap(f'{TMP}/pocket_res.dat', np.int32, 'w+', shape=(n_pres,))
pocket_ca = np.memmap(f'{TMP}/pocket_ca.dat', np.float32, 'w+', shape=(n_pres, 3))
lig_xyz = np.memmap(f'{TMP}/lig_xyz.dat', np.float32, 'w+', shape=(n_lig, 3))
# small per-system arrays kept in RAM
sid=[]; uni=[]; seq=[]; nres=[]
poff=[0]; coff=[0]; loff=[0]
cp=cc=cl=0  # running fill positions
for f in parts:
    d = np.load(f, allow_pickle=True)
    sid.append(d['system_id']); uni.append(d['uniprot']); seq.append(d['seq']); nres.append(d['n_res'])
    k = d['res_row'].size
    res_row[cc:cc+k]=d['res_row']; atom_idx[cc:cc+k]=d['atom_idx']
    d_ca[cc:cc+k]=d['d_ca']; d_min[cc:cc+k]=d['d_min']
    for L in np.diff(d['contact_offsets']): coff.append(coff[-1]+int(L))
    cc += k
    kp = d['pocket_res'].size
    pocket_res[cp:cp+kp]=d['pocket_res']; pocket_ca[cp:cp+kp]=d['pocket_ca']
    for L in np.diff(d['pocket_offsets']): poff.append(poff[-1]+int(L))
    cp += kp
    kl = d['lig_xyz'].shape[0]
    lig_xyz[cl:cl+kl]=d['lig_xyz']
    for L in np.diff(d['lig_offsets']): loff.append(loff[-1]+int(L))
    cl += kl
    del d
np.savez_compressed(OUT,
    system_id=np.concatenate(sid), uniprot=np.concatenate(uni),
    seq=np.concatenate(seq), n_res=np.concatenate(nres),
    pocket_offsets=np.array(poff,np.int64), pocket_res=np.asarray(pocket_res), pocket_ca=np.asarray(pocket_ca),
    contact_offsets=np.array(coff,np.int64), res_row=np.asarray(res_row), atom_idx=np.asarray(atom_idx),
    d_ca=np.asarray(d_ca), d_min=np.asarray(d_min),
    lig_offsets=np.array(loff,np.int64), lig_xyz=np.asarray(lig_xyz),
    cutoff=np.float32(15.0), source=np.array("experimental_crystal"))
print("merged systems", np.concatenate(sid).size, "contacts", n_con)
