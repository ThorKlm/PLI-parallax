"""One-off repair of the smina crystal arm after the re-dock.

Builds the corrected meta table from the re-docked contacts, checks it against the
superseded pair for schema and set identity, and recomputes the coverage figures
quoted in the article. The reliability field is fitted separately by
reliability/fit_reliability.py, which reads the deposit rather than the re-dock
staging directory.
"""

import glob, os, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
L='/workspace/deposit_v3/labels'
ST='/workspace/final_five_rerun'
os.makedirs(ST, exist_ok=True)

print('=== 1. build corrected smina crystal meta')
old=pq.read_table(f'{L}/labels_smina_crystal_meta.parquet')
schema=old.schema
new_ids=set(pq.read_table(f'{ST}/labels_smina_crystal_contacts.parquet',
                          columns=['system_id']).column('system_id').to_pylist())
o=old.to_pandas()
keep=o[o.system_id.isin(new_ids)].copy()
res=pd.read_csv('/workspace/smina_v2_b/v3/out/results_v4.csv')
aff=dict(zip(res.system_id, pd.to_numeric(res.affinity, errors='coerce')))
keep['affinity']=keep.system_id.map(aff).astype('float32')
def ids_of(f): return set(pq.read_table(f'{L}/{f}',columns=['system_id']).column('system_id').to_pylist())
c=ids_of('labels_chai1_crystal_meta.parquet'); b=ids_of('labels_boltz2_crystal_meta.parquet')
keep['n_teachers']=[1+ (s in c) + (s in b) for s in keep.system_id]
keep['in_triple_core']=[(s in c) and (s in b) for s in keep.system_id]
tbl=pa.Table.from_pandas(keep[schema.names], schema=schema, preserve_index=False)
pq.write_table(tbl, f'{ST}/labels_smina_crystal_meta.parquet', compression='snappy')
print(f'  rows {tbl.num_rows} (was {old.num_rows}) | affinity non-null {keep.affinity.notna().sum()}')
print(f'  in_triple_core True: {keep.in_triple_core.sum()}')

print('\n=== 2. verify the staged pair')
cs=pq.read_schema(f'{ST}/labels_smina_crystal_contacts.parquet')
os_=pq.read_schema(f'{L}/labels_smina_crystal_contacts.parquet')
print(f'  contacts schema names match: {cs.names==os_.names}')
print(f'  contacts types match: {[str(t) for t in cs.types]==[str(t) for t in os_.types]}')
print(f'  meta schema identical: {tbl.schema.equals(schema)}')
mset=set(keep.system_id); cset=new_ids
print(f'  meta and contacts system sets equal: {mset==cset} ({len(mset)} vs {len(cset)})')
ct=pq.read_table(f'{ST}/labels_smina_crystal_contacts.parquet', columns=['d_ca','d_min'])
dc=ct.column('d_ca').to_numpy(); dm=ct.column('d_min').to_numpy()
print(f'  d_min <= d_ca violations: {(dm>dc).sum()} | NaN {np.isnan(dm).sum()} | over 15A {(dm>15).sum()}')
metas=[pq.read_table(f) for f in sorted(glob.glob(f'{L}/*_meta.parquet'))
       if 'smina_crystal' not in f]+[tbl]
try:
    print(f'  naive concat over all meta: {pa.concat_tables(metas).num_rows} rows')
except Exception as e:
    print('  CONCAT FAILS:', e)

print('\n=== 3. recompute quoted figures against the corrected deposit')
S={k:ids_of(f'labels_{k}_meta.parquet') for k in
   ('smina_corpus','boltz2_corpus','chai1_corpus','boltz2_msa_corpus',
    'boltz2_crystal','chai1_crystal','boltz2_msa_crystal','crystal_groundtruth')}
S['smina_crystal']=cset
claims=[('corpus triple', len(S['smina_corpus']&S['boltz2_corpus']&S['chai1_corpus']), 23451),
        ('corpus union', len(S['smina_corpus']|S['boltz2_corpus']|S['chai1_corpus']), 31746),
        ('crystal triple', len(S['chai1_crystal']&S['boltz2_crystal']&S['smina_crystal']), None),
        ('crystal triple+GT', len(S['chai1_crystal']&S['boltz2_crystal']&S['smina_crystal']&S['crystal_groundtruth']), None),
        ('crystal smina', len(S['smina_crystal']), None),
        ('total meta rows', sum(len(v) for v in S.values()), None)]
for n,got,want in claims:
    tag='' if want is None else (' PASS' if got==want else f' FAIL (quoted {want:,})')
    print(f'  {n:22s} {got:>10,}{tag}')