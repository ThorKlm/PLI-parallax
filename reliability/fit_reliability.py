"""Build the corrected smina crystal meta, verify the staged pair, recompute every
quoted figure against the corrected deposit, and emit the reliability columns."""
import collections, glob, itertools, os, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
L='/workspace/deposit_v3/labels'
SM='/workspace/smina_v2_b/v3/labels/exp_smina_v3_*.npz'
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

print('\n=== 4. reliability columns')
def parq(n, tier, thr='contact_4A'):
    fn=('labels_crystal_groundtruth_contacts.parquet' if n=='crystal_groundtruth'
        else f'labels_{n}_{tier}_contacts.parquet')
    t=pq.read_table(f'{L}/{fn}', columns=['system_id','res_row',thr])
    d=t.filter(t[thr]).to_pydict(); o=collections.defaultdict(set)
    for s,r in zip(d['system_id'],d['res_row']): o[s].add(r)
    return o
def npzs(pat, ang=4.0):
    o={}
    for f in sorted(glob.glob(pat)):
        d=np.load(f, allow_pickle=True)
        sid,off,rr,dm=d['system_id'],d['contact_offsets'],d['res_row'],d['d_min']
        for i,s in enumerate(sid):
            a,b=int(off[i]),int(off[i+1]); o[str(s)]=set(rr[a:b][dm[a:b]<=ang].tolist())
    return o
def jac(a,b): return len(a&b)/len(a|b) if (a|b) else np.nan
def agree(T,ids): return np.array([np.mean([jac(T[a][s],T[b][s])
                    for a,b in itertools.combinations(T,2)]) for s in ids])
from sklearn.isotonic import IsotonicRegression
GT=parq('crystal_groundtruth','crystal')
Tk={'chai1':parq('chai1','crystal'),'boltz2':parq('boltz2','crystal'),'smina':npzs(SM)}
ik=sorted(set(GT).intersection(*[set(v) for v in Tk.values()]))
ak=agree(Tk,ik); mk=np.array([np.mean([jac(Tk[n][s],GT[s]) for n in Tk]) for s in ik])
rng=np.random.default_rng(0); perm=rng.permutation(len(ik)); h=len(ik)//2
iso=IsotonicRegression(out_of_bounds='clip').fit(ak[perm[:h]], mk[perm[:h]])
q90=np.quantile(np.abs(iso.predict(ak[perm[h:]])-mk[perm[h:]]), 0.9)
np.savez('/workspace/reports/reliability_fit.npz', cal_agreement=ak[perm[h:]], cal_observed=mk[perm[h:]], fit_agreement=ak[perm[:h]], fit_observed=mk[perm[:h]])
Tc={n:parq(n,'corpus') for n in ('chai1','boltz2','smina')}
ic=sorted(set.intersection(*[set(v) for v in Tc.values()]))
ac=agree(Tc,ic)
sysrel=pd.concat([
  pd.DataFrame({'system_id':ik,'tier':'crystal','agreement':ak,
                'pred_accuracy':iso.predict(ak),'conformal_halfwidth_90':q90}),
  pd.DataFrame({'system_id':ic,'tier':'corpus','agreement':ac,
                'pred_accuracy':iso.predict(ac),'conformal_halfwidth_90':q90})])
sysrel.to_parquet(f'{ST}/system_reliability.parquet', index=False)
print(f'  system_reliability.parquet {len(sysrel):,} rows '
      f'(crystal {len(ik):,}, corpus {len(ic):,})')
rows=[]
for tier,T,ids in (('crystal',Tk,ik),('corpus',Tc,ic)):
    for s in ids:
        cnt=collections.Counter()
        for n in T:
            for r in T[n][s]: cnt[r]+=1
        for r,k in cnt.items(): rows.append((s,tier,r,k))
sup=pd.DataFrame(rows, columns=['system_id','tier','res_row','n_teachers_asserting'])
sup.to_parquet(f'{ST}/residue_support.parquet', index=False)
print(f'  residue_support.parquet {len(sup):,} rows')
print(sup.groupby(['tier','n_teachers_asserting']).size().to_string())
