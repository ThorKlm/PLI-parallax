"""Recompute every coverage figure quoted in the reference documents from the
deposit. Prints PASS or FAIL per claim with the recomputed value."""
import collections, pyarrow.parquet as pq
L = '/workspace/deposit_v3/labels'
def meta(n):
    fn = ('labels_crystal_groundtruth_meta.parquet' if n == 'crystal_groundtruth'
          else f'labels_{n}_meta.parquet')
    return set(pq.read_table(f'{L}/{fn}', columns=['system_id']).column('system_id').to_pylist())
def rows(f):
    return pq.read_metadata(f'{L}/{f}').num_rows
S = {k: meta(k) for k in
     ('smina_corpus','boltz2_corpus','chai1_corpus','boltz2_msa_corpus',
      'smina_crystal','boltz2_crystal','chai1_crystal','boltz2_msa_crystal',
      'crystal_groundtruth')}
claims = [
 ('corpus smina', len(S['smina_corpus']), 31713),
 ('corpus boltz2', len(S['boltz2_corpus']), 23494),
 ('corpus chai1', len(S['chai1_corpus']), 23485),
 ('corpus boltz2_msa', len(S['boltz2_msa_corpus']), 161),
 ('corpus triple', len(S['smina_corpus'] & S['boltz2_corpus'] & S['chai1_corpus']), 23451),
 ('corpus union', len(S['smina_corpus'] | S['boltz2_corpus'] | S['chai1_corpus']), 31746),
 ('crystal ground truth', len(S['crystal_groundtruth']), 19350),
 ('crystal smina', len(S['smina_crystal']), 11543),
 ('crystal chai1', len(S['chai1_crystal']), 9410),
 ('crystal boltz2', len(S['boltz2_crystal']), 8725),
 ('crystal boltz2_msa', len(S['boltz2_msa_crystal']), 1755),
 ('crystal chai1 and boltz2', len(S['chai1_crystal'] & S['boltz2_crystal']), 8720),
 ('crystal triple', len(S['chai1_crystal'] & S['boltz2_crystal'] & S['smina_crystal']), 7166),
 ('crystal triple plus GT', len(S['chai1_crystal'] & S['boltz2_crystal']
                                & S['smina_crystal'] & S['crystal_groundtruth']), 7166),
 ('GT contact rows', rows('labels_crystal_groundtruth_contacts.parquet'), 46881795),
 ('smina corpus contact rows', rows('labels_smina_corpus_contacts.parquet'), 70033209),
 ('chai1 crystal contact rows', rows('labels_chai1_crystal_contacts.parquet'), 21558787),
 ('total meta rows', sum(len(v) for v in S.values()), 129636),
]
bad = 0
for name, got, want in claims:
    ok = got == want
    bad += not ok
    print(f'{"PASS" if ok else "FAIL":4s} {name:30s} recomputed {got:>10,}  quoted {want:>10,}')
m = pq.read_table(f'{L}/labels_chai1_corpus_meta.parquet', columns=['protein_id'])
print(f'\ncorpus distinct proteins: {len(set(x for x in m.column("protein_id").to_pylist() if x))} (quoted 906)')
import json
d = json.load(open('/workspace/deposit_v3/metadata/ligand_identity.json'))
print(f'distinct corpus inchikeys: {len({v["inchikey"] for v in d.values()})} (quoted 31,193 / 31,064)')
print(f'\n{bad} FAIL' if bad else '\nall claims reproduce')
