#!/usr/bin/env python3
"""Select PLINDER high-quality holo systems whose pocket UniProt overlaps my set.
Single protein chain, single proper ligand, validation-passing. Spread across
distinct proteins, prefer the 192 pair-accessions, cap 500. Read-only."""
import csv, json, collections
import pyarrow.parquet as pq

# ---- my accession sets ----
pair_acc = set()
with open('/workspace/docking/output/boltz_in_index.tsv') as fh:
    for row in csv.DictReader(fh, delimiter='\t'):
        pair_acc.add(row['accession'])
meta = json.load(open('/workspace/docking/input/meta.json'))
meta_pids = set(meta['protein_ids'])
print(f"pair-accessions: {len(pair_acc)}; meta protein_ids: {len(meta_pids)}")

# ---- annotation table ----
t = pq.read_table('plinder_cache/annotation_proj.parquet',
    columns=['system_id','entry_pdb_id','system_pocket_UniProt','system_num_protein_chains',
             'system_pass_validation_criteria','system_type','ligand_ccd_code',
             'ligand_smiles','ligand_rdkit_canonical_smiles','ligand_is_ion',
             'ligand_is_artifact','ligand_is_proper','ligand_num_interacting_residues'])
cols = {c: t.column(c).to_pylist() for c in t.column_names}
n = t.num_rows

# rows per system_id (to enforce single-ligand: exactly one ligand instance total)
rows_per_sys = collections.Counter(cols['system_id'])

cand = {}          # system_id -> meta dict (qualifying)
for i in range(n):
    sid = cols['system_id'][i]
    if rows_per_sys[sid] != 1:                      # single-ligand only
        continue
    if cols['system_type'][i] != 'holo':            # holo only (exclude 'ion')
        continue
    if cols['system_pass_validation_criteria'][i] is not True:
        continue
    if cols['system_num_protein_chains'][i] != 1:   # single protein chain
        continue
    if cols['ligand_is_proper'][i] is not True:      # drop ions/artifacts/non-proper
        continue
    if cols['ligand_is_ion'][i] or cols['ligand_is_artifact'][i]:
        continue
    up = cols['system_pocket_UniProt'][i]
    if not up:
        continue
    tier = 'A' if up in pair_acc else ('B' if up in meta_pids else None)
    if tier is None:
        continue
    cand[sid] = {
        'system_id': sid, 'pdb_id': cols['entry_pdb_id'][i], 'uniprot': up,
        'tier': tier, 'ccd': cols['ligand_ccd_code'][i] or '',
        'smiles': cols['ligand_smiles'][i] or '',
        'canon_smiles': cols['ligand_rdkit_canonical_smiles'][i] or '',
        'n_interacting': cols['ligand_num_interacting_residues'][i] or 0,
        'zip': (cols['entry_pdb_id'][i] or '0000')[1:3],
    }

print(f"qualifying candidate systems: {len(cand)}")
byA = [c for c in cand.values() if c['tier']=='A']
byB = [c for c in cand.values() if c['tier']=='B']
print(f"  tier A (192 pair-accessions): {len(byA)} systems, "
      f"{len(set(c['uniprot'] for c in byA))} distinct proteins")
print(f"  tier B (broader meta pids):   {len(byB)} systems, "
      f"{len(set(c['uniprot'] for c in byB))} distinct proteins")

# ---- spread across distinct proteins: round-robin within tier, A before B ----
def by_protein(lst):
    d = collections.defaultdict(list)
    for c in lst:
        d[c['uniprot']].append(c)
    # within a protein, prefer systems with more interacting residues (richer pocket)
    for u in d:
        d[u].sort(key=lambda c: -c['n_interacting'])
    return d

LIMIT = 500
chosen = []
seen = set()
for tier_list in (byA, byB):
    pd = by_protein(tier_list)
    # round-robin: take 1 per protein per pass until limit
    proteins = sorted(pd.keys())
    depth = 0
    while len(chosen) < LIMIT:
        progressed = False
        for u in proteins:
            if depth < len(pd[u]):
                c = pd[u][depth]
                if c['system_id'] not in seen:
                    chosen.append(c); seen.add(c['system_id'])
                    progressed = True
                    if len(chosen) >= LIMIT: break
        if not progressed: break
        depth += 1
    if len(chosen) >= LIMIT: break

distinct = len(set(c['uniprot'] for c in chosen))
print(f"\nCHOSEN: {len(chosen)} systems across {distinct} distinct proteins")
print(f"  tier A chosen: {sum(c['tier']=='A' for c in chosen)}; "
      f"tier B chosen: {sum(c['tier']=='B' for c in chosen)}")
json.dump(chosen, open('plinder_cache/chosen_systems.json','w'), indent=1)
print("wrote plinder_cache/chosen_systems.json")
