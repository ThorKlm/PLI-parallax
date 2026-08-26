"""Chain schema for the 19,350 crystal ground-truth systems.

Re-parses every deposited mmCIF with the *identical* filter used by
``datasets/extract_crystal_labels.py`` (model 0; every chain in mmCIF iteration
order; standard amino acids only; residue kept iff it has a CA and >=1 heavy
atom) and records the per-chain structure that the deposited labels throw away.

Writes /workspace/reports/crystal_chain_schema.parquet.  Read-only with respect
to every existing artefact.
"""
import os, sys, glob, json
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gemmi

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
T = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
     "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
     "TYR":"Y","VAL":"V"}

STR = "/workspace/datasets/experimental_expansion/structures"
META = "/workspace/deposit_v3/labels/labels_crystal_groundtruth_meta.parquet"
CONTACTS = "/workspace/deposit_v3/labels/labels_crystal_groundtruth_contacts.parquet"
LABEL_SHARDS = "/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz"
OUT = "/workspace/reports/crystal_chain_schema.parquet"

HOMO_COMP_RATIO = 0.90   # relaxed entity grouping threshold (matches the audit)


def parse_pdb(pdb):
    """(pdb, [chain_id], [n_res], [seq]) over chains contributing >=1 kept residue."""
    path = f"{STR}/{pdb}.cif"
    if not os.path.exists(path):
        return pdb, None, None, None
    try:
        m = gemmi.read_structure(path)[0]
    except Exception:
        return pdb, None, None, None
    ids, counts, seqs = [], [], []
    for ch in m:
        letters = []
        for res in ch:
            if res.name not in AA:
                continue
            ca = False; heavy = False
            for a in res:
                if a.element.name in ('H', 'D'):
                    continue
                heavy = True
                if a.name == 'CA':
                    ca = True
            if not ca or not heavy:
                continue
            letters.append(T[res.name])
        if letters:
            ids.append(ch.name); counts.append(len(letters)); seqs.append(''.join(letters))
    return pdb, ids, counts, seqs


def entity_groups_exact(seqs):
    """group index per chain, by exact sequence equality; ids in first-appearance order."""
    seen, out = {}, []
    for s in seqs:
        if s not in seen:
            seen[s] = len(seen)
        out.append(seen[s])
    return out


def entity_groups_relaxed(seqs):
    """Union-find grouping: two chains share an entity when their amino-acid
    compositions overlap by >= HOMO_COMP_RATIO of the longer chain.  Tolerates
    copies of one entity truncated differently by disorder."""
    n = len(seqs)
    comps = [Counter(s) for s in seqs]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            inter = sum((comps[i] & comps[j]).values())
            if inter / max(len(seqs[i]), len(seqs[j])) >= HOMO_COMP_RATIO:
                parent[find(j)] = find(i)
    # relabel in first-appearance order
    lab, out = {}, []
    for i in range(n):
        r = find(i)
        if r not in lab:
            lab[r] = len(lab)
        out.append(lab[r])
    return out


def stoich_string(groups):
    """'A2B1' style, entities lettered in first-appearance order."""
    c = Counter(groups)
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = []
    for g in sorted(c):
        parts.append(f"{alpha[g] if g < 26 else 'E%d' % g}{c[g]}")
    return ''.join(parts)


def main():
    meta = pd.read_parquet(META, columns=['system_id', 'ligand_ccd', 'n_res'])
    sids = meta['system_id'].tolist()
    dep_nres = dict(zip(meta['system_id'], meta['n_res'].astype(int)))
    print(f"systems: {len(sids)}", flush=True)

    # deposited per-system sequence (the exact enumeration res_row indexes)
    dep_seq = {}
    for f in sorted(glob.glob(LABEL_SHARDS)):
        d = np.load(f, allow_pickle=True)
        for s, q in zip(d['system_id'], d['seq']):
            dep_seq[str(s)] = str(q)
        del d
    print(f"deposited sequences loaded: {len(dep_seq)}", flush=True)

    pdbs = sorted({s.split('_')[0] for s in sids})
    print(f"unique mmCIFs to parse: {len(pdbs)}", flush=True)
    with Pool(24) as p:
        parsed = {}
        for i, (pdb, ids, counts, seqs) in enumerate(p.imap_unordered(parse_pdb, pdbs, chunksize=16)):
            parsed[pdb] = (ids, counts, seqs)
            if (i + 1) % 2000 == 0:
                print(f"  parsed {i+1}/{len(pdbs)}", flush=True)
    print("parse done", flush=True)

    # per-pdb derived entity structure (independent of ligand -> compute once)
    pdb_ent = {}
    for pdb, (ids, counts, seqs) in parsed.items():
        if ids is None:
            pdb_ent[pdb] = None
            continue
        ge = entity_groups_exact(seqs)
        gr = entity_groups_relaxed(seqs) if len(seqs) <= 60 else ge
        pdb_ent[pdb] = (ge, gr)

    # ---- ligand contact chain, from the deposited contacts (res_row space) ----
    print("reading contacts ...", flush=True)
    ct = pd.read_parquet(CONTACTS, columns=['system_id', 'res_row', 'contact_4A', 'contact_5A'])
    max_rr = ct.groupby('system_id', observed=True)['res_row'].max().to_dict()
    c4 = ct.loc[ct['contact_4A'], ['system_id', 'res_row']].drop_duplicates()
    c5 = ct.loc[ct['contact_5A'], ['system_id', 'res_row']].drop_duplicates()
    del ct
    con4 = c4.groupby('system_id', observed=True)['res_row'].apply(np.array).to_dict()
    con5 = c5.groupby('system_id', observed=True)['res_row'].apply(np.array).to_dict()
    del c4, c5
    print(f"contact_4A systems: {len(con4)}  contact_5A systems: {len(con5)}", flush=True)

    rows = []
    problems = []
    for sid in sids:
        pdb, ccd, keych = sid.split('_', 2)
        ids, counts, seqs = parsed[pdb]
        if ids is None:
            problems.append((sid, "mmCIF missing or unparseable"))
            rows.append(dict(system_id=sid, pdb=pdb, ligand_ccd=ccd, key_chain=keych,
                             n_chains=0, chain_ids=[], chain_res_counts=[], chain_seqs=[],
                             chain_res_offsets=[], n_res_reconstructed=0,
                             n_res_deposited=dep_nres[sid], n_res_match=False,
                             seq_match=False, longest_chain_len=0, fused_len=0,
                             n_entities_exact=0, stoichiometry_exact="",
                             is_homo_oligomer_exact=False, n_entities_relaxed=0,
                             stoichiometry_relaxed="", is_homo_oligomer_relaxed=False,
                             lig_top_chain="", lig_top_chain_index=-1,
                             n_contact_res_4A=0, n_contact_chains_4A=0,
                             frac_contacts_top_chain_4A=float('nan'),
                             interface_binder_4A=False, n_contact_chains_5A=0,
                             interface_binder_5A=False, max_res_row=-1,
                             res_row_in_range=False, foldable_fused_800=False,
                             foldable_longest_800=False, parse_status="parse_failed"))
            continue

        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int).tolist() if counts else []
        total = int(sum(counts))
        recon_seq = ''.join(seqs)
        nres_dep = dep_nres[sid]
        nres_ok = (total == nres_dep)
        seq_ok = (recon_seq == dep_seq[sid])
        if not nres_ok:
            problems.append((sid, f"n_res mismatch: reconstructed={total} deposited={nres_dep}"))
        if not seq_ok:
            problems.append((sid, "reconstructed concatenated sequence != deposited seq"))

        ge, gr = pdb_ent[pdb]
        ce, cr = Counter(ge), Counter(gr)

        # chain of each contact residue, via the offsets
        bounds = np.array(offsets + [total])
        rr4 = con4.get(sid, np.zeros(0, int))
        rr5 = con5.get(sid, np.zeros(0, int))
        if rr4.size:
            ch4 = np.searchsorted(bounds, rr4, side='right') - 1
            cnt = Counter(ch4.tolist())
            top_idx = max(cnt, key=lambda k: (cnt[k], -k))
            top_chain = ids[top_idx] if 0 <= top_idx < len(ids) else ""
            frac = cnt[top_idx] / rr4.size
            n_ch4 = len(cnt)
        else:
            top_idx, top_chain, frac, n_ch4 = -1, "", float('nan'), 0
        n_ch5 = len(set((np.searchsorted(bounds, rr5, side='right') - 1).tolist())) if rr5.size else 0

        mrr = int(max_rr.get(sid, -1))
        rows.append(dict(
            system_id=sid, pdb=pdb, ligand_ccd=ccd, key_chain=keych,
            n_chains=len(ids), chain_ids=ids, chain_res_counts=[int(x) for x in counts],
            chain_seqs=seqs, chain_res_offsets=offsets,
            n_res_reconstructed=total, n_res_deposited=int(nres_dep), n_res_match=bool(nres_ok),
            seq_match=bool(seq_ok),
            longest_chain_len=int(max(counts)), fused_len=total,
            n_entities_exact=len(ce), stoichiometry_exact=stoich_string(ge),
            is_homo_oligomer_exact=bool(len(ids) > 1 and len(ce) == 1),
            n_entities_relaxed=len(cr), stoichiometry_relaxed=stoich_string(gr),
            is_homo_oligomer_relaxed=bool(len(ids) > 1 and len(cr) == 1),
            lig_top_chain=top_chain, lig_top_chain_index=int(top_idx),
            n_contact_res_4A=int(rr4.size), n_contact_chains_4A=int(n_ch4),
            frac_contacts_top_chain_4A=float(frac),
            interface_binder_4A=bool(n_ch4 >= 2),
            n_contact_chains_5A=int(n_ch5), interface_binder_5A=bool(n_ch5 >= 2),
            max_res_row=mrr, res_row_in_range=bool(mrr < total),
            foldable_fused_800=bool(10 <= total <= 800),
            foldable_longest_800=bool(max(counts) <= 800 and total >= 10),
            parse_status="ok"))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT, compression='zstd')
    print(f"wrote {OUT}: {len(df)} rows", flush=True)

    # ---------------- validation report ----------------
    n_nres_bad = int((~df.n_res_match).sum())
    n_seq_bad = int((~df.seq_match).sum())
    n_rr_bad = int((~df.res_row_in_range).sum())
    print("\n=== VALIDATION ===")
    print(f"systems                                : {len(df)}")
    print(f"summed chain residues == deposited n_res: {len(df)-n_nres_bad}/{len(df)}  (failures {n_nres_bad})")
    print(f"concatenated chain seq == deposited seq : {len(df)-n_seq_bad}/{len(df)}  (failures {n_seq_bad})")
    print(f"max(res_row) < n_res                    : {len(df)-n_rr_bad}/{len(df)}  (failures {n_rr_bad})")
    if problems:
        print("\nPROBLEM SYSTEMS:")
        for s, why in problems[:200]:
            print(f"  {s}: {why}")
        print(f"  ... total {len(problems)}")
    with open('/workspace/reports/crystal_chain_schema_validation.txt', 'w') as fh:
        fh.write(f"systems: {len(df)}\n")
        fh.write(f"n_res match failures: {n_nres_bad}\n")
        fh.write(f"seq match failures: {n_seq_bad}\n")
        fh.write(f"res_row out of range: {n_rr_bad}\n")
        for s, why in problems:
            fh.write(f"{s}\t{why}\n")
    print("\nchain-count distribution:")
    print(df.n_chains.value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
