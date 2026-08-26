"""Chain-aware cofolding inputs for the crystal tier.

Replaces the single fused protein entity of ``datasets/build_exp_inputs.py`` with
one protein entity per crystal chain, in mmCIF chain iteration order, so a
homo-N-mer is handed to the cofolder as N chains rather than an N-fold tandem
repeat.  Nothing existing is read-modify-written: outputs go to new directories.

  boltz_in_mc/<system_id>.yaml      one ``- protein:`` block per chain + ligand
  chai_in_mc/<system_id>.fasta      one ``>protein|`` record per chain + ligand
  exp_fold_index_mc.tsv / .parquet  manifest incl. the chain table and the
                                    Boltz chain-emission permutation

``--msa``
---------
Default output carries ``msa: empty`` on every protein entity, which is what puts
Boltz-2 in single-sequence mode.  ``--msa`` omits that key instead, so ``boltz
predict --use_msa_server`` generates a ColabFold MSA per chain; those YAMLs go to
``boltz_in_mc_msa`` and the manifest to the ``_msa``-suffixed paths, so the
default outputs are never rewritten.  The Chai FASTA carries no MSA directive and
is identical either way, so ``--msa`` does not re-emit it.  Nothing else about the
default run changes.

The 800-residue cap is applied to the **longest single chain**, not to the fused
length.

Chain order and ``res_row``
---------------------------
The deposited ``res_row`` indexes a chain-major enumeration in mmCIF chain
order.  Both prediction enumerators (``extract_exp_teacher.parse_pred``,
``bisy_rmsd_v2.predicted_residues``) walk ``for chain in model`` in *file*
order, so the two align iff the cofolder emits chains in input order:

* **Chai-1** does (``raw_inputs_to_entitites_data`` keeps FASTA record order,
  duplicate sequences share an ``entity_id`` but stay separate chains).
* **Boltz-2** does **not**.  ``parse_boltz_schema`` groups the ``sequences``
  items by ``(entity_type, sequence)`` before assigning ``asym_id``, and the
  mmCIF writer emits atoms in ``asym_id`` order.  Chains sharing a sequence are
  therefore emitted contiguously even when they are interleaved in the crystal.
  The resulting permutation is deterministic and is computed and stored here as
  ``boltz_chain_order`` -- do not assume identity.
"""
import argparse
import os, sys, json, glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "/workspace/reports/crystal_chain_schema.parquet"
CCD_SMILES = "/workspace/datasets/experimental_expansion/ccd_smiles.json"
LABEL_SHARDS = "/workspace/datasets/experimental_expansion/crystal_labels/crystal_labels_*.npz"
BASE = "/workspace/datasets/experimental_expansion"
OUTB = f"{BASE}/boltz_in_mc"
OUTB_MSA = f"{BASE}/boltz_in_mc_msa"
OUTC = f"{BASE}/chai_in_mc"
IDX_TSV = f"{BASE}/exp_fold_index_mc.tsv"
IDX_PQ = f"{BASE}/exp_fold_index_mc.parquet"
QUEUE = f"{BASE}/chai_queue_mc.txt"
SKIPPED = "/workspace/reports/exp_fold_index_mc_skipped.tsv"

MAXCHAIN = 800     # cap on the LONGEST SINGLE CHAIN
MINTOTAL = 10
LIG_ID = "LIG"


def chain_ids(n):
    """n fixed-width base-26 uppercase ids: A..Z, then AA..ZZ, ...

    Fixed width is deliberate -- lexicographic order then equals emission order,
    so a writer that sorts chains by name cannot silently reorder them.
    """
    width = 1
    while 26 ** width < n:
        width += 1
    out = []
    for i in range(n):
        s, x = "", i
        for _ in range(width):
            s = chr(ord('A') + x % 26) + s
            x //= 26
        out.append(s)
    return out


def boltz_chain_order(seqs):
    """Indices of the crystal chains in the order Boltz-2 will emit them.

    Replicates ``parse_boltz_schema``: items are grouped by (entity_type,
    sequence) with ``dict.setdefault``, groups are then walked in first-
    appearance order and each group's ids are appended in item order.
    """
    rank = {}
    for s in seqs:
        if s not in rank:
            rank[s] = len(rank)
    return sorted(range(len(seqs)), key=lambda i: (rank[seqs[i]], i))


def residue_permutation(counts, order):
    """crystal residue index -> residue index in the emitted prediction.

    ``counts`` are per-chain residue counts in crystal order; ``order`` is the
    emission order of the chains.
    """
    counts = list(counts)
    starts_crystal = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    out = np.empty(int(sum(counts)), np.int64)
    pos = 0
    for ci in order:
        n = counts[ci]
        out[starts_crystal[ci]:starts_crystal[ci] + n] = np.arange(pos, pos + n)
        pos += n
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--msa", action="store_true",
        help="omit the 'msa: empty' key so boltz predict --use_msa_server generates a "
             "ColabFold MSA per chain; writes boltz_in_mc_msa/ and _msa-suffixed "
             "manifests, leaving the default outputs untouched",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Every default path keeps its name; --msa only ever adds a suffixed sibling.
    suffix = "_msa" if args.msa else ""
    outb = OUTB_MSA if args.msa else OUTB
    idx_pq = IDX_PQ.replace(".parquet", f"{suffix}.parquet")
    idx_tsv = IDX_TSV.replace(".tsv", f"{suffix}.tsv")
    queue = QUEUE.replace(".txt", f"{suffix}.txt")
    skipped_path = SKIPPED.replace(".tsv", f"{suffix}.tsv")

    df = pd.read_parquet(SCHEMA)
    smiles = json.load(open(CCD_SMILES))
    os.makedirs(outb, exist_ok=True)
    if not args.msa:
        os.makedirs(OUTC, exist_ok=True)

    rows, skipped = [], []
    for r in df.itertuples(index=False):
        sid = r.system_id
        if r.parse_status != "ok":
            skipped.append((sid, "mmcif_unparseable")); continue
        smi = smiles.get(r.ligand_ccd)
        if not smi:
            skipped.append((sid, "no_smiles")); continue
        counts = list(r.chain_res_counts)
        seqs = list(r.chain_seqs)
        total = int(sum(counts))
        longest = int(max(counts))
        if total < MINTOTAL:
            skipped.append((sid, f"total_len<{MINTOTAL}")); continue
        if longest > MAXCHAIN:
            skipped.append((sid, f"longest_chain>{MAXCHAIN} ({longest})")); continue

        ids = chain_ids(len(seqs))
        safe = sid.replace('.', '_')

        # With --msa the key is left out entirely rather than set to some other value:
        # Boltz treats a protein entity with no 'msa' as one to generate an MSA for.
        msa_line = "" if args.msa else "      msa: empty\n"
        with open(f"{outb}/{safe}.yaml", 'w') as fh:
            fh.write("version: 1\nsequences:\n")
            for cid, sq in zip(ids, seqs):
                # ids and sequences are quoted: YAML 1.1 coerces the bare tokens
                # N, Y, NO, ON, YES, FALSE ... to booleans, and those occur both as
                # chain ids (838 systems) and as 1-residue chain sequences (4 systems).
                fh.write(f'  - protein:\n      id: "{cid}"\n      sequence: "{sq}"\n' + msa_line)
            fh.write(f"  - ligand:\n      id: {LIG_ID}\n      smiles: '{smi}'\n")

        if not args.msa:
            with open(f"{OUTC}/{safe}.fasta", 'w') as fh:
                for cid, sq in zip(ids, seqs):
                    fh.write(f">protein|name={cid}\n{sq}\n")
                fh.write(f">ligand|name={LIG_ID}\n{smi}\n")

        border = boltz_chain_order(seqs)
        rows.append(dict(
            system_id=sid, safe_name=safe, pdb=r.pdb, ligand_ccd=r.ligand_ccd, smiles=smi,
            n_chains=len(seqs), total_len=total, longest_chain_len=longest,
            min_chain_len=int(min(counts)),
            crystal_chain_ids=list(r.chain_ids), emitted_chain_ids=ids,
            chain_res_counts=[int(c) for c in counts],
            chain_res_offsets=[int(o) for o in r.chain_res_offsets],
            boltz_chain_order=[int(i) for i in border],
            boltz_order_is_identity=bool(border == list(range(len(seqs)))),
            chai_order_is_identity=True,
            n_entities_exact=int(r.n_entities_exact),
            stoichiometry_exact=r.stoichiometry_exact,
            stoichiometry_relaxed=r.stoichiometry_relaxed,
            is_homo_oligomer_exact=bool(r.is_homo_oligomer_exact),
            is_homo_oligomer_relaxed=bool(r.is_homo_oligomer_relaxed),
            lig_top_chain=r.lig_top_chain, interface_binder_4A=bool(r.interface_binder_4A),
            fused_cap_would_admit=bool(r.foldable_fused_800),
        ))

    out = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), idx_pq, compression='zstd')
    with open(idx_tsv, 'w') as fh:
        fh.write("system_id\tccd\tn_chains\ttotal_len\tlongest_chain_len\tchain_res_counts\t"
                 "boltz_chain_order\tboltz_order_is_identity\tsmiles\n")
        for r in out.itertuples(index=False):
            fh.write(f"{r.system_id}\t{r.ligand_ccd}\t{r.n_chains}\t{r.total_len}\t"
                     f"{r.longest_chain_len}\t{','.join(map(str, r.chain_res_counts))}\t"
                     f"{','.join(map(str, r.boltz_chain_order))}\t{int(r.boltz_order_is_identity)}\t{r.smiles}\n")
    with open(queue, 'w') as fh:
        for r in out.itertuples(index=False):
            fh.write(r.safe_name + "\n")
    with open(skipped_path, 'w') as fh:
        fh.write("system_id\treason\n")
        for s, why in skipped:
            fh.write(f"{s}\t{why}\n")

    print(f"built {len(out)} multimer inputs; skipped {len(skipped)}"
          + (" (--msa: ColabFold MSA, no 'msa: empty')" if args.msa else ""))
    print(f"  boltz yaml : {outb}")
    print(f"  chai fasta : {'(unchanged, --msa)' if args.msa else OUTC}")
    print(f"  index      : {idx_pq}")
    if skipped:
        rs = pd.Series([w.split(' ')[0] for _, w in skipped]).value_counts()
        print("skip reasons:\n" + rs.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
