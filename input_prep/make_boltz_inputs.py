"""Emit one Boltz YAML per unique pair (protein sequence in CA order to match
proteins_v10 rows, ligand SMILES, single-sequence MSA). Boltz ignores pockets,
so this is per pair, not per pocket. A length cap keeps complexes within the
24 GB card. An index TSV maps each YAML name back to pair/compound identity.
Run: python make_boltz_inputs.py /workspace/docking/input \
         /workspace/docking/output/manifest_shuf.tsv \
         /workspace/docking/output/boltz_in --max-len 800 --limit 0
"""
import argparse, csv, os
T = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
     "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
     "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def sequence(pdb):
    seq, seen = [], set()
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[16] in " A":
            k = (ln[21], ln[22:26].strip())
            if k in seen: continue
            seen.add(k); seq.append(T.get(ln[17:20].strip(), "X"))
    return "".join(seq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("manifest"); ap.add_argument("indir")
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.indir, exist_ok=True)
    seen_pairs = set(); n, skip_len = 0, 0
    idx_path = a.indir.rstrip("/") + "_index.tsv"  # sibling, NOT inside indir
    with open(a.manifest) as fh, open(idx_path, "w") as ix:
        ix.write("name\tpair_idx\taccession\tcompound_idx\tseq_len\tsmiles\n")
        for r in csv.DictReader(fh, delimiter="\t"):
            pair = int(r["pair_idx"])
            if pair in seen_pairs: continue
            seen_pairs.add(pair)
            acc = r["accession"]; cid = r["compound_idx"]; smi = r["smiles"]
            seq = sequence(os.path.join(a.root, "pdbs", f"AF-{acc}-F1-model_v6.pdb"))
            if not seq or len(seq) > a.max_len: skip_len += 1; continue
            name = str(pair)
            with open(os.path.join(a.indir, f"{name}.yaml"), "w") as y:
                y.write("version: 1\nsequences:\n  - protein:\n      id: A\n"
                        f"      sequence: {seq}\n      msa: empty\n"
                        f"  - ligand:\n      id: L\n      smiles: '{smi}'\n")
            ix.write(f"{name}\t{pair}\t{acc}\t{cid}\t{len(seq)}\t{smi}\n")
            n += 1
            if a.limit and n >= a.limit: break
    print(f"wrote {n} yaml inputs, skipped_len={skip_len}, index={idx_path}")

if __name__ == "__main__":
    main()
