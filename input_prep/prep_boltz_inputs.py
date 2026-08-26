"""Write Boltz-2 inputs from pairs_order.tsv, one YAML per pair, in the given
(shuffled, smina-overlapping) order. Single-sequence MSA for speed and offline
use. Proteins longer than --max-len are skipped to stay within 24 GB on the 3090.
A meta TSV records rec_name to pair so the extractor can recover head ordering.
Run: python prep_boltz_inputs.py /workspace/docking/input \
        /workspace/docking/output/pairs_order.tsv \
        /workspace/docking/output/boltz_in --limit 4000 --max-len 750
"""
import argparse, csv, os

T = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
     "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
     "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def seq_from_pdb(pdb):
    s, seen = [], set()
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[16] in " A":
            k = (ln[21], ln[22:26].strip())
            if k in seen: continue
            seen.add(k); s.append(T.get(ln[17:20].strip(), "X"))
    return "".join(s)

YAML = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n"
        "      msa: empty\n  - ligand:\n      id: L\n      smiles: '{smi}'\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("pairs"); ap.add_argument("indir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=750)
    a = ap.parse_args()
    os.makedirs(a.indir, exist_ok=True)
    meta = open(os.path.join(a.indir, "_meta.tsv"), "w")
    meta.write("rec\tpair_idx\taccession\tcompound_idx\tsmiles\n")
    rows = list(csv.DictReader(open(a.pairs), delimiter="\t"))
    if a.limit: rows = rows[:a.limit]
    written = skipped = 0
    for i, r in enumerate(rows):
        acc = r["accession"]
        pdb = os.path.join(a.root, "pdbs", f"AF-{acc}-F1-model_v6.pdb")
        seq = seq_from_pdb(pdb)
        if not seq or len(seq) > a.max_len:
            skipped += 1; continue
        rec = f"{i:06d}_{r['pair_idx']}"
        with open(os.path.join(a.indir, rec + ".yaml"), "w") as w:
            w.write(YAML.format(seq=seq, smi=r["smiles"]))
        meta.write(f"{rec}\t{r['pair_idx']}\t{acc}\t{r['compound_idx']}\t{r['smiles']}\n")
        written += 1
    meta.close()
    print(f"written={written} skipped_long_or_empty={skipped} indir={a.indir}")

if __name__ == "__main__":
    main()
