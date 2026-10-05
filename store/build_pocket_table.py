"""Consolidate the three-detector pocket ensemble into the deposited Parquet.

Input is the merged ensemble produced by the pocket-detection pipeline:
pocket_features.npz holding a (n_pockets, 21) feature matrix, per-protein
offsets and centroids, and meta.json holding the accession list in the same
order as those offsets. The alignment between the two is the load-bearing
assumption and is asserted rather than trusted.

Residue membership was not retained by the step that merged fpocket, P2Rank and
VN-EGNN, so centroids and features ship and membership does not.

Usage: build_pocket_table.py --npz PATH --meta PATH --out PATH
"""
import argparse, json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    feat, off, cent = d["pocket_features"], d["pocket_offsets"], d["pocket_centroids"]
    names = [str(x) for x in d["feature_names"]]
    acc = json.load(open(a.meta))["protein_ids"]
    assert len(off) == len(acc) + 1, f"offsets {len(off)} vs accessions {len(acc)}"
    assert off[-1] == feat.shape[0], f"offset tail {off[-1]} vs rows {feat.shape[0]}"
    assert cent.shape[0] == feat.shape[0], "centroid rows do not match feature rows"
    rows_acc, rows_rank = [], []
    for i, name in enumerate(acc):
        for r in range(int(off[i+1]) - int(off[i])):
            rows_acc.append(name); rows_rank.append(r)
    cols = {"protein_id": pa.array(rows_acc, pa.string()),
            "pocket_rank": pa.array(rows_rank, pa.int16())}
    for j, n in enumerate(names):
        cols[n] = pa.array(feat[:, j].astype(np.float32), pa.float32())
    t = pa.table(cols)
    pq.write_table(t, a.out, compression="snappy")
    print(f"{t.num_rows:,} pockets over {len(set(rows_acc)):,} accessions -> {a.out}")
    for n in ("source_fpocket", "source_p2rank", "source_vnegnn"):
        v = t.column(n).to_numpy()
        print(f"  {n:16s} set on {int((v > 0).sum()):,}")

if __name__ == "__main__":
    main()
