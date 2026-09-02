#!/usr/bin/env python3
"""Protein clustering at 30/40/50 percent identity plus an all-vs-all identity matrix.

The 30 and 40 percent clusterings already exist in the workspace; this script
re-uses them verbatim and generates the missing 50 percent clustering with the
*same* mmseqs invocation that produced them (recovered from
``reports/mmseqs40.log``)::

    mmseqs easy-cluster corpus_proteins.fasta protclust<N> tmp<N> \
        --min-seq-id 0.<N> -c 0.8 --cluster-mode 2

It also runs an all-vs-all ``easy-search`` so that "maximum sequence identity
between any test protein and any train protein" is a measured number rather than
an assertion.  Pairs with no reportable alignment are recorded as identity 0,
which is a *floor*, not a claim of zero homology -- the detection limit is stated
in the report.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402


def run_mmseqs_cluster(threshold: str, workdir: Path, threads: int,
                       fasta: Path = None) -> Path:
    """Cluster corpus_proteins.fasta at ``threshold`` and return the *_cluster.tsv path."""
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = workdir / f"protclust_{threshold}"
    tsv = Path(str(prefix) + "_cluster.tsv")
    if tsv.exists():
        print(f"[prot] reusing existing {tsv}")
        return tsv
    tmp = workdir / f"tmp_{threshold}"
    cmd = [
        str(sf.MMSEQS), "easy-cluster", str(fasta or sf.CORPUS_FASTA),
        str(prefix), str(tmp),
        "--min-seq-id", threshold, "-c", "0.8", "--cluster-mode", "2",
        "--threads", str(threads),
    ]
    print("[prot] " + " ".join(cmd), flush=True)
    log = sf.SPLITS / "logs" / f"mmseqs{threshold}.log"
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    shutil.rmtree(tmp, ignore_errors=True)
    if not tsv.exists():
        raise RuntimeError(f"mmseqs produced no {tsv}; see {log}")
    return tsv


def load_cluster_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["cluster_rep", "accession"])
    return df


def run_all_vs_all(workdir: Path, threads: int, sensitivity: float, max_seqs: int,
                   fasta: Path = None) -> Path:
    out = workdir / "allvsall.tsv"
    if out.exists():
        print(f"[prot] reusing existing {out}")
        return out
    tmp = workdir / "tmp_search"
    cmd = [
        str(sf.MMSEQS), "easy-search", str(fasta or sf.CORPUS_FASTA),
        str(fasta or sf.CORPUS_FASTA),
        str(out), str(tmp),
        "-s", str(sensitivity), "-e", "10000", "--max-seqs", str(max_seqs),
        "--min-seq-id", "0", "-c", "0.0", "--cov-mode", "0", "--seq-id-mode", "0",
        "--threads", str(threads), "-a", "1",
        "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
    ]
    print("[prot] " + " ".join(cmd), flush=True)
    log = sf.SPLITS / "logs" / "mmseqs_allvsall.log"
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", nargs="*", default=list(sf.PROTEIN_THRESHOLDS))
    ap.add_argument("--workdir", type=Path, default=sf.ART / "protein")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--max-seqs", type=int, default=1000)
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--out", type=Path, default=sf.ART / "protein_clusters.parquet")
    ap.add_argument("--identity-out", type=Path, default=sf.ART / "protein_identity.npz")
    ap.add_argument("--report", type=Path, default=sf.ART / "protein_cluster_report.json")
    ap.add_argument("--fasta", type=Path, default=sf.CORPUS_FASTA)
    ap.add_argument("--no-precomputed", action="store_true",
                    help="ignore sf.PRECOMPUTED_PROTCLUST; cluster from --fasta")
    args = ap.parse_args(argv)

    sf.ensure_dirs()
    args.workdir.mkdir(parents=True, exist_ok=True)
    fasta = sf.read_fasta(args.fasta)
    accessions = sorted(fasta)
    report = {"n_sequences": len(accessions), "mmseqs_version": None, "clusterings": {}}
    report["mmseqs_version"] = subprocess.run(
        [str(sf.MMSEQS), "version"], capture_output=True, text=True
    ).stdout.strip()

    frames = []
    for thr in args.thresholds:
        pre = None if args.no_precomputed else sf.PRECOMPUTED_PROTCLUST.get(thr)
        if pre and pre.exists():
            src, tsv = "precomputed", pre
        else:
            src, tsv = "generated", run_mmseqs_cluster(thr, args.workdir, args.threads,
                                                       fasta=args.fasta)
        df = load_cluster_tsv(tsv)
        missing = set(accessions) - set(df.accession)
        if missing:
            raise RuntimeError(f"clustering {thr} misses {len(missing)} accessions")
        sizes = df.cluster_rep.value_counts()
        report["clusterings"][thr] = {
            "source": src,
            "path": str(tsv),
            "n_members": int(df.accession.nunique()),
            "n_clusters": int(df.cluster_rep.nunique()),
            "singletons": int((sizes == 1).sum()),
            "singleton_fraction": float((sizes == 1).mean()),
            "largest_cluster": int(sizes.max()),
            "top10_share_of_proteins": float(sizes.nlargest(10).sum() / sizes.sum()),
        }
        df = df.assign(protein_threshold=thr, protein_cluster=thr + ":" + df.cluster_rep)
        frames.append(df[["accession", "protein_threshold", "protein_cluster"]])
        print(f"[prot] {thr}: {report['clusterings'][thr]['n_clusters']} clusters ({src})")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"[prot] wrote {args.out}")

    if not args.skip_search:
        t0 = time.time()
        tsv = run_all_vs_all(args.workdir, args.threads, args.sensitivity, args.max_seqs,
                             fasta=args.fasta)
        hits = pd.read_csv(
            tsv, sep="\t", header=None,
            names=["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"],
        )
        idx = {a: i for i, a in enumerate(accessions)}
        n = len(accessions)
        lengths = np.array([len(fasta[a]) for a in accessions], dtype=np.float64)
        qi = hits["query"].map(idx).to_numpy()
        ti = hits["target"].map(idx).to_numpy()
        fid = hits["fident"].to_numpy(dtype=np.float64)
        aln = hits["alnlen"].to_numpy(dtype=np.float64)
        mincov = np.minimum(hits["qcov"].to_numpy(), hits["tcov"].to_numpy())

        # Three identity definitions.  ``fident`` alone is NOT a usable leakage
        # metric at -c 0.0: mmseqs happily reports 7-residue perfect local hits at
        # 0.5% coverage, which would show as "100% identity" between two entirely
        # unrelated proteins.  The headline metric is therefore identity computed
        # over the shorter of the two sequences.
        defs = {
            # identity over the shorter sequence -- the usual split-leakage definition
            "global": fid * aln / np.minimum(lengths[qi], lengths[ti]),
            # identity restricted to alignments meeting the clustering's own
            # coverage criterion (-c 0.8 --cov-mode 0), so it is directly
            # comparable to the mmseqs clusters used to build the folds
            "cov80": np.where(mincov >= 0.8, fid, 0.0),
            # raw local identity, kept only for transparency
            "local": fid,
        }
        mats = {}
        for name, vals in defs.items():
            m = np.zeros((n, n), dtype=np.float32)
            np.maximum.at(m, (qi, ti), vals.astype(np.float32))
            m = np.maximum(m, m.T)  # symmetrise: mmseqs reports per direction
            np.fill_diagonal(m, 1.0)
            mats[name] = m
        np.savez_compressed(
            args.identity_out,
            identity=mats["global"],
            identity_global=mats["global"],
            identity_cov80=mats["cov80"],
            identity_local=mats["local"],
            accession=np.array(accessions, dtype=object),
        )
        eye = np.eye(n, dtype=bool)
        report["all_vs_all"] = {
            "hit_rows": int(len(hits)),
            "seconds": round(time.time() - t0, 1),
            "sensitivity": args.sensitivity,
            "max_seqs": args.max_seqs,
            "pairs_total": int(n * (n - 1)),
            "pairs_with_alignment": int((mats["local"][~eye] > 0).sum()),
            "pair_detection_rate": float((mats["local"][~eye] > 0).mean()),
            "note": "pairs with no reportable alignment are stored as identity 0.0 "
                    "(a detection floor, not measured zero homology)",
            "headline_definition": "global = fident * alnlen / min(qlen, tlen)",
            "definitions": {
                name: {
                    "max_offdiagonal": float(m[~eye].max()),
                    "mean_offdiagonal": float(m[~eye].mean()),
                    "pairs_ge_0.30": int((m[~eye] >= 0.30).sum()),
                    "pairs_ge_0.90": int((m[~eye] >= 0.90).sum()),
                }
                for name, m in mats.items()
            },
        }
        for name, st in report["all_vs_all"]["definitions"].items():
            print(f"[prot] identity[{name}]: max off-diagonal {st['max_offdiagonal']:.3f}, "
                  f"pairs >=0.30: {st['pairs_ge_0.30']}, >=0.90: {st['pairs_ge_0.90']}")
        print(f"[prot] identity matrices {mats['global'].shape} -> {args.identity_out}")

    sf.write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
