#!/usr/bin/env python3
"""Task 5 -- measure the train/test shift of every emitted split.

Four quantities per split, none of them an assertion:

``mmd2``            unbiased MMD^2 between the train and test *system-level*
                    ligand fingerprint distributions under the Tanimoto (Jaccard)
                    kernel, which is positive semi-definite on binary vectors, with
                    a permutation p-value.
``c2st_auc/_acc``   classifier two-sample test -- out-of-fold AUC and accuracy of an
                    L2 logistic regression trained to tell train ligands from test
                    ligands.  0.5 means the two folds are indistinguishable.
``max_protein_identity``  the largest sequence identity between any test protein
                    and any train protein, computed over the *shorter* of the two
                    sequences (fident * alnlen / min(qlen, tlen)).  Raw mmseqs
                    ``fident`` is reported separately as ``..._local`` and must not
                    be read as a leakage number: at low coverage mmseqs reports
                    7-residue perfect local hits as 100 percent identity.
``max_ligand_tanimoto``   the largest Morgan/Tanimoto similarity between any test
                    ligand and any train ligand.

The two maxima are the numbers a reviewer asks for; the per-test-entity *mean* of
the nearest-train-neighbour is reported alongside, because a single shared
cofactor can pin the maximum at 1.0 while the bulk of the test set is far away.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402
from build_splits import stable_seed  # noqa: E402

_G: dict = {}


# ------------------------------------------------------------- kernels ----
def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto/Jaccard kernel between two sets of unpacked binary fingerprints."""
    Af = A.astype(np.float32, copy=False)
    Bf = B.astype(np.float32, copy=False)
    inter = Af @ Bf.T
    na = Af.sum(1)[:, None]
    nb = Bf.sum(1)[None, :]
    union = na + nb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(union > 0, inter / union, 0.0)
    return k.astype(np.float32)


def mmd2_from_kernel(K: np.ndarray, m: int) -> float:
    """Unbiased MMD^2 from a pooled kernel matrix whose first m rows are sample X."""
    n = K.shape[0] - m
    Kxx = K[:m, :m]
    Kyy = K[m:, m:]
    Kxy = K[:m, m:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sxy = Kxy.mean()
    return float(sxx + syy - 2.0 * sxy)


def _mmd2_from_indicator(K: np.ndarray, KU: np.ndarray, K1: np.ndarray,
                         diag: np.ndarray, U: np.ndarray, m: int) -> np.ndarray:
    """Unbiased MMD^2 for many label assignments at once.

    ``U`` is (N, n_perm) with a 1 in row i of column p when sample i belongs to X
    under permutation p, and ``KU = K @ U``.  Every block sum is then a quadratic
    form, so a whole permutation null costs one matmul instead of n_perm
    submatrix copies.
    """
    n = K.shape[0] - m
    Sxx = (U * KU).sum(0)
    Sx1 = (U * K1[:, None]).sum(0)
    Sxy = Sx1 - Sxx
    Syy = K1.sum() - 2.0 * Sxy - Sxx
    Dx = (U * diag[:, None]).sum(0)
    Dy = diag.sum() - Dx
    return ((Sxx - Dx) / (m * (m - 1)) + (Syy - Dy) / (n * (n - 1))
            - 2.0 * Sxy / (m * n))


def mmd_permutation_test(K: np.ndarray, m: int, n_perm: int, rng) -> tuple:
    """Unbiased MMD^2 and its permutation p-value."""
    obs = mmd2_from_kernel(K, m)
    total = K.shape[0]
    K1 = K.sum(1)
    diag = np.diag(K).copy()
    U = np.zeros((total, n_perm), dtype=np.float32)
    for p in range(n_perm):
        U[rng.permutation(total)[:m], p] = 1.0
    null = _mmd2_from_indicator(K, K @ U, K1, diag, U, m)
    return obs, float((int((null >= obs).sum()) + 1) / (n_perm + 1))


def c2st(X_tr: np.ndarray, X_te: np.ndarray, rng, folds: int = 5) -> dict:
    """Classifier two-sample test on balanced samples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    k = min(len(X_tr), len(X_te))
    a = X_tr[rng.choice(len(X_tr), k, replace=False)]
    b = X_te[rng.choice(len(X_te), k, replace=False)]
    X = np.vstack([a, b]).astype(np.float32)
    y = np.concatenate([np.zeros(k, dtype=np.int8), np.ones(k, dtype=np.int8)])
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(rng.integers(2**31)))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    acc = float(((oof >= 0.5).astype(np.int8) == y).mean())
    auc = float(roc_auc_score(y, oof))
    # one-sided binomial p-value for accuracy > 0.5 under the null of no shift
    from scipy.stats import binomtest

    n_correct = int(((oof >= 0.5).astype(np.int8) == y).sum())
    p = float(binomtest(n_correct, len(y), 0.5, alternative="greater").pvalue)
    return {"c2st_auc": auc, "c2st_acc": acc, "c2st_p": p, "c2st_n_per_class": k}


# --------------------------------------------------------- diagnostics ----
def nearest_train_tanimoto(test_idx: np.ndarray, train_idx: np.ndarray,
                           topk_idx, topk_sim, packed) -> tuple:
    """Per-test-ligand max Tanimoto to any train ligand (exact)."""
    is_train = np.zeros(topk_idx.shape[0], dtype=bool)
    is_train[train_idx] = True
    cand = topk_idx[test_idx]
    ok = np.where(cand >= 0, is_train[np.clip(cand, 0, None)], False)
    first = np.argmax(ok, axis=1)
    found = ok[np.arange(len(test_idx)), first]
    best = np.where(found, topk_sim[test_idx][np.arange(len(test_idx)), first], -1.0)

    # Exact fallback for any test ligand whose entire top-K landed outside train.
    miss = np.flatnonzero(~found)
    n_fallback = int(miss.size)
    if n_fallback:
        pop = np.unpackbits(packed, axis=1).astype(np.float32)
        R = pop[train_idx]
        nb = R.sum(1)[None, :]
        for lo in range(0, miss.size, 64):
            sel = test_idx[miss[lo : lo + 64]]
            Q = pop[sel]
            inter = Q @ R.T
            union = Q.sum(1)[:, None] + nb - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, inter / union, 0.0)
            best[miss[lo : lo + 64]] = sim.max(1)
    return best, n_fallback


def measure_one(args_tuple) -> dict:
    row, opts = args_tuple
    audit = _G["audit"]
    ctx = audit[
        (audit.protein_threshold == row["protein_threshold"])
        & (audit.ligand_variant == row["ligand_variant"])
    ]
    fold = pd.read_parquet(row["path"])
    df = ctx.merge(fold, on="system_id", how="left", validate="one_to_one")
    if df.fold.isna().any():
        raise RuntimeError(f"{row['split_tag']}: fold file does not cover every system")

    lig_pos = _G["lig_pos"]
    prot_pos = _G["prot_pos"]
    bits = _G["bits"]
    ident = _G["identity"]

    tr = df[df.fold == "train"]
    te = df[df.fold == "test"]
    out = dict(row)
    if not len(te) or not len(tr):
        out.update({"measurable": False})
        return out
    out["measurable"] = True

    rng = np.random.default_rng(
        stable_seed(row["family"], row["protein_threshold"], row["ligand_variant"],
                    int(row["seed"])) ^ 0xA5A5A5
    )

    # ---- system-level fingerprint samples (a ligand recurs as its systems do)
    tr_rows = tr.inchikey.map(lig_pos).to_numpy()
    te_rows = te.inchikey.map(lig_pos).to_numpy()
    m = min(opts["n_sample"], len(tr_rows), len(te_rows))
    xs = bits[rng.choice(tr_rows, m, replace=False)]
    ys = bits[rng.choice(te_rows, m, replace=False)]
    K = tanimoto_kernel(np.vstack([xs, ys]), np.vstack([xs, ys]))
    mmd2, mmd_p = mmd_permutation_test(K, m, opts["n_perm"], rng)
    out["mmd2"] = mmd2
    out["mmd_p"] = mmd_p
    out["mmd_n_per_side"] = int(m)
    out.update(c2st(xs, ys, rng))

    # ---- leakage diagnostics on distinct entities
    tr_lig = np.unique(tr.inchikey.map(lig_pos).to_numpy())
    te_lig = np.unique(te.inchikey.map(lig_pos).to_numpy())
    nn, n_fb = nearest_train_tanimoto(te_lig, tr_lig, _G["topk_idx"], _G["topk_sim"],
                                      _G["packed"])
    out["max_ligand_tanimoto"] = float(nn.max())
    out["mean_ligand_nn_tanimoto"] = float(nn.mean())
    out["p95_ligand_nn_tanimoto"] = float(np.percentile(nn, 95))
    out["frac_test_ligands_nn_ge_0.4"] = float((nn >= 0.4).mean())
    out["n_ligand_nn_fallback"] = n_fb
    out["n_test_ligands"] = int(te_lig.size)
    out["n_train_ligands"] = int(tr_lig.size)

    tr_prot = np.unique(tr.accession.map(prot_pos).to_numpy())
    te_prot = np.unique(te.accession.map(prot_pos).to_numpy())
    for name, mat in ident.items():
        pnn = mat[np.ix_(te_prot, tr_prot)].max(1)
        sfx = "" if name == "global" else f"_{name}"
        out[f"max_protein_identity{sfx}"] = float(pnn.max())
        out[f"mean_protein_nn_identity{sfx}"] = float(pnn.mean())
        if name == "global":
            out["p95_protein_nn_identity"] = float(np.percentile(pnn, 95))
            out["frac_test_proteins_nn_ge_0.3"] = float((pnn >= 0.3).mean())
    out["n_test_proteins"] = int(te_prot.size)
    out["n_train_proteins"] = int(tr_prot.size)
    return out


def _init(audit, lig_pos, prot_pos, bits, packed, topk_idx, topk_sim, identity):
    _G.update(audit=audit, lig_pos=lig_pos, prot_pos=prot_pos, bits=bits,
              packed=packed, topk_idx=topk_idx, topk_sim=topk_sim, identity=identity)


def main(argv=None) -> int:
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=sf.ART / "split_index.parquet")
    ap.add_argument("--audit", type=Path, default=sf.ART / "split_audit.parquet")
    ap.add_argument("--n-sample", type=int, default=1500,
                    help="systems sampled per side for MMD and the C2ST")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N splits")
    ap.add_argument("--out", type=Path, default=sf.ART / "split_summary.parquet")
    ap.add_argument("--ligand-fp", type=Path, default=sf.ART / "ligand_fp.npz")
    ap.add_argument("--ligand-neighbors", type=Path, default=sf.ART / "ligand_neighbors.npz")
    ap.add_argument("--protein-identity", type=Path, default=sf.ART / "protein_identity.npz")
    args = ap.parse_args(argv)

    sf.ensure_dirs()
    idx = pd.read_parquet(args.index)
    if args.limit:
        idx = idx.head(args.limit)
    audit = pd.read_parquet(args.audit)

    zf = np.load(args.ligand_fp, allow_pickle=True)
    packed = zf["packed"]
    lig_keys = list(zf["inchikey"])
    lig_pos = {k: i for i, k in enumerate(lig_keys)}
    bits = np.unpackbits(packed, axis=1)

    zn = np.load(args.ligand_neighbors, allow_pickle=True)
    if list(zn["inchikey"]) != lig_keys:
        raise RuntimeError("ligand_fp.npz and ligand_neighbors.npz disagree on ligand order")
    topk_idx, topk_sim = zn["topk_idx"], zn["topk_sim"]

    zp = np.load(args.protein_identity, allow_pickle=True)
    # ``global`` (identity over the shorter sequence) is the headline definition;
    # ``local`` is raw mmseqs fident and is dominated by short spurious local hits.
    identity = {k: zp[f"identity_{k}"] for k in ("global", "cov80", "local")}
    prot_pos = {a: i for i, a in enumerate(list(zp["accession"]))}

    opts = {"n_sample": args.n_sample, "n_perm": args.n_perm}
    jobs = [(r, opts) for r in idx.to_dict("records")]
    print(f"[shift] measuring {len(jobs)} splits on {args.workers} workers", flush=True)

    results = []
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(audit, lig_pos, prot_pos, bits, packed, topk_idx, topk_sim,
                           identity)) as pool:
        for i, r in enumerate(pool.imap_unordered(measure_one, jobs, chunksize=1), 1):
            results.append(r)
            if i % 25 == 0:
                print(f"[shift]   {i}/{len(jobs)}", flush=True)

    out = pd.DataFrame(results).sort_values(
        ["family", "protein_threshold", "ligand_variant", "seed"]
    )
    out.to_parquet(args.out, index=False)
    print(f"[shift] wrote {args.out} ({out.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
