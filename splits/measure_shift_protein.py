#!/usr/bin/env python
"""Protein-axis two-sample test for the split families.

measure_shift.py computes MMD and a classifier two-sample test on ligand
fingerprints only, which is why it ranks the families oppositely on the two
tiers: it separates ligand-cold families sharply and protein-cold families
weakly. This adds the complementary measurement on the protein axis, using
mean-pooled ESM-2 embeddings of the sequence each system's receptor was built
from.

A length-only baseline is computed alongside. Mean-pooled protein language model
embeddings encode length and amino-acid composition heavily, so a classifier can
separate two folds on size rather than on homology. Where the embedding AUC
barely exceeds the length AUC, the protein diagnostic is measuring size and must
be read as such.

Writes new columns into a copy of the summary table; the input is not modified.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def mmd2_rbf(X, Y, gamma):
    """Unbiased squared MMD with an RBF kernel."""
    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())

def c2st(X, Y, rng, folds=5):
    """Out-of-fold AUC of a classifier separating X from Y."""
    Z = np.vstack([X, Y])
    y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    if min(len(X), len(Y)) < folds * 2:
        return np.nan, np.nan
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True,
                          random_state=int(rng.integers(2**31)))
    for tr, te in skf.split(Z, y):
        s = StandardScaler().fit(Z[tr])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(s.transform(Z[tr]), y[tr])
        p[te] = m.predict_proba(s.transform(Z[te]))[:, 1]
    return float(roc_auc_score(y, p)), float(((p > 0.5) == y).mean())

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", required=True, help="consolidated fold table")
    ap.add_argument("--entities", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--summary", required=True, help="input summary table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sample", type=int, default=800)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    z = np.load(a.embeddings, allow_pickle=True)
    acc = list(z["accession"]); E = z["emb"].astype(np.float64)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    pos = {p: i for i, p in enumerate(acc)}
    ent = pd.read_parquet(a.entities, columns=["system_id", "accession"])
    s2a = dict(zip(ent.system_id, ent.accession))
    # length proxy from the embedding norm is meaningless after normalisation, so
    # the baseline uses the first principal component of the unnormalised matrix,
    # which on mean-pooled PLM embeddings is dominated by length.
    raw = z["emb"].astype(np.float64)
    L = (raw - raw.mean(0)) @ np.linalg.svd(raw - raw.mean(0),
                                            full_matrices=False)[2][0]
    L = L.reshape(-1, 1)

    folds = pd.read_parquet(a.folds, columns=["split_tag", "system_id", "fold"])
    by_tag = {k: v for k, v in folds.groupby("split_tag")}
    summ = pd.read_parquet(a.summary)
    tags = list(summ.split_tag)
    if a.limit:
        tags = tags[:a.limit]
    rng = np.random.default_rng(0)
    rows = []
    for i, tag in enumerate(tags, 1):
        g = by_tag[tag]
        tr = {s2a.get(s) for s in g.loc[g.fold == "train", "system_id"]}
        te = {s2a.get(s) for s in g.loc[g.fold == "test", "system_id"]}
        tr = sorted(x for x in tr - te if x in pos)
        te = sorted(x for x in te if x in pos)
        r = {"split_tag": tag, "n_train_proteins_emb": len(tr),
             "n_test_proteins_emb": len(te)}
        if len(tr) < 20 or len(te) < 20:
            r.update({k: np.nan for k in
                      ("c2st_auc_protein", "c2st_acc_protein",
                       "c2st_auc_protein_length")})
            r["c2st_n_per_class_protein"] = min(len(tr), len(te))
            rows.append(r); continue
        n = min(a.n_sample, len(tr), len(te))
        ti = rng.choice(len(tr), n, replace=False)
        ei = rng.choice(len(te), n, replace=False)
        Xi = np.array([pos[tr[j]] for j in ti])
        Yi = np.array([pos[te[j]] for j in ei])
        X, Y = E[Xi], E[Yi]
        auc, acc_ = c2st(X, Y, rng)
        lauc, _ = c2st(L[Xi], L[Yi], rng)
        # The permutation MMD is dropped: on the ligand axis its p-values sit at
        # the permutation floor for every family except C1, so it adds nothing
        # the classifier AUC does not already carry, and it dominates runtime.
        r.update({"c2st_auc_protein": auc, "c2st_acc_protein": acc_,
                  "c2st_auc_protein_length": lauc,
                  "c2st_n_per_class_protein": n})
        rows.append(r)
        if i % 25 == 0:
            print(f"  {i}/{len(tags)}", flush=True)
    add = pd.DataFrame(rows)
    out = summ.merge(add, on="split_tag", how="left")
    out.to_parquet(a.out, index=False, compression="zstd")
    print(f"wrote {a.out}: {len(out)} rows, {out.shape[1]} columns")
    ok = out.dropna(subset=["c2st_auc_protein"])
    if len(ok):
        g = ok.groupby("family")[["c2st_auc_protein", "c2st_auc_protein_length",
                                  "c2st_auc"]].mean()
        g["excess_over_length"] = g.c2st_auc_protein - g.c2st_auc_protein_length
        print(g.round(4).to_string())
    return 0

if __name__ == "__main__":
    sys.exit(main())
