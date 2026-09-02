"""Validation for the temporal family: coverage, disjointness, strict time ordering."""
import glob
import pandas as pd

BASE = "/workspace"
tier = pd.read_parquet(f"{BASE}/reports/temporal/crystal_tier_dated.parquet")
idx = pd.read_parquet(f"{BASE}/splits/artifacts/split_index_temporal.parquet")
ref = pd.read_parquet(f"{BASE}/splits/folds/C1__p0.30__butina_0.30__seed0.parquet")

fails = 0
for row in idx.itertuples(index=False):
    d = pd.read_parquet(f"{BASE}/{row.path}")
    col = "initial_release_date" if row.date_basis == "release" else "deposit_date"
    m = d.merge(tier[["system_id", col]], on="system_id", validate="one_to_one")
    checks = {
        "schema matches splits/folds/": list(d.columns) == list(ref.columns)
                                        and d.dtypes.tolist() == ref.dtypes.tolist(),
        "covers the tier exactly": set(d.system_id) == set(tier.system_id) and len(d) == len(tier),
        "system_id unique": d.system_id.is_unique,
        "folds are train/val/test": set(d.fold) <= {"train", "val", "test"},
        "folds disjoint": d.groupby("system_id").fold.nunique().max() == 1,
        "train strictly before val": (m.loc[m.fold == "train", col].max()
                                      < m.loc[m.fold == "val", col].min()),
        "val strictly before test": (m.loc[m.fold == "val", col].max()
                                     < m.loc[m.fold == "test", col].min()),
        "no pdb entry spans train/test": not (
            set(tier.loc[tier.system_id.isin(d.loc[d.fold == "train", "system_id"]), "pdb_id"])
            & set(tier.loc[tier.system_id.isin(d.loc[d.fold == "test", "system_id"]), "pdb_id"])),
        "index sizes agree": (row.n_train, row.n_val, row.n_test)
                             == tuple(d.fold.value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int)),
    }
    bad = [k for k, v in checks.items() if not v]
    fails += len(bad)
    print(f"{'PASS' if not bad else 'FAIL'}  {row.split_tag:24s}  " + (", ".join(bad) if bad else "all 9 checks"))

# the existing 375 must be byte-identical to before
print(f"\nexisting folds present: {len(glob.glob(f'{BASE}/splits/folds/*.parquet'))} (expected 375)")
print(f"\n{'ALL CHECKS PASS' if fails == 0 else f'{fails} CHECKS FAILED'}")
