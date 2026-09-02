"""Build the temporal split family (T1) on the crystal tier.

Writes NEW files only:
  splits/folds_temporal/T1__*.parquet            {system_id, fold}  -- same schema as splits/folds/
  splits/artifacts/split_index_temporal.parquet  one row per configuration
Nothing under splits/folds/ or splits/artifacts/split_index.parquet is touched.

Temporal folds are strictly ordered: train < val_start <= val < test_start <= test.
The validation fold sits between train and test in time, so model selection never
sees anything later than the test boundary.
"""
import json
import pandas as pd

BASE = "/workspace"
FOLD_DIR = f"{BASE}/splits/folds_temporal"
import os
os.makedirs(FOLD_DIR, exist_ok=True)

tier = pd.read_parquet(f"{BASE}/reports/temporal/crystal_tier_dated.parquet")
N = len(tier)

CONFIGS = [
    # tag,                       date column,            val_start,    test_start,   note
    ("T1__release__q80q90", "initial_release_date", None, None,
     "headline: 80/10/10 by release-date quantile"),
    ("T1__deposit__q80q90", "deposit_date", None, None,
     "80/10/10 by deposition-date quantile"),
    ("T1__release__boltz2cut", "initial_release_date", "2022-06-30", "2023-06-01",
     "test == exactly the systems released after the Boltz-2 training cutoff"),
    ("T1__deposit__boltz2cut", "deposit_date", "2022-06-30", "2023-06-01",
     "test == systems deposited after the Boltz-2 cutoff date"),
    ("T1__release__chai1cut", "initial_release_date", "2020-01-12", "2021-01-12",
     "test == exactly the systems released after the Chai-1 training cutoff"),
    ("T1__deposit__chai1cut", "deposit_date", "2020-01-12", "2021-01-12",
     "test == systems deposited after the Chai-1 cutoff date"),
]

rows = []
for tag, col, vs, ts in [(c[0], c[1], c[2], c[3]) for c in CONFIGS]:
    note = dict((c[0], c[4]) for c in CONFIGS)[tag]
    if vs is None:                                  # quantile-driven 80/10/10
        vs = tier[col].quantile(0.80)
        ts = tier[col].quantile(0.90)
    vs, ts = pd.Timestamp(vs), pd.Timestamp(ts)
    assert vs <= ts, tag

    fold = pd.Series("train", index=tier.index)
    fold[(tier[col] >= vs) & (tier[col] < ts)] = "val"
    fold[tier[col] >= ts] = "test"

    df = pd.DataFrame({"system_id": tier.system_id.values, "fold": fold.values})
    assert len(df) == N and df.system_id.is_unique
    df.to_parquet(f"{FOLD_DIR}/{tag}.parquet", index=False)

    tr = tier[fold == "train"]
    te = tier[fold == "test"]
    va = tier[fold == "val"]
    rows.append({
        "split_tag": tag, "family": "T1", "family_desc": "temporal: train before a date, test after",
        "tier": "crystal", "date_basis": col.replace("initial_release_date", "release")
                                            .replace("deposit_date", "deposit"),
        "val_start": str(vs.date()), "test_start": str(ts.date()), "note": note,
        "path": f"splits/folds_temporal/{tag}.parquet",
        "n_systems": N, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_excluded": 0,
        "train_frac": len(tr) / N, "val_frac": len(va) / N, "test_frac": len(te) / N,
        "train_date_min": str(tr[col].min().date()), "train_date_max": str(tr[col].max().date()),
        "test_date_min": str(te[col].min().date()) if len(te) else None,
        "test_date_max": str(te[col].max().date()) if len(te) else None,
        "n_train_proteins": tr.protein_id.nunique(), "n_test_proteins": te.protein_id.nunique(),
        "n_train_pdb": tr.pdb_id.nunique(), "n_test_pdb": te.pdb_id.nunique(),
        # leakage diagnostics -- a pure temporal split controls neither entity axis
        "test_frac_protein_in_train": float(te.protein_id.isin(set(tr.protein_id)).mean()),
        "test_frac_ligand_ccd_in_train": float(te.ligand_ccd.isin(set(tr.ligand_ccd)).mean()),
        "test_frac_pdb_in_train": float(te.pdb_id.isin(set(tr.pdb_id)).mean()),
    })

idx = pd.DataFrame(rows)
idx.to_parquet(f"{BASE}/splits/artifacts/split_index_temporal.parquet", index=False)
idx.to_csv(f"{BASE}/splits/artifacts/split_index_temporal.csv", index=False)
print(idx[["split_tag", "date_basis", "val_start", "test_start",
           "n_train", "n_val", "n_test", "test_frac",
           "test_frac_protein_in_train", "test_frac_ligand_ccd_in_train"]].to_string(index=False))
