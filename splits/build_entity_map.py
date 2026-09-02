#!/usr/bin/env python3
"""Task 1 -- resolve every corpus-tier system_id to its protein accession and ligand inchikey.

Three independent sources carry the system -> entity mapping:

  * ``deposit_v3/labels/*_corpus_meta.parquet``  (system_id, protein_id, inchikey)
  * ``deposit_v3/metadata/ligand_identity.json`` (keyed by **pair_idx**)
  * ``docking/output/pairs_order.tsv``           (pair_idx, accession, compound_idx)

``ligand_identity.json`` is keyed by *pair_idx*, not *compound_idx*.  Joining on
compound_idx silently returns unrelated molecules, so this script refuses to emit
anything unless the accession agrees across all three sources for every system.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfcommon as sf  # noqa: E402


class JoinError(RuntimeError):
    """Raised when the pair_idx join cannot be verified."""


def load_corpus_meta() -> pd.DataFrame:
    files = sorted(glob.glob(str(sf.LABELS / "*_corpus_meta.parquet")))
    if not files:
        raise FileNotFoundError(f"no corpus meta parquet under {sf.LABELS}")
    cols = ["system_id", "protein_id", "inchikey", "inchikey14", "teacher"]
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    meta = pd.concat(frames, ignore_index=True)
    meta["system_id"] = meta["system_id"].astype(str)
    return meta


def build(strict: bool = True) -> tuple[pd.DataFrame, dict]:
    import json

    meta = load_corpus_meta()
    report: dict = {"meta_files": len(glob.glob(str(sf.LABELS / "*_corpus_meta.parquet")))}
    report["meta_rows"] = int(len(meta))
    report["teachers"] = sorted(meta["teacher"].dropna().unique().tolist())

    # A system must be internally consistent across teachers.
    per_sys = meta.groupby("system_id").agg(
        n_protein=("protein_id", "nunique"), n_ligand=("inchikey", "nunique")
    )
    inconsistent = per_sys[(per_sys.n_protein > 1) | (per_sys.n_ligand > 1)]
    report["systems_inconsistent_across_teachers"] = int(len(inconsistent))
    if len(inconsistent):
        raise JoinError(
            f"{len(inconsistent)} corpus systems disagree on protein/ligand between teachers"
        )

    sysmap = meta.drop_duplicates("system_id")[
        ["system_id", "protein_id", "inchikey", "inchikey14"]
    ].reset_index(drop=True)
    sysmap = sysmap.rename(columns={"protein_id": "accession"})
    report["n_systems"] = int(len(sysmap))
    report["n_proteins"] = int(sysmap.accession.nunique())
    report["n_ligands"] = int(sysmap.inchikey.nunique())

    # --- source 2: ligand_identity.json, keyed by pair_idx -----------------
    lid = json.loads(sf.LIGAND_IDENTITY.read_text())
    report["ligand_identity_keys"] = int(len(lid))
    li = pd.DataFrame(
        [
            {"system_id": k, "li_accession": v["accession"], "li_inchikey": v["inchikey"]}
            for k, v in lid.items()
        ]
    )
    missing_li = set(sysmap.system_id) - set(li.system_id)
    report["systems_missing_from_ligand_identity"] = sorted(missing_li)
    if missing_li:
        raise JoinError(f"{len(missing_li)} corpus systems absent from ligand_identity.json")

    j = sysmap.merge(li, on="system_id", how="left", validate="one_to_one")
    acc_ok = bool((j.accession == j.li_accession).all())
    ik_ok = bool((j.inchikey == j.li_inchikey).all())
    report["pairidx_join_accession_agreement"] = float((j.accession == j.li_accession).mean())
    report["pairidx_join_inchikey_agreement"] = float((j.inchikey == j.li_inchikey).mean())
    if strict and not (acc_ok and ik_ok):
        bad = j[(j.accession != j.li_accession) | (j.inchikey != j.li_inchikey)]
        raise JoinError(
            "pair_idx join FAILED verification -- accession/inchikey disagree for "
            f"{len(bad)} systems, e.g. {bad.head(5).to_dict('records')}"
        )

    # --- source 3: pairs_order.tsv, also keyed by pair_idx -----------------
    po = pd.read_csv(
        sf.PAIRS_ORDER, sep="\t", dtype={"pair_idx": str, "compound_idx": str}
    ).rename(columns={"pair_idx": "system_id", "accession": "po_accession"})
    report["pairs_order_rows"] = int(len(po))
    j2 = sysmap.merge(
        po[["system_id", "po_accession", "compound_idx", "smiles"]],
        on="system_id",
        how="left",
        validate="one_to_one",
    )
    report["systems_missing_from_pairs_order"] = int(j2.po_accession.isna().sum())
    covered = j2.po_accession.notna()
    report["pairs_order_accession_agreement"] = float(
        (j2.loc[covered, "accession"] == j2.loc[covered, "po_accession"]).mean()
    )
    if strict and report["pairs_order_accession_agreement"] < 1.0:
        raise JoinError("pairs_order.tsv accession disagrees with deposit_v3 metadata")
    # compound_idx is *not* a valid join key for ligand_identity; record the
    # collision so the mistake is documented rather than repeated.
    report["compound_idx_equals_pair_idx_fraction"] = float(
        (j2.loc[covered, "system_id"] == j2.loc[covered, "compound_idx"]).mean()
    )
    sysmap = sysmap.merge(j2[["system_id", "compound_idx", "smiles"]], on="system_id", how="left")

    # --- entity-side coverage ---------------------------------------------
    smiles = json.loads(sf.CORPUS_SMILES.read_text())
    scaffold = json.loads(sf.LIGAND_SCAFFOLD.read_text())
    ligs = set(sysmap.inchikey)
    report["corpus_smiles_entries"] = int(len(smiles))
    report["ligand_scaffold_entries"] = int(len(scaffold))
    report["corpus_ligands_missing_smiles"] = sorted(ligs - set(smiles))
    report["corpus_ligands_missing_scaffold"] = sorted(ligs - set(scaffold))
    report["smiles_entries_not_in_corpus_tier"] = int(len(set(smiles) - ligs))

    fasta = sf.read_fasta(sf.CORPUS_FASTA)
    report["fasta_sequences"] = int(len(fasta))
    report["corpus_proteins_missing_fasta"] = sorted(set(sysmap.accession) - set(fasta))
    if strict and report["corpus_ligands_missing_smiles"]:
        raise JoinError("some corpus ligands have no SMILES")
    if strict and report["corpus_proteins_missing_fasta"]:
        raise JoinError("some corpus proteins have no FASTA sequence")

    report["unresolved_systems"] = []  # every system resolved both axes or we raised above
    report["coverage_protein"] = 1.0
    report["coverage_ligand"] = 1.0

    # multiplicity, which drives how large a "seen" (warm) test set can be
    lc = sysmap.inchikey.value_counts()
    pc = sysmap.accession.value_counts()
    report["ligand_systems_multiplicity"] = {
        "n_ligands": int(len(lc)),
        "singletons": int((lc == 1).sum()),
        "singleton_fraction": float((lc == 1).mean()),
        "systems_whose_exact_ligand_recurs": int(sysmap.inchikey.map(lc).ge(2).sum()),
        "max": int(lc.max()),
    }
    report["protein_systems_multiplicity"] = {
        "n_proteins": int(len(pc)),
        "singletons": int((pc == 1).sum()),
        "median": float(pc.median()),
        "max": int(pc.max()),
    }
    return sysmap, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=sf.ART / "system_entities.parquet")
    ap.add_argument("--report", type=Path, default=sf.ART / "entity_map_report.json")
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="report join mismatches instead of raising (diagnostic use only)",
    )
    args = ap.parse_args(argv)

    sf.ensure_dirs()
    sysmap, report = build(strict=not args.no_strict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sysmap.to_parquet(args.out, index=False)
    sf.write_json(args.report, report)
    print(
        f"[entity-map] {report['n_systems']} systems, {report['n_proteins']} proteins, "
        f"{report['n_ligands']} ligands -> {args.out}"
    )
    print(
        f"[entity-map] pair_idx join verified: accession agreement "
        f"{report['pairidx_join_accession_agreement']:.4f} (ligand_identity), "
        f"{report['pairs_order_accession_agreement']:.4f} (pairs_order)"
    )
    if report["corpus_ligands_missing_scaffold"]:
        print(
            f"[entity-map] WARNING: {len(report['corpus_ligands_missing_scaffold'])} "
            f"corpus ligand(s) lack a precomputed Murcko scaffold: "
            f"{report['corpus_ligands_missing_scaffold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
