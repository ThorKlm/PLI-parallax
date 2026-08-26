#!/usr/bin/env python
"""Symmetry-corrected RMSD for smina poses, with atom correspondence established
by graph matching against a properly bond-perceived reference.

Why this exists
---------------
``datasets/bisy_smina.py`` compares a smina pose against the crystal ligand using
``spyrmsd.symmrmsd`` with an adjacency matrix built from a 1.9 A distance cutoff on
both sides.  Two things go wrong with that:

  * the pose and the crystal residue are in *different atom orders* (a smina pose has
    round-tripped through PDBQT; the crystal residue is in deposition order).  On a
    35-system sample the element orders agree in 3 of 35, so anything that falls back
    to file order is comparing unrelated atoms;
  * a 1.9 A cutoff cannot express metal coordination, so every HEM/HEC system yields a
    graph spyrmsd cannot match (NaN, or dropped by the caller's exception handler --
    the shipped ``bisy_smina_*.npz`` contains 0 of 452 HEM and 0 of 217 HEC systems).

This script establishes the correspondence explicitly, before any RMSD is computed:

  1. the reference is the CCD-templated crystal ligand -- real bond orders, charges and
     aromaticity from the chemical component dictionary, geometry from the structure --
     read from ``/workspace/pb_inputs/<system_id>_ref.sdf`` where the cofolder extractor
     already wrote one, and otherwise rebuilt here through the same code path
     (``extract_crystal_ref_ligands.build_reference_ligand``);
  2. the pose is mapped onto that reference with ``atom_match.map_pose_to_ref``, i.e. by
     assigning bond orders from the reference template and taking the graph
     isomorphism, not by distance-cutoff adjacency;
  3. that mapping is widened to every symmetry-equivalent alternative -- enumerated on the
     element-and-connectivity graph, so resonance-equivalent atoms are interchangeable --
     and every candidate mapping is ranked by
     bond-length plausibility (the reference bond graph imposed on the pose
     coordinates, scored by ``extract_crystal_ref_ligands``'s limits) and only then by
     RMSD, and the choice is recorded per system, so a wrong-but-isomorphic mapping
     cannot pass silently;
  4. metal macrocycles are handled explicitly: if the coordinated graph will not match,
     coordination bonds are stripped from *both* sides for the match and reinstated for
     the plausibility ranking, so a mis-assigned pyrrole nitrogen stretches an Fe-N bond
     and is caught rather than accepted.

RMSD is reported in the crystal frame with no superposition -- the pose is docked into
the deposited receptor, so the frames already agree and superposing would measure
conformer quality rather than docking accuracy.  ``rmsd_superposed`` is written
alongside as a *diagnostic only*: a system with a unique mapping and a large in-place
RMSD but a small superposed RMSD is a correctly-corresponded pose in the wrong
orientation (1a0g_PMP_B: 5.83 A in place, 1.31 A superposed), not a matching failure.

Nothing here writes to ``experimental_expansion/bisy``; output goes to ``bisy_v2``.

Usage:
    python bisy_smina_v2.py --shard 0 --nshards 16       # one shard
    python bisy_smina_v2.py --limit 200 --tag val200     # validation sample
    python bisy_smina_v2.py --report --tag val200        # aggregate + statistics
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/datasets")

from atom_match import map_pose_to_ref  # noqa: E402
import extract_crystal_ref_ligands as X  # noqa: E402

RDLogger.DisableLog("rdApp.*")

POSE_DIR = "/workspace/datasets/experimental_expansion/smina_out/poses"
REF_DIR = "/workspace/pb_inputs"
STRUCT_DIR = "/workspace/datasets/experimental_expansion/structures"
CCD_SMILES = "/workspace/datasets/experimental_expansion/ccd_smiles.json"
OUT_DIR = "/workspace/datasets/experimental_expansion/bisy_v2"

# Enumeration cap for candidate mappings.  Symmetric ligands have many automorphisms --
# inositol tetrakisphosphate is the worst seen at 2592 -- so the cap sits well above that
# while still bounding a pathological query.  Hitting it is recorded per system, because a
# truncated candidate set means the ranking below saw only part of the symmetry group.
MAX_MATCHES = 20000

# A mapping is "plausible" when every reference bond it implies is within the same limits
# extract_crystal_ref_ligands uses to accept a coordinate transfer.
STRAIN_OK = 1.0

CSV_FIELDS = [
    "system_id", "pdb", "ccd", "chain", "status", "reason",
    "n_ref_heavy", "n_pose_heavy", "centroid_offset", "rmsd", "rmsd_superposed",
    "ref_source", "map_mode", "match_route", "metal_class",
    "n_candidates", "n_plausible", "candidates_truncated",
    "chosen_candidate", "chosen_strain", "rmsd_spread",
]


class Failure(Exception):
    """A per-system failure with a machine-readable stage tag."""

    def __init__(self, stage, detail=""):
        super().__init__(f"{stage}: {detail}" if detail else stage)
        self.stage = stage
        self.detail = detail


# --------------------------------------------------------------------------------------
# Molecule loading
# --------------------------------------------------------------------------------------


def strip_hydrogens(mol, stage):
    """Heavy-atom copy.  Poses carry smina's added Hs; references are already heavy-only."""
    try:
        out = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        out = Chem.Mol(mol)
    if any(a.GetAtomicNum() == 1 for a in out.GetAtoms()):
        # RemoveHs keeps Hs it considers structurally meaningful (isotopes, charged,
        # unbonded).  For a correspondence the heavy graph is what matters, so drop them.
        rw = Chem.RWMol(out)
        for idx in sorted((a.GetIdx() for a in out.GetAtoms() if a.GetAtomicNum() == 1),
                          reverse=True):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
    if out.GetNumAtoms() == 0:
        raise Failure(stage, "no heavy atoms")
    if out.GetNumConformers() == 0:
        raise Failure(stage, "no conformer")
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(out)
    except Exception:
        pass
    return out


def read_sdf(path, stage):
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    mol = next(iter(supplier), None)
    if mol is None:
        raise Failure(stage, f"unreadable SDF {os.path.basename(path)}")
    return mol


def sanitized(mol):
    """Best-effort sanitization; AssignBondOrdersFromTemplate needs ring info on the
    template, but CCD templates with metals routinely fail a strict valence check."""
    out = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        pass
    out = Chem.Mol(mol)
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES,
                         catchErrors=True)
    except Exception:
        pass
    return out


def load_reference(system_id, ccd_smiles):
    """(ref_mol, source).  The extractor's SDF is preferred; the CCD path is the fallback.

    Preferring the written SDF is not just a shortcut: it is the same molecule the
    cofolder pose-quality inputs are scored against, so smina numbers stay comparable to
    the teachers without re-deriving anything.
    """
    sdf = os.path.join(REF_DIR, f"{system_id}_ref.sdf")
    # A zero-byte or half-written SDF means the extractor is mid-run or failed on that
    # system; treat it as absent and rebuild rather than failing the system on it.
    if os.path.exists(sdf) and os.path.getsize(sdf) > 0:
        try:
            return strip_hydrogens(read_sdf(sdf, "reference"), "reference"), "pb_inputs"
        except (Failure, OSError, ValueError):
            pass
    return build_reference_from_ccd(system_id, ccd_smiles)


def build_reference_from_ccd(system_id, ccd_smiles):
    """Rebuild the reference through extract_crystal_ref_ligands' own code path.

    Only reached for the ~3% of poses the extractor never wrote an SDF for.  The ligand
    copy is chosen by the crystal-label centroid -- deliberately *not* by proximity to the
    pose, which would bias the RMSD it is about to be used to compute.
    """
    import gemmi
    from cryst_lig_helper import _load_centroids

    pdb, ccd, _ = parse_system_id(system_id)
    cif = os.path.join(STRUCT_DIR, f"{pdb}.cif")
    if not os.path.exists(cif):
        raise Failure("reference", f"no structure {pdb}.cif")
    smiles = ccd_smiles.get(ccd)
    if not smiles:
        raise Failure("reference", f"no CCD SMILES for {ccd}")

    centroid = _load_centroids().get(system_id)
    if centroid is None:
        raise Failure("reference", "no crystal-label centroid to disambiguate copies")

    model = gemmi.read_structure(cif)[0]
    best, best_d = None, float("inf")
    for chain in model:
        for res in chain:
            if res.name != ccd:
                continue
            xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in X.heavy_atoms(res)], float)
            if not len(xyz):
                continue
            d = float(np.linalg.norm(xyz.mean(0) - centroid))
            if d < best_d:
                best, best_d = (res, chain.name), d
    if best is None:
        raise Failure("reference", f"no {ccd} copy in {pdb}")

    res, chain_id = best
    try:
        template, _ = X.load_template(smiles, ccd)
        crystal = X.mol_from_pdb_block(X.ligand_pdb_block(res, chain_id))
        mol, _mode = X.build_reference_ligand(crystal, template)
    except X.SystemFailure as exc:
        raise Failure("reference", f"{exc.stage}: {exc.reason}")
    except Exception as exc:
        raise Failure("reference", f"{type(exc).__name__}: {exc}")
    return strip_hydrogens(mol, "reference"), "ccd_template"


# --------------------------------------------------------------------------------------
# Atom correspondence
# --------------------------------------------------------------------------------------


def has_metal(mol):
    return any(a.GetSymbol() in X.METALS for a in mol.GetAtoms())


def classify_metal(ref):
    """"" | "metal" | "metal_macrocycle" -- the last is the HEM/HEC class that the
    distance-cutoff adjacency in bisy_smina.py could not represent at all."""
    if not has_metal(ref):
        return ""
    ring_info = ref.GetRingInfo()
    for atom in ref.GetAtoms():
        if atom.GetSymbol() not in X.METALS:
            continue
        coordinated = [n for n in atom.GetNeighbors() if ring_info.NumAtomRings(n.GetIdx())]
        if len(coordinated) >= 3:
            return "metal_macrocycle"
    return "metal"


def bond_order_template(ref, pose):
    """The pose with bond orders taken from the reference -- atom_match's first step."""
    try:
        return AllChem.AssignBondOrdersFromTemplate(ref, pose)
    except Exception:
        return pose


def enumerate_candidates(ref, pose):
    """(matches, route).  ``match[pose_i] = ref_i`` for every graph isomorphism found.

    Candidates are enumerated on the *element-and-connectivity* graph, with bond orders,
    charges and aromaticity flattened away.  That is deliberate and it is the community
    convention (spyrmsd, PoseBusters): which oxygen of a carboxylate or phosphate carries
    the double bond is an arbitrary choice made at deposition, so resonance-equivalent
    atoms have to be interchangeable or the RMSD is inflated by a bookkeeping artefact.
    Enumerating on the bond-order-aware graph instead was measurably wrong -- on a
    250-system check it never found a better mapping than the flattened graph and missed
    one by up to 1.4 A on 134 of them.

    Bond orders still do work upstream: ``correspond`` runs ``atom_match.map_pose_to_ref``
    first, which assigns them from the reference template and so establishes that the pose
    really is the reference compound rather than something merely isomorphic to it.

    Routes are tried loosest-last and the one that hit is recorded per system.
    """
    fixed = bond_order_template(ref, pose)
    routes = [("topology", X.flatten(ref), X.flatten(fixed))]
    if has_metal(ref) or has_metal(pose):
        # Metal coordination is geometrically variable and perceived inconsistently on
        # the two sides; drop it for the match and let the ranking below reinstate it.
        bare_ref = X.strip_metal_bonds(ref)
        bare_pose = X.strip_metal_bonds(fixed)
        if bare_ref is not None or bare_pose is not None:
            routes.append((
                "metalfree",
                X.flatten(bare_ref if bare_ref is not None else ref),
                X.flatten(bare_pose if bare_pose is not None else fixed),
            ))
    for name, target, query in routes:
        try:
            matches = target.GetSubstructMatches(
                query, uniquify=False, useChirality=False, maxMatches=MAX_MATCHES)
        except Exception:
            continue
        if matches:
            return [list(m) for m in matches], name
    raise Failure("match", "no atom-graph isomorphism between pose and reference")


def reference_bonds(ref):
    """(begin, end, limit) arrays for the reference bond graph.

    ``limit`` is the length above which a bond is implausible -- extract_crystal_ref_ligands'
    MAX_BOND, or MAX_METAL_BOND where either end is a metal.
    """
    begin, end, limit = [], [], []
    for bond in ref.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        begin.append(a.GetIdx())
        end.append(b.GetIdx())
        limit.append(X.MAX_METAL_BOND
                     if (a.GetSymbol() in X.METALS or b.GetSymbol() in X.METALS)
                     else X.MAX_BOND)
    return (np.array(begin, np.int64), np.array(end, np.int64), np.array(limit, float))


def score_candidates(ref, ref_xyz, pose_xyz, matches):
    """(strain, rmsd) per candidate mapping, vectorised over the candidate set.

    ``strain[k]`` is the worst reference bond length implied by mapping ``k``, relative to
    its plausibility limit; <= 1.0 is clean.  This is extract_crystal_ref_ligands'
    geometry_strain check applied to a *mapping* rather than a coordinate transfer: a
    mapping that pairs the wrong atoms stretches bonds across the molecule.

    It only discriminates where the match was made on a reduced graph -- the metalfree
    route.  A genuine automorphism of the matched graph maps the bond set onto itself and
    every candidate scores identically, correctly so, since those candidates really are
    indistinguishable and the RMSD tie-break decides between them.
    """
    perms = np.asarray(matches, np.int64)                     # perms[k, pose_i] = ref_i
    rmsd = np.sqrt(((pose_xyz[None] - ref_xyz[perms]) ** 2).sum(-1).mean(-1))

    begin, end, limit = reference_bonds(ref)
    if not len(begin):
        return np.zeros(len(perms)), rmsd
    inverse = np.argsort(perms, axis=1)                       # inverse[k, ref_i] = pose_i
    in_ref_order = pose_xyz[inverse]                          # pose coords, reference order
    lengths = np.linalg.norm(in_ref_order[:, begin] - in_ref_order[:, end], axis=-1)
    return (lengths / limit).max(1), rmsd


def in_place_rmsd(ref_xyz, pose_xyz, perm):
    return float(np.sqrt(((pose_xyz - ref_xyz[perm]) ** 2).sum(1).mean()))


def superposed_rmsd(ref_xyz, pose_xyz, perm):
    """Kabsch RMSD of the corresponded atoms.  Diagnostic: separates a mis-corresponded
    pose (large) from a correctly-corresponded pose in the wrong place (small)."""
    P = pose_xyz - pose_xyz.mean(0)
    Q = ref_xyz[perm] - ref_xyz[perm].mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((R @ P.T).T - Q) ** 2).sum(1).mean()))


def correspond(ref, pose):
    """Establish the atom correspondence and score it.

    ``atom_match.map_pose_to_ref`` runs first -- it is the mapping the cofolder extractors
    rely on, and going through the reference template is what proves the pose is the same
    compound.  Its answer is then widened to the full symmetry-equivalent candidate set
    (see ``enumerate_candidates``) and the candidates are ranked, because a single
    arbitrary isomorphism is not the symmetry-corrected RMSD.
    """
    ref_xyz = ref.GetConformer().GetPositions()
    pose_xyz = pose.GetConformer().GetPositions()

    primary = None
    try:
        primary, _ambiguous = map_pose_to_ref(sanitized(ref), pose)
    except Exception:
        primary = None  # metal coordination, unsanitizable template: the routes below cope

    matches, route = enumerate_candidates(ref, pose)
    truncated = int(len(matches) >= MAX_MATCHES)

    strain, rmsd = score_candidates(ref, ref_xyz, pose_xyz, matches)
    plausible = np.flatnonzero(strain <= STRAIN_OK)
    if not len(plausible):
        # Nothing is clean.  Keep the least-strained candidates rather than dropping the
        # system, and say so in the reason column.
        plausible = np.flatnonzero(strain <= strain.min() + 1e-9)
    # Rank by bond-length plausibility first, RMSD second: among chemically
    # indistinguishable mappings the closest one is the symmetry-corrected answer, but a
    # mapping that strains the molecule never wins on a low RMSD alone.
    idx = int(min(plausible, key=lambda k: (strain[k], rmsd[k])))
    perm, worst = matches[idx], float(strain[idx])
    spread = float(rmsd[plausible].max() - rmsd[plausible].min())

    # The template-aware mapping must be one of the candidates -- the candidate graph is
    # strictly looser.  If it is not, the two disagree about the molecule itself and the
    # system is not scored rather than scored on the wrong correspondence.
    if primary is not None and route == "topology" and not any(m == primary for m in matches):
        raise Failure("match", "map_pose_to_ref result absent from the candidate set")

    return {
        "perm": perm,
        "map_mode": "unique" if len(matches) == 1 else "ranked",
        "match_route": route if primary is not None else f"{route}_no_template",
        "n_candidates": len(matches),
        "n_plausible": len(plausible),
        "candidates_truncated": truncated,
        "chosen_candidate": idx,
        "chosen_strain": worst,
        "rmsd_spread": spread,
    }


# --------------------------------------------------------------------------------------
# Per-system driver
# --------------------------------------------------------------------------------------


def parse_system_id(system_id):
    parts = system_id.split("_")
    if len(parts) < 3:
        raise Failure("system_id", f"unparseable system_id {system_id}")
    return parts[0], parts[1], "_".join(parts[2:])


def composition_delta(ref, pose):
    """Human-readable element imbalance, e.g. "pose missing 2 Fe, 2 S"."""
    from collections import Counter
    r = Counter(a.GetSymbol() for a in ref.GetAtoms())
    p = Counter(a.GetSymbol() for a in pose.GetAtoms())
    missing = ", ".join(f"{n} {el}" for el, n in sorted((r - p).items()))
    extra = ", ".join(f"{n} {el}" for el, n in sorted((p - r).items()))
    parts = []
    if missing:
        parts.append(f"pose missing {missing}")
    if extra:
        parts.append(f"pose has extra {extra}")
    return "; ".join(parts) or "same composition, different count"


def blank_row(system_id):
    pdb, ccd, chain = system_id.split("_")[0], "", ""
    try:
        pdb, ccd, chain = parse_system_id(system_id)
    except Failure:
        pass
    row = {k: "" for k in CSV_FIELDS}
    row.update(system_id=system_id, pdb=pdb, ccd=ccd, chain=chain)
    return row


def process(system_id, pose_path, ccd_smiles):
    row = blank_row(system_id)
    try:
        ref, ref_source = load_reference(system_id, ccd_smiles)
        row["ref_source"] = ref_source
        row["metal_class"] = classify_metal(ref)
        row["n_ref_heavy"] = ref.GetNumAtoms()

        pose = strip_hydrogens(read_sdf(pose_path, "pose"), "pose")
        row["n_pose_heavy"] = pose.GetNumAtoms()

        ref_xyz = ref.GetConformer().GetPositions()
        pose_xyz = pose.GetConformer().GetPositions()
        row["centroid_offset"] = round(float(np.linalg.norm(pose_xyz.mean(0) - ref_xyz.mean(0))), 4)

        if ref.GetNumAtoms() != pose.GetNumAtoms():
            # Usually the PDBQT round-trip dropped part of the ligand -- whole fragments of
            # a metal cluster, most often.  Name the missing elements so the class is
            # identifiable from the CSV without reopening the files.
            raise Failure("atom_count",
                          f"reference {ref.GetNumAtoms()} vs pose {pose.GetNumAtoms()} heavy "
                          f"atoms ({composition_delta(ref, pose)})")

        m = correspond(ref, pose)
        perm = m["perm"]
        if sorted(perm) != list(range(ref.GetNumAtoms())):
            raise Failure("match", "mapping is not a bijection")

        row.update(
            status="ok", reason="",
            rmsd=round(in_place_rmsd(ref_xyz, pose_xyz, perm), 4),
            rmsd_superposed=round(superposed_rmsd(ref_xyz, pose_xyz, perm), 4),
            map_mode=m["map_mode"], match_route=m["match_route"],
            n_candidates=m["n_candidates"], n_plausible=m["n_plausible"],
            candidates_truncated=m["candidates_truncated"],
            chosen_candidate=m["chosen_candidate"],
            chosen_strain=round(m["chosen_strain"], 4),
            rmsd_spread=round(m["rmsd_spread"], 4),
        )
        if m["chosen_strain"] > STRAIN_OK:
            row["reason"] = f"strained_mapping_{m['chosen_strain']:.2f}x"
    except Failure as exc:
        row.update(status="fail", reason=f"{exc.stage}: {exc.detail}")
    except Exception as exc:  # never let one system abort a shard
        row.update(status="fail", reason=f"unexpected: {type(exc).__name__}: {exc}")
    return row


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def write_shard(rows, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"bisy_smina_v2_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    npz_path = os.path.join(OUT_DIR, f"bisy_smina_v2_{tag}.npz")
    np.savez_compressed(
        npz_path,
        system_id=np.array([r["system_id"] for r in ok]),
        ligand_rmsd=np.array([r["rmsd"] for r in ok], np.float32),
        centroid_offset=np.array([r["centroid_offset"] for r in ok], np.float32),
        rmsd_superposed=np.array([r["rmsd_superposed"] for r in ok], np.float32),
        map_mode=np.array([r["map_mode"] for r in ok]),
        match_route=np.array([r["match_route"] for r in ok]),
        metal_class=np.array([r["metal_class"] for r in ok]),
        n_candidates=np.array([r["n_candidates"] for r in ok], np.int32),
        chosen_strain=np.array([r["chosen_strain"] for r in ok], np.float32),
    )
    return csv_path, npz_path


def report(tag):
    """Aggregate every CSV matching ``tag`` (or all of them) and print the statistics."""
    pattern = f"bisy_smina_v2_{tag}*.csv" if tag else "bisy_smina_v2_*.csv"
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
        with open(path) as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        print(f"no CSVs matching {pattern} under {OUT_DIR}")
        return 1

    ok = [r for r in rows if r["status"] == "ok"]
    rmsd = np.array([float(r["rmsd"]) for r in ok])
    cen = np.array([float(r["centroid_offset"]) for r in ok])
    sup = np.array([float(r["rmsd_superposed"]) for r in ok])

    print(f"systems attempted : {len(rows)}")
    print(f"resolved          : {len(ok)} ({100.0 * len(ok) / len(rows):.1f}%)")
    print()
    print("RMSD distribution (crystal frame, no superposition)")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} {np.percentile(rmsd, q):8.2f}")
    print(f"  mean {rmsd.mean():8.2f}   max {rmsd.max():8.2f}")
    print(f"  < 2 A : {(rmsd < 2).sum():5d}  ({100.0 * (rmsd < 2).mean():.1f}%)")
    print(f"  < 5 A : {(rmsd < 5).sum():5d}  ({100.0 * (rmsd < 5).mean():.1f}%)")
    print()
    print(f"centroid offset vs RMSD : pearson r = {np.corrcoef(cen, rmsd)[0, 1]:.3f}  "
          f"spearman r = {spearman(cen, rmsd):.3f}")
    print(f"superposed RMSD median  : {np.median(sup):.2f} A "
          f"(> 2.5 A on {(sup > 2.5).sum()} systems -- correspondence suspects)")
    print()
    print("mapping")
    for mode in ("unique", "ranked"):
        n = sum(1 for r in ok if r["map_mode"] == mode)
        print(f"  {mode:<8s} {n:5d}  ({100.0 * n / len(ok):.1f}%)")
    print("  route:")
    for route in sorted({r["match_route"] for r in ok}):
        print(f"    {route:<12s} {sum(1 for r in ok if r['match_route'] == route):5d}")
    print("  reference source:")
    for src in sorted({r["ref_source"] for r in ok}):
        print(f"    {src:<12s} {sum(1 for r in ok if r['ref_source'] == src):5d}")
    strained = [r for r in ok if r["reason"]]
    if strained:
        print(f"  strained mappings kept with a warning: {len(strained)}")

    print()
    print("metal class")
    for cls in sorted({r["metal_class"] for r in rows}):
        sub = [r for r in rows if r["metal_class"] == cls]
        sok = [r for r in sub if r["status"] == "ok"]
        label = cls or "(none)"
        print(f"  {label:<16s} attempted {len(sub):5d}  resolved {len(sok):5d}")

    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        print()
        print("failures by stage")
        stages = {}
        for r in failures:
            stages[r["reason"].split(":")[0]] = stages.get(r["reason"].split(":")[0], 0) + 1
        for stage, n in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"  {stage:<16s} {n:5d}")
    return 0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N systems")
    ap.add_argument("--systems", default="", help="comma-separated system_ids or a file of them")
    ap.add_argument("--tag", default="", help="output tag; defaults to the shard index")
    ap.add_argument("--report", action="store_true", help="aggregate CSVs and print statistics")
    args = ap.parse_args()

    if args.report:
        return report(args.tag)

    poses = sorted(glob.glob(os.path.join(POSE_DIR, "*.sdf")))
    if args.systems:
        if os.path.exists(args.systems):
            wanted = {l.strip() for l in open(args.systems) if l.strip()}
        else:
            wanted = {s.strip() for s in args.systems.split(",") if s.strip()}
        poses = [p for p in poses if os.path.basename(p)[:-4] in wanted]
    if args.nshards > 1:
        poses = [p for i, p in enumerate(poses) if i % args.nshards == args.shard]
    if args.limit:
        poses = poses[:args.limit]

    import json
    with open(CCD_SMILES) as fh:
        ccd_smiles = json.load(fh)

    rows = [process(os.path.basename(p)[:-4], p, ccd_smiles) for p in poses]

    tag = args.tag or str(args.shard)
    csv_path, npz_path = write_shard(rows, tag)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"smina v2 [{tag}]: attempted={len(rows)} resolved={ok} failed={len(rows) - ok}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
