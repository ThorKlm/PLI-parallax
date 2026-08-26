#!/usr/bin/env python
"""Symmetry-corrected ligand RMSD for the cofolder teachers (chai1, boltz2 and its
MSA / r3-empty variants), with atom correspondence established by graph matching
against a properly bond-perceived reference.

Why this exists
---------------
``datasets/bisy_rmsd.py`` (and its ``bisy_msa*.py`` siblings) compare a predicted ligand
against the crystal ligand with ``spyrmsd.symmrmsd`` over an adjacency matrix built from a
1.9 A distance cutoff on both sides.  ``bisy_smina_v2.py`` documents what that costs; the
same three defects are here:

  * a 1.9 A cutoff cannot express metal coordination, so a heme's four Fe-N bonds are
    simply absent from the graph, ``spyrmsd`` raises ``NonIsomorphicGraphs`` and the bare
    ``except Exception: fail += 1`` drops the system with no record of why.  Worse than for
    smina, *which* metal systems survive depends on whether the **prediction's** Fe-N
    length happens to land on the same side of 1.9 A as the crystal's, so the surviving
    subset is selected on the very thing being measured;
  * partially-modelled crystal ligands are dropped outright on ``len(clig) != len(plig)``.
    A cofolder is handed the full CCD SMILES, so it always predicts the complete ligand;
    any crystal copy with an unmodelled tail therefore fails the equality test, even though
    the RMSD over the atoms that *were* modelled is perfectly well defined;
  * every failure mode -- missing file, unparsable ligand, too few pocket residues,
    non-isomorphic graphs -- collapses into a single ``fail`` counter.

This script establishes the correspondence explicitly, before any RMSD is computed:

  1. the reference is the CCD-templated crystal ligand -- real bond orders, charges and
     aromaticity from the chemical component dictionary, geometry from the structure --
     read from ``/workspace/pb_inputs/<system_id>_ref.sdf`` where the extractor already
     wrote one, otherwise rebuilt through the same code path;
  2. the *prediction* is bond-perceived the same way rather than by distance cutoff: the
     predicted ligand residue is lifted out of the model, connectivity is re-derived from
     coordinates at several covalent tolerances, and the CCD template is fitted onto it
     (``extract_crystal_ref_ligands.build_reference_ligand``), so the pose carries real
     bond orders and real metal coordination;
  3. reference and pose are matched on the element-and-connectivity graph, every
     symmetry-equivalent candidate mapping is enumerated, and the candidates are ranked by
     bond-length plausibility first and RMSD second, with the choice recorded per system;
  4. metals are handled explicitly -- coordination bonds are dropped from both sides for
     the match and reinstated for the plausibility ranking;
  5. a partially-modelled reference is matched *into* the complete prediction as a
     substructure and the RMSD is taken over the modelled atoms, rather than dropping the
     system.  ``match_scope`` says which of the two happened.

Frames
------
Unlike a smina pose, a cofolder prediction is in its own frame, so the pocket-residue
Kabsch superposition the old scripts do is kept.  Its correspondence is taken from the
label generator's own residue enumeration instead of being re-derived: ``res_row`` indexes
the list of CA-bearing standard-AA residues in model-0 file order, and
``crystal_labels`` already stores those CA coordinates, so the crystal side needs no
re-enumeration at all.  (``bisy_rmsd.py``'s ``ca_all`` re-walks the structure but emits a
NaN row for standard-AA residues that have no CA, which the label generator skips -- every
index after the first such residue is shifted by one.)  The prediction side is indexed by
the same ``res_row`` and *verified* residue-by-residue against the stored crystal sequence,
falling back to a sequence alignment when the model is not residue-for-residue identical.

``rmsd`` is therefore the pocket-aligned, symmetry-corrected ligand RMSD in the crystal
frame.  ``pocket_rmsd`` (the residual of that superposition) and ``rmsd_superposed`` (a
ligand-only Kabsch fit) are written alongside as diagnostics: a system with a small
``rmsd_superposed`` and a large ``rmsd`` is a correct conformer in the wrong place, and a
large ``pocket_rmsd`` means the prediction's *fold* is wrong and the ligand number is
measuring that.

Nothing here writes to ``experimental_expansion/bisy``; output goes to ``bisy_v2``.

Usage:
    python bisy_rmsd_v2.py --teacher chai1 --shard 0 --nshards 16
    python bisy_rmsd_v2.py --teacher boltz2_msa --limit 200 --tag val
    python bisy_rmsd_v2.py --report                    # all teachers
    python bisy_rmsd_v2.py --compare                   # v2 vs the shipped bisy/ numbers
"""

import argparse
import csv
import glob
import json
import multiprocessing
import os
import sys
from difflib import SequenceMatcher

import gemmi
import numpy as np
from rdkit import RDLogger

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/datasets")

import bisy_smina_v2 as S  # noqa: E402  -- reference loading and metal handling, shared
import extract_crystal_ref_ligands as X  # noqa: E402
from bisy_smina_v2 import Failure  # noqa: E402

RDLogger.DisableLog("rdApp.*")

D = "/workspace/datasets/experimental_expansion"
OUT_DIR = f"{D}/bisy_v2"
STRUCT_DIR = f"{D}/structures"
CCD_SMILES = f"{D}/ccd_smiles.json"
POCKET_CACHE = f"{OUT_DIR}/_pocket_cache.npz"
OLD_DIR = f"{D}/bisy"

# Where each teacher's model-0 prediction lives, and what its ligand residue is called.
# ``boltz2`` is the single-sequence run (``msa: empty`` in the input YAML); ``boltz2_msa``
# and ``boltz2_msa2`` are the two MSA samples; ``boltz2_r3empty`` is the r3 single-sequence
# rerun.  chai1 has no MSA variant -- it runs on ESM embeddings (``run_chai.py``).
TEACHERS = {
    "chai1": {
        "path": "/workspace/docking/output/chai_out_exp/{sid}/pred.model_idx_0.cif",
        "ligand": "LIG2",
        "note": "single sequence (ESM embeddings)",
    },
    "boltz2": {
        "path": f"{D}/boltz_out/boltz_results_boltz_in/predictions/{{sid}}/{{sid}}_model_0.pdb",
        "ligand": "LIG",
        "note": "single sequence (msa: empty)",
    },
    "boltz2_msa": {
        "path": f"{D}/boltz_out_msa/boltz_results_boltz_in_msa/predictions/{{sid}}/{{sid}}_model_0.pdb",
        "ligand": "LIG",
        "note": "MSA sample 1",
    },
    "boltz2_msa2": {
        "path": f"{D}/boltz_out_msa2/boltz_results_boltz_in_msa2/predictions/{{sid}}/{{sid}}_model_0.pdb",
        "ligand": "LIG",
        "note": "MSA sample 2",
    },
    "boltz2_r3empty": {
        "path": f"{D}/boltz_out_r3_empty_full/boltz_results_boltz_in_r3_empty_full/"
                f"predictions/{{sid}}/{{sid}}_model_0.pdb",
        "ligand": "LIG",
        "note": "r3 single-sequence rerun",
    },
}

# The old scripts' npz per teacher, for the coverage-vs-value comparison.
OLD_FILES = {
    "chai1": "bisy_chai1_*.npz",
    "boltz2": "bisy_boltz2_[0-9]*.npz",
    "boltz2_msa": "bisy_boltz2_msa.npz",
    "boltz2_msa2": "bisy_boltz2_msa2.npz",
    "boltz2_r3empty": None,  # never scored by the old scripts
}

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

MIN_POCKET = 4          # Kabsch needs 3 non-collinear points; the old scripts used 4

# Per-system wall-clock budget.  A handful of large, highly symmetric ligands send RDKit's
# substructure search into a combinatorial blow-up that ``maxMatches`` does not bound -- it
# caps the matches *returned*, not the search tree walked.  8qln_TEW_A, an Anderson
# polyoxotungstate, is the worst: six equivalent tungstens and two dozen bridging oxygens,
# and neither the reference build nor the candidate enumeration terminates in any useful
# time.  Rather than let one system stall a shard, it is abandoned with a recorded reason
# like any other failure.  60 s is ~5000x the median system.
SYSTEM_TIMEOUT = 60
MAX_MATCHES = S.MAX_MATCHES
STRAIN_OK = S.STRAIN_OK

CSV_FIELDS = [
    "teacher", "system_id", "pdb", "ccd", "chain", "status", "reason",
    "n_ref_heavy", "n_pose_heavy", "n_matched",
    "rmsd", "rmsd_superposed", "centroid_offset", "pocket_rmsd", "n_pocket", "pocket_map",
    "ref_source", "pose_source", "map_mode", "match_route", "match_scope", "metal_class",
    "n_candidates", "n_plausible", "candidates_truncated",
    "chosen_candidate", "chosen_strain", "rmsd_spread",
]


# --------------------------------------------------------------------------------------
# Pocket bookkeeping
# --------------------------------------------------------------------------------------


class Pockets:
    """The label generator's residue enumeration, per system.

    ``res_row`` in ``crystal_labels`` indexes the list of standard-AA residues that have a
    CA *and* at least one heavy atom, in model-0 file order across all chains.  The CA
    coordinates of the extended pocket are stored there too, and every contact residue is
    inside the extended pocket by construction, so the crystal side of the superposition
    is read straight out of the labels -- no re-enumeration, and no chance of drifting out
    of step with them.
    """

    def __init__(self, path=POCKET_CACHE):
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing -- run with --build-cache first")
        d = np.load(path, allow_pickle=True)
        self.index = {str(s): i for i, s in enumerate(d["system_id"])}
        self.seq = d["seq"]
        self.n_res = d["n_res"]
        self.res_row, self.res_off = d["res_row"], d["res_off"]
        self.pocket_res, self.pocket_ca = d["pocket_res"], d["pocket_ca"]
        self.pocket_off = d["pocket_off"]

    def __contains__(self, system_id):
        return system_id in self.index

    def get(self, system_id):
        i = self.index.get(system_id)
        if i is None:
            raise Failure("pocket", "system not in crystal_labels")
        a, b = int(self.res_off[i]), int(self.res_off[i + 1])
        c, e = int(self.pocket_off[i]), int(self.pocket_off[i + 1])
        ca_by_res = dict(zip(self.pocket_res[c:e].tolist(), self.pocket_ca[c:e]))
        return {
            "seq": str(self.seq[i]),
            "n_res": int(self.n_res[i]),
            "contacts": self.res_row[a:b],
            "ca_by_res": ca_by_res,
        }



def build_pocket_cache(path=POCKET_CACHE):
    """Condense the crystal_labels shards into the few fields this script needs.

    The shards are 278 MB and their ``seq`` column is a fixed-width unicode array (182 kB
    per row, sized by the longest sequence in the set), so touching them costs about a
    gigabyte of RAM per shard.  Every field below is copied verbatim from the label
    generator's own residue enumeration -- which is precisely what makes ``res_row``
    indexable later without re-deriving it from the structure.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sid, nres, seqs = [], [], []
    rows, roff = [], [0]
    pres, pca, poff = [], [], [0]
    for shard in sorted(glob.glob(f"{D}/crystal_labels/crystal_labels_*.npz")):
        d = np.load(shard, allow_pickle=True)
        ids = [str(x) for x in d["system_id"]]
        n, co, rr = d["n_res"], d["contact_offsets"], d["res_row"]
        po, pr, pc = d["pocket_offsets"], d["pocket_res"], d["pocket_ca"]
        seq = d["seq"]
        for i, s in enumerate(ids):
            sid.append(s)
            nres.append(int(n[i]))
            seqs.append(str(seq[i]))
            unique = np.unique(rr[int(co[i]):int(co[i + 1])]).astype(np.int32)
            rows.append(unique)
            roff.append(roff[-1] + len(unique))
            a, b = int(po[i]), int(po[i + 1])
            pres.append(pr[a:b].astype(np.int32))
            pca.append(pc[a:b].astype(np.float32))
            poff.append(poff[-1] + (b - a))
        del d, seq
        print(f"  {os.path.basename(shard)}: {len(sid)} systems", flush=True)
    np.savez_compressed(
        path,
        system_id=np.array(sid), n_res=np.array(nres, np.int32),
        seq=np.array(seqs, dtype=object),
        res_row=np.concatenate(rows), res_off=np.array(roff, np.int64),
        pocket_res=np.concatenate(pres), pocket_ca=np.concatenate(pca),
        pocket_off=np.array(poff, np.int64),
    )
    print(f"wrote {path} ({len(sid)} systems)")
    return 0


def predicted_residues(model):
    """(one-letter sequence, CA coordinates) over the prediction's standard-AA residues,
    in the same order the label generator enumerated the crystal's."""
    letters, xyz = [], []
    for chain in model:
        for res in chain:
            if res.name not in AA3:
                continue
            ca = None
            for atom in res:
                if atom.name == "CA" and atom.element.name not in ("H", "D"):
                    ca = (atom.pos.x, atom.pos.y, atom.pos.z)
            if ca is None:
                continue
            letters.append(AA3[res.name])
            xyz.append(ca)
    return "".join(letters), np.array(xyz, float).reshape(-1, 3)


def residue_mapping(crystal_seq, pred_seq):
    """(map, mode): ``map[crystal_res_index] = prediction_res_index``, or -1.

    The cofolder was handed the crystal sequence verbatim, so the identity map is right
    almost always -- but it is *checked* rather than assumed, and a model that dropped or
    duplicated residues falls back to a sequence alignment instead of silently pairing
    unrelated residues, which is the failure mode a raw index carries.
    """
    if pred_seq == crystal_seq:
        return np.arange(len(crystal_seq), dtype=np.int64), "identity"
    out = np.full(len(crystal_seq), -1, np.int64)
    for a, b, size in SequenceMatcher(None, crystal_seq, pred_seq, autojunk=False)\
            .get_matching_blocks():
        out[a:a + size] = np.arange(b, b + size)
    if (out >= 0).sum() < MIN_POCKET:
        raise Failure("pocket", "prediction sequence does not align to the crystal one")
    return out, "aligned"


def kabsch(P, Q):
    """(R, P_centroid, Q_centroid) rotating P onto Q."""
    Pc, Qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T, Pc, Qc


def pocket_superposition(pockets, system_id, pred_model):
    """Rigid transform taking the prediction into the crystal frame, from pocket CAs."""
    info = pockets.get(system_id)
    pred_seq, pred_ca = predicted_residues(pred_model)
    if not len(pred_ca):
        raise Failure("pocket", "prediction has no CA-bearing standard residues")
    mapping, mode = residue_mapping(info["seq"], pred_seq)

    contacts = info["contacts"]
    contacts = contacts[(contacts >= 0) & (contacts < len(mapping))]
    rows = [(info["ca_by_res"][int(r)], pred_ca[mapping[int(r)]])
            for r in contacts
            if int(r) in info["ca_by_res"] and mapping[int(r)] >= 0]
    if len(rows) < MIN_POCKET:
        raise Failure("pocket", f"only {len(rows)} usable contact residues "
                                f"({len(contacts)} labelled)")
    Q = np.array([r[0] for r in rows], float)      # crystal
    P = np.array([r[1] for r in rows], float)      # prediction
    good = ~(np.isnan(P).any(1) | np.isnan(Q).any(1))
    P, Q = P[good], Q[good]
    if len(P) < MIN_POCKET:
        raise Failure("pocket", f"only {len(P)} contact residues with coordinates")

    R, Pc, Qc = kabsch(P, Q)
    resid = float(np.sqrt((((R @ (P - Pc).T).T + Qc - Q) ** 2).sum(1).mean()))
    return (lambda xyz: (R @ (xyz - Pc).T).T + Qc), resid, len(P), mode


# --------------------------------------------------------------------------------------
# The predicted ligand, bond-perceived
# --------------------------------------------------------------------------------------


def find_ligand_residue(model, resname):
    for chain in model:
        for res in chain:
            if res.name == resname:
                return res, chain.name
    return None, None


def load_pose(path, resname, ccd, ccd_smiles):
    """(pose_mol, source).  The predicted ligand with real bonds, in CCD template order.

    The prediction is a bare coordinate model: PDB/mmCIF carry no bond information, which
    is exactly why the old scripts fell back to a distance cutoff.  Instead the residue is
    serialised through ``extract_crystal_ref_ligands``'s own ligand path, which re-derives
    connectivity from coordinates at several covalent tolerances and then fits the CCD
    template onto it -- so the pose ends up with the CCD's bond orders, charges,
    aromaticity and metal coordination, not a thresholded distance matrix.
    """
    try:
        model = gemmi.read_structure(path)[0]
    except Exception as exc:
        raise Failure("pose_read", f"{type(exc).__name__}: {exc}")
    res, chain_id = find_ligand_residue(model, resname)
    if res is None:
        raise Failure("pose_read", f"no {resname} residue in the prediction")

    try:
        crude = X.mol_from_pdb_block(X.ligand_pdb_block(res, chain_id))
    except X.SystemFailure as exc:
        raise Failure("pose_parse", f"{exc.stage}: {exc.reason}")
    except Exception as exc:
        raise Failure("pose_parse", f"{type(exc).__name__}: {exc}")

    smiles = ccd_smiles.get(ccd)
    if smiles:
        try:
            template, _ = X.load_template(smiles, ccd)
            pose, _mode = X.build_reference_ligand(crude, template)
            return S.strip_hydrogens(pose, "pose"), "ccd_template", model
        except (X.SystemFailure, Failure, Exception):
            # The template would not fit -- an unmodelled prediction, a CCD whose SMILES
            # disagrees with what the cofolder was actually given.  The re-perceived
            # connectivity is still far better than a 1.9 A cutoff, so fall back to it and
            # say so in ``pose_source`` rather than dropping the system.
            pass
    return S.strip_hydrogens(reperceived(crude), "pose"), "perceived", model


def reperceived(mol):
    """Best connectivity RDKit will give for a bare coordinate ligand."""
    for factor in (1.3, 1.5):
        alt = X.reperceive(mol, factor)
        if alt is not None and alt.GetNumBonds() >= mol.GetNumBonds():
            return alt
    return mol


# --------------------------------------------------------------------------------------
# Atom correspondence
# --------------------------------------------------------------------------------------


def enumerate_candidates(ref, pose):
    """(matches, route, scope).  ``match[ref_i] = pose_i`` for every isomorphism found.

    The direction is reference-into-pose, the opposite of ``bisy_smina_v2``'s, because a
    crystal ligand is routinely modelled with fewer atoms than the CCD while a cofolder --
    handed the full CCD SMILES -- always predicts all of them.  Matching the reference as a
    *substructure* of the prediction therefore scores the atoms that were actually
    observed, instead of failing the system on a count mismatch the way the old scripts do.
    ``scope`` records whether the match was atom-for-atom or partial.

    As in ``bisy_smina_v2``, candidates are enumerated on the element-and-connectivity
    graph with bond orders, charges and aromaticity flattened away: which oxygen of a
    phosphate carries the double bond is an arbitrary deposition choice, so
    resonance-equivalent atoms have to be interchangeable.  Bond orders still do work --
    they are what ``load_pose`` used to establish that the prediction is the CCD compound
    at all.
    """
    scope = "full" if ref.GetNumAtoms() == pose.GetNumAtoms() else "partial"
    routes = [("topology", X.flatten(pose), X.flatten(ref))]
    if S.has_metal(ref) or S.has_metal(pose):
        bare_ref, bare_pose = X.strip_metal_bonds(ref), X.strip_metal_bonds(pose)
        if bare_ref is not None or bare_pose is not None:
            routes.append((
                "metalfree",
                X.flatten(bare_pose if bare_pose is not None else pose),
                X.flatten(bare_ref if bare_ref is not None else ref),
            ))
    for name, target, query in routes:
        try:
            matches = target.GetSubstructMatches(
                query, uniquify=False, useChirality=False, maxMatches=MAX_MATCHES)
        except Exception:
            continue
        if matches:
            return [list(m) for m in matches], name, scope
    raise Failure("match", "no atom-graph isomorphism between prediction and reference")


def reference_bonds(ref):
    """(begin, end, limit) for the reference bond graph; ``limit`` is the length above
    which that bond is implausible (``extract_crystal_ref_ligands``' own limits)."""
    begin, end, limit = [], [], []
    for bond in ref.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        begin.append(a.GetIdx())
        end.append(b.GetIdx())
        limit.append(X.MAX_METAL_BOND
                     if (a.GetSymbol() in X.METALS or b.GetSymbol() in X.METALS)
                     else X.MAX_BOND)
    return np.array(begin, np.int64), np.array(end, np.int64), np.array(limit, float)


def score_candidates(ref, ref_xyz, pose_xyz, matches):
    """(strain, rmsd) per candidate mapping.

    ``strain[k]`` is the worst reference bond length implied by mapping ``k``, relative to
    its plausibility limit; <= 1.0 is clean.  A genuine automorphism maps the bond set onto
    itself and every candidate scores identically -- correctly, since those candidates are
    indistinguishable and the RMSD tie-break decides.  It bites where the graph was reduced
    for the match (the metalfree route) or where the reference matched only partially.
    """
    perms = np.asarray(matches, np.int64)                     # perms[k, ref_i] = pose_i
    placed = pose_xyz[perms]                                  # pose coords in ref order
    rmsd = np.sqrt(((placed - ref_xyz[None]) ** 2).sum(-1).mean(-1))

    begin, end, limit = reference_bonds(ref)
    if not len(begin):
        return np.zeros(len(perms)), rmsd
    lengths = np.linalg.norm(placed[:, begin] - placed[:, end], axis=-1)
    return (lengths / limit).max(1), rmsd


def superposed_rmsd(ref_xyz, pose_xyz):
    """Kabsch RMSD of the corresponded atoms alone -- diagnostic, separating a
    mis-corresponded ligand (large) from a correct one placed wrongly (small)."""
    R, Pc, Qc = kabsch(pose_xyz, ref_xyz)
    return float(np.sqrt((((R @ (pose_xyz - Pc).T).T + Qc - ref_xyz) ** 2).sum(1).mean()))


def correspond(ref, ref_xyz, pose, pose_xyz):
    """Establish the atom correspondence and score it (see ``enumerate_candidates``)."""
    matches, route, scope = enumerate_candidates(ref, pose)
    truncated = int(len(matches) >= MAX_MATCHES)

    strain, rmsd = score_candidates(ref, ref_xyz, pose_xyz, matches)
    plausible = np.flatnonzero(strain <= STRAIN_OK)
    if not len(plausible):
        # Nothing is clean.  Keep the least-strained candidates and say so in ``reason``
        # rather than dropping the system.
        plausible = np.flatnonzero(strain <= strain.min() + 1e-9)
    # Plausibility first, RMSD second: among chemically indistinguishable mappings the
    # closest is the symmetry-corrected answer, but a mapping that strains the molecule
    # never wins on a low RMSD alone.
    idx = int(min(plausible, key=lambda k: (strain[k], rmsd[k])))
    return {
        "perm": matches[idx],
        "map_mode": "unique" if len(matches) == 1 else "ranked",
        "match_route": route,
        "match_scope": scope,
        "n_candidates": len(matches),
        "n_plausible": len(plausible),
        "candidates_truncated": truncated,
        "chosen_candidate": idx,
        "chosen_strain": float(strain[idx]),
        "rmsd_spread": float(rmsd[plausible].max() - rmsd[plausible].min()),
    }


# --------------------------------------------------------------------------------------
# Per-system driver
# --------------------------------------------------------------------------------------




def run_isolated(target, args, seconds):
    """Run ``target(*args)`` in a forked child and abandon it if it overruns.

    The budget cannot be enforced in-process.  The blow-up that makes it necessary happens
    inside RDKit's C++ substructure matcher, which holds the GIL for the whole call, so a
    ``signal.alarm`` handler is never scheduled until that call returns -- which is exactly
    what is not happening.  A forked child can be killed.  Fork is cheap here because the
    pocket cache and the CCD table are inherited copy-on-write: ~17 ms per system against a
    ~11 ms median system, paid once.

    Returns the target's return value, or ``None`` if it overran or died.
    """
    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(False)

    def child():
        sender.send(target(*args))

    worker = ctx.Process(target=child)
    worker.start()
    sender.close()
    result = None
    if receiver.poll(seconds):
        try:
            result = receiver.recv()
        except EOFError:
            result = None            # the child died without answering
    else:
        worker.terminate()
    receiver.close()
    worker.join(5)
    if worker.is_alive():
        worker.kill()
        worker.join()
    return result


def blank_row(teacher, system_id):
    row = {k: "" for k in CSV_FIELDS}
    row.update(teacher=teacher, system_id=system_id)
    try:
        pdb, ccd, chain = S.parse_system_id(system_id)
    except Failure:
        pdb, ccd, chain = system_id, "", ""
    row.update(pdb=pdb, ccd=ccd, chain=chain)
    return row


def score_system(teacher, system_id, pockets, ccd_smiles, row):
    """Fill ``row`` for one system; every failure is raised as a tagged ``Failure``."""
    spec = TEACHERS[teacher]
    _pdb, ccd, _chain = S.parse_system_id(system_id)
    path = spec["path"].format(sid=system_id)
    if not os.path.exists(path):
        raise Failure("prediction", "no model-0 file")

    ref, ref_source = S.load_reference(system_id, ccd_smiles)
    row["ref_source"] = ref_source
    row["metal_class"] = S.classify_metal(ref)
    row["n_ref_heavy"] = ref.GetNumAtoms()

    pose, pose_source, model = load_pose(path, spec["ligand"], ccd, ccd_smiles)
    row["pose_source"] = pose_source
    row["n_pose_heavy"] = pose.GetNumAtoms()
    if pose.GetNumAtoms() < ref.GetNumAtoms():
        # The prediction is missing atoms the crystal has -- the reverse of the usual
        # case, and not something a substructure match can rescue.
        raise Failure("atom_count",
                      f"reference {ref.GetNumAtoms()} vs prediction "
                      f"{pose.GetNumAtoms()} heavy atoms "
                      f"({S.composition_delta(ref, pose)})")

    to_crystal, pocket_rmsd, n_pocket, pocket_map = \
        pocket_superposition(pockets, system_id, model)
    row.update(pocket_rmsd=round(pocket_rmsd, 4), n_pocket=n_pocket, pocket_map=pocket_map)

    ref_xyz = ref.GetConformer().GetPositions()
    pose_xyz = to_crystal(pose.GetConformer().GetPositions())

    m = correspond(ref, ref_xyz, pose, pose_xyz)
    perm = m["perm"]
    if len(set(perm)) != len(perm):
        raise Failure("match", "mapping is not injective")
    matched = pose_xyz[perm]

    row.update(
        status="ok", reason="",
        n_matched=len(perm),
        rmsd=round(float(np.sqrt(((matched - ref_xyz) ** 2).sum(1).mean())), 4),
        rmsd_superposed=round(superposed_rmsd(ref_xyz, matched), 4),
        centroid_offset=round(float(np.linalg.norm(matched.mean(0) - ref_xyz.mean(0))), 4),
        map_mode=m["map_mode"], match_route=m["match_route"],
        match_scope=m["match_scope"], n_candidates=m["n_candidates"],
        n_plausible=m["n_plausible"], candidates_truncated=m["candidates_truncated"],
        chosen_candidate=m["chosen_candidate"],
        chosen_strain=round(m["chosen_strain"], 4),
        rmsd_spread=round(m["rmsd_spread"], 4),
    )
    if m["chosen_strain"] > STRAIN_OK:
        row["reason"] = f"strained_mapping_{m['chosen_strain']:.2f}x"


def score_row(teacher, system_id, pockets, ccd_smiles):
    row = blank_row(teacher, system_id)
    try:
        score_system(teacher, system_id, pockets, ccd_smiles, row)
    except Failure as exc:
        row.update(status="fail", reason=f"{exc.stage}: {exc.detail}")
    except Exception as exc:  # never let one system abort a shard
        row.update(status="fail", reason=f"unexpected: {type(exc).__name__}: {exc}")
    return row


def process(teacher, system_id, pockets, ccd_smiles):
    row = run_isolated(score_row, (teacher, system_id, pockets, ccd_smiles),
                       SYSTEM_TIMEOUT)
    if row is None:
        row = blank_row(teacher, system_id)
        row.update(status="fail", reason=f"timeout: no result within {SYSTEM_TIMEOUT}s")
    return row


def systems_for(teacher, pockets):
    """Every system this teacher produced a prediction for, in a stable order.

    The universe is the crystal-label systems -- a prediction with no labels has nothing to
    be scored against -- and the denominator reported is *available predictions*, so a
    teacher that was never run on a system is not counted against its resolve rate.
    """
    spec = TEACHERS[teacher]
    root = spec["path"].split("{sid}")[0].rstrip("/")
    try:
        present = set(os.listdir(root))
    except OSError:
        return []
    return sorted(s for s in present if s in pockets)


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def write_shard(rows, teacher, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"bisy_rmsd_v2_{teacher}_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    npz_path = os.path.join(OUT_DIR, f"bisy_rmsd_v2_{teacher}_{tag}.npz")
    np.savez_compressed(
        npz_path,
        system_id=np.array([r["system_id"] for r in ok]),
        ligand_rmsd=np.array([r["rmsd"] for r in ok], np.float32),
        rmsd_superposed=np.array([r["rmsd_superposed"] for r in ok], np.float32),
        centroid_offset=np.array([r["centroid_offset"] for r in ok], np.float32),
        pocket_resid=np.array([r["pocket_rmsd"] for r in ok], np.float32),
        n_matched=np.array([r["n_matched"] for r in ok], np.int32),
        match_scope=np.array([r["match_scope"] for r in ok]),
        map_mode=np.array([r["map_mode"] for r in ok]),
        match_route=np.array([r["match_route"] for r in ok]),
        metal_class=np.array([r["metal_class"] for r in ok]),
        chosen_strain=np.array([r["chosen_strain"] for r in ok], np.float32),
    )
    return csv_path, npz_path


def read_rows(teacher=""):
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "bisy_rmsd_v2_*.csv"))):
        with open(path) as fh:
            # Filter on the column rather than the filename: teacher names are prefixes of
            # each other ("boltz2" of "boltz2_msa"), so a glob would fold the variants
            # together.
            rows.extend(r for r in csv.DictReader(fh)
                        if not teacher or r["teacher"] == teacher)
    # A system can appear in more than one CSV if a shard was rerun under a new tag.
    seen, unique = set(), []
    for r in rows:
        key = (r["teacher"], r["system_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def old_rmsd(teacher):
    """{system_id: rmsd} from the shipped ``bisy/`` npz files, or None if never scored."""
    pattern = OLD_FILES.get(teacher)
    if not pattern:
        return None
    out = {}
    for path in sorted(glob.glob(os.path.join(OLD_DIR, pattern))):
        d = np.load(path, allow_pickle=True)
        for sid, val in zip(d["system_id"], d["ligand_rmsd"]):
            out[str(sid)] = float(val)
    return out



def old_unattempted(teacher, pockets):
    """The systems the old run never reached, as distinct from the ones it dropped.

    ``bisy_rmsd.py`` shards ``sorted(crystal_labels system_ids)`` 16 ways and writes one
    npz per shard.  The shipped ``bisy/`` directory holds 15 of the 16 for both chai1 and
    boltz2 -- shard 7 never wrote one -- so about 1/16 of the corpus is absent from the old
    numbers for a reason that has nothing to do with the metric.  Counting those as
    "silently dropped" would overstate the fix, so they are separated out here.
    """
    pattern = OLD_FILES.get(teacher) or ""
    if "[0-9]" not in pattern:
        return set()                      # the MSA runs were not sharded
    present = set()
    for path in glob.glob(os.path.join(OLD_DIR, pattern)):
        stem = os.path.basename(path)[:-len(".npz")]
        present.add(int(stem.rsplit("_", 1)[1]))
    n_shards = 16                          # the value bisy_rmsd.py was driven with
    missing = set(range(n_shards)) - present
    if not missing:
        return set()
    ordered = sorted(pockets.index)
    return {s for j, s in enumerate(ordered) if j % n_shards in missing}


def describe(rmsd, indent="  "):
    for q in (5, 25, 50, 75, 95):
        print(f"{indent}p{q:<3d} {np.percentile(rmsd, q):8.2f}")
    print(f"{indent}mean {rmsd.mean():8.2f}   max {rmsd.max():8.2f}")
    print(f"{indent}< 2 A : {(rmsd < 2).sum():6d}  ({100.0 * (rmsd < 2).mean():5.1f}%)")
    print(f"{indent}< 5 A : {(rmsd < 5).sum():6d}  ({100.0 * (rmsd < 5).mean():5.1f}%)")


def report(teacher=""):
    pockets = Pockets()
    names = [teacher] if teacher else list(TEACHERS)
    for name in names:
        rows = [r for r in read_rows(name)]
        if not rows:
            print(f"== {name}: no CSVs under {OUT_DIR}\n")
            continue
        available = len(systems_for(name, pockets))
        ok = [r for r in rows if r["status"] == "ok"]
        rmsd = np.array([float(r["rmsd"]) for r in ok])
        print(f"== {name}  ({TEACHERS[name]['note']})")
        print(f"  predictions available : {available}")
        print(f"  attempted             : {len(rows)}")
        print(f"  resolved              : {len(ok)} "
              f"({100.0 * len(ok) / max(len(rows), 1):.1f}% of attempted)")
        if not len(ok):
            print()
            continue
        print("  RMSD (pocket-aligned, crystal frame)")
        describe(rmsd, "    ")
        pocket = np.array([float(r["pocket_rmsd"]) for r in ok])
        sup = np.array([float(r["rmsd_superposed"]) for r in ok])
        print(f"    pocket superposition residual median {np.median(pocket):.2f} A "
              f"(> 5 A on {(pocket > 5).sum()} systems)")
        print(f"    ligand-only superposed RMSD  median {np.median(sup):.2f} A")
        partial = sum(1 for r in ok if r["match_scope"] == "partial")
        print(f"    partially-modelled references scored: {partial}")
        for cls in sorted({r["metal_class"] for r in rows}):
            sub = [r for r in rows if r["metal_class"] == cls]
            sok = [r for r in sub if r["status"] == "ok"]
            print(f"    metal_class {cls or '(none)':<16s} attempted {len(sub):5d} "
                  f"resolved {len(sok):5d}")
        fails = [r for r in rows if r["status"] != "ok"]
        if fails:
            stages = {}
            for r in fails:
                stage = r["reason"].split(":")[0]
                stages[stage] = stages.get(stage, 0) + 1
            print("    failures by stage: " + ", ".join(
                f"{k}={v}" for k, v in sorted(stages.items(), key=lambda kv: -kv[1])))
        print()
    return 0


def compare(teacher=""):
    """v2 against the shipped bisy/ numbers, splitting coverage from value."""
    pockets = Pockets()
    names = [teacher] if teacher else list(TEACHERS)
    for name in names:
        rows = read_rows(name)
        if not rows:
            print(f"== {name}: no v2 CSVs\n")
            continue
        old = old_rmsd(name)
        new = {r["system_id"]: float(r["rmsd"]) for r in rows if r["status"] == "ok"}
        attempted = {r["system_id"] for r in rows}
        print(f"== {name}")
        if old is None:
            print(f"  never scored by the old scripts; v2 resolves {len(new)} systems\n")
            continue
        shared = sorted(set(old) & set(new))
        gained = sorted(set(new) - set(old))
        lost = sorted(set(old) - set(new))
        unreached = old_unattempted(name, pockets)
        dropped = attempted - set(old)                  # old produced no number
        never = dropped & unreached                     # ... because it never ran
        silent = dropped - unreached                    # ... because it failed silently
        print(f"  old resolved {len(old)}   v2 resolved {len(new)}   "
              f"(v2 attempted {len(attempted)})")
        if never:
            print(f"  never reached by the old run (missing shard output): {len(never)}")
        print(f"  silently dropped by the old script: {len(silent)} "
              f"({100.0 * len(silent) / max(len(attempted) - len(never), 1):.1f}% of the "
              f"predictions it actually walked); v2 recovers {len(silent & set(new))}")
        if shared:
            a = np.array([old[s] for s in shared])
            b = np.array([new[s] for s in shared])
            d = b - a
            print(f"  VALUE  on the {len(shared)} systems both scored:")
            print(f"    median old {np.median(a):6.2f}  ->  new {np.median(b):6.2f}")
            print(f"    |delta|  p50 {np.percentile(abs(d), 50):.3f}  "
                  f"p95 {np.percentile(abs(d), 95):.3f}  max {abs(d).max():.3f} A")
            print(f"    identical within 0.01 A : {(abs(d) < 0.01).sum()} "
                  f"({100.0 * (abs(d) < 0.01).mean():.1f}%)")
            print(f"    < 2 A  old {100.0 * (a < 2).mean():5.1f}%  ->  "
                  f"new {100.0 * (b < 2).mean():5.1f}%")
            print(f"    < 5 A  old {100.0 * (a < 5).mean():5.1f}%  ->  "
                  f"new {100.0 * (b < 5).mean():5.1f}%")
        if gained:
            g = np.array([new[s] for s in gained])
            print(f"  COVERAGE  the {len(gained)} systems only v2 scores:")
            print(f"    median {np.median(g):.2f}   < 2 A {100.0 * (g < 2).mean():.1f}%   "
                  f"< 5 A {100.0 * (g < 5).mean():.1f}%")
            ccds = {}
            for s in gained:
                ccds[s.split("_")[1]] = ccds.get(s.split("_")[1], 0) + 1
            top = sorted(ccds.items(), key=lambda kv: -kv[1])[:8]
            print("    top CCDs: " + ", ".join(f"{c} {n}" for c, n in top))
        if shared and gained:
            b = np.array([new[s] for s in shared])
            g = np.array([new[s] for s in gained])
            print(f"    newly-included vs previously-scored (same v2 metric): "
                  f"median {np.median(g):.2f} vs {np.median(b):.2f}, "
                  f"< 2 A {100.0 * (g < 2).mean():.1f}% vs {100.0 * (b < 2).mean():.1f}%, "
                  f"mannwhitney p = {mannwhitney(g, b):.2e}")
        if lost:
            print(f"  scored by the old script but not by v2: {len(lost)}")
        print()
    return 0


def mannwhitney(x, y):
    """Two-sided Mann-Whitney U p-value via the normal approximation with tie correction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if not n1 or not n2:
        return float("nan")
    both = np.concatenate([x, y])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty(len(both), float)
    sorted_vals = both[order]
    i = 0
    ties = 0.0
    while i < len(both):
        j = i
        while j + 1 < len(both) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    from math import erfc, sqrt
    return float(erfc(abs(u - mu) / sqrt(var) / sqrt(2.0)))


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", default="", choices=[""] + list(TEACHERS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--systems", default="", help="comma-separated system_ids or a file")
    ap.add_argument("--tag", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--build-cache", action="store_true",
                    help="condense crystal_labels into the pocket cache (once, ~3 min)")
    args = ap.parse_args()

    if args.build_cache:
        return build_pocket_cache()
    if args.report:
        return report(args.teacher)
    if args.compare:
        return compare(args.teacher)
    if not args.teacher:
        ap.error("--teacher is required unless --report/--compare")

    pockets = Pockets()
    systems = systems_for(args.teacher, pockets)
    if args.systems:
        if os.path.exists(args.systems):
            wanted = {l.strip() for l in open(args.systems) if l.strip()}
        else:
            wanted = {s.strip() for s in args.systems.split(",") if s.strip()}
        systems = [s for s in systems if s in wanted]
    if args.nshards > 1:
        systems = [s for i, s in enumerate(systems) if i % args.nshards == args.shard]
    if args.limit:
        systems = systems[:args.limit]

    with open(CCD_SMILES) as fh:
        ccd_smiles = json.load(fh)

    rows = []
    for i, system_id in enumerate(systems):
        rows.append(process(args.teacher, system_id, pockets, ccd_smiles))
        if (i + 1) % 250 == 0:
            print(f"  {args.teacher} [{args.tag or args.shard}] {i + 1}/{len(systems)}",
                  flush=True)
    tag = args.tag or str(args.shard)
    csv_path, npz_path = write_shard(rows, args.teacher, tag)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"{args.teacher} [{tag}]: attempted={len(rows)} resolved={ok} "
          f"failed={len(rows) - ok}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
