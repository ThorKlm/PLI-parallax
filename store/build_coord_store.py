"""Coordinate store builder, v6.
Three coordinates per residue: ca, sc_centroid, pep_c, all full-protein.
Residue gate matched to the label extractor, so the store and the distance
tables share one residue enumeration.

Derived from build_coord_store.py (md5 895ef6f2c4318eeb7ee4c447e9dc4bab) with
four changes, all made on 28 August 2026:

1. res_types is written as a dataset rather than an HDF5 attribute. As an
   attribute it exceeds the 64 KB object-header limit above roughly 21,800
   CA residues, which raised OSError after the group and ligand data had been
   written and left a malformed partial group. The July crystal store contains
   42 such groups.

2. gather() accepts {sid}_model_0.cif as well as {sid}*_model_0.pdb for the
   boltz2 teacher. The regenerated corpus arm was written with
   --output_format mmcif; the superseded arm was PDB.

3. A receptor fallback for smina. smina docks into an AlphaFold model and
   writes only an SDF, so the per-teacher protein loop produces no protein
   group for it and every docking-only system previously carried a ligand and
   no receptor. The fallback reads that same AlphaFold model, so the pose is
   already in its frame and no superposition is needed. Requires --af and
   --sys2acc. Recovers roughly 8,100 corpus systems.

4. Used with roots_corpus_v3.json rather than roots_corpus.json. The v2 file
   lists six of the eight Chai roots and points boltz2 at boltz_out_fill, the
   1,339 systems that survived the deletion of the original corpus arm. The v3
   file names one root per teacher, each a merged tree built in v2 roots order
   with no-clobber so per-system precedence is unchanged:
     chai_union         24,738 symlinks over eight Chai roots
     boltz_corpus_final 23,494 systems, regenerated arm, first-wins by instance
     smina_union        121,367 SDFs over 22 docking roots
   roots_corpus.json is retained as the record of per-pose provenance.

Invocation used for the deposited corpus tier is in run_v3.sh.
"""
"""Build the tripose coordinate datasets from raw teacher structures.

Multi-teacher 3D coordinates for protein-ligand pose supervision, keyed by
system_id, from raw mmCIF (Chai), PDB (Boltz), SDF (smina), crystal.

Views per system:
  A residue-level : full-protein Ca per residue + ligand per atom
  B shell atom    : protein atoms within SHELL A of the ligand + ligand per atom

Teacher axis (fixed), 0 = crystal ground truth:
  0 crystal | 1 chai1 | 2 boltz2 | 3 boltz2_msa | 4 smina
Pose axis best-first up to MAX_POSE: crystal 1, boltz 1, boltz_msa 1, chai 5,
smina 5 (pooled across pocket files, global top-5 by minimizedAffinity, most
negative first).

int16 fixed-point at 0.01 A after mean-centering; centroid float32 kept.

Atom correspondence: a reference molecule is built from the ligand SMILES in
DEPOSIT ORDER -- largest '.'-separated fragment, RDKit input order, no
canonicalisation -- which is the ordering deposit_v3's label tables index with
`atom_idx`. Each teacher pose is matched to it; SYMMETRY-equivalent matches are
resolved against a per-system anchor pose by best geometric alignment, so a
given atom slot is the same physical atom across all teachers even for symmetric
ligands. (Independent per-teacher matching leaves symmetric atoms inconsistent,
which shows up as a ~1 A residual in the pose-invariant distance matrix.) That
anchor step is the one deliberate departure from the tables, which resolve
symmetry per teacher by taking the first substructure match; on ligands with a
non-trivial automorphism group the store and the tables may therefore differ by
an automorphism of the ligand graph. Those atoms are interchangeable, so this is
not a correspondence error -- see reports/atom_order_defect.md, which measures it.

--smiles-index must name the tier's OWN index: docking/output/boltz_in_index.tsv
for the corpus tier, datasets/experimental_expansion/exp_fold_index.tsv for the
crystal tier. Falling back to ref_from_crystal() for a crystal system that has a
deposited SMILES yields a connectivity-only perceived molecule whose atom order
matches neither the tables nor the other tier.

Validated on C.40516683: int16 round-trip 0.002 A; bond length 1.39 A; smina
files pocket x pose; SMILES via boltz_in_index.tsv; smina score
'minimizedAffinity' (kcal/mol, negative, lower better).
"""
import argparse, collections, csv, glob, json, os
import numpy as np
import h5py
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SCALE = 100
SHELL = 15.0
MAX_POSE = 5
LIG_NAMES = ("LIG", "LIG1", "LIG2", "UNL")
TEACHERS = ("crystal", "chai1", "boltz2", "boltz2_msa", "smina")
TIDX = {t: i for i, t in enumerate(TEACHERS)}
POSE_CAP = {"crystal": 1, "chai1": 5, "boltz2": 1, "boltz2_msa": 1, "smina": 5}
SCORE_KIND = {"crystal": "none|na", "chai1": "chai_conf|higher",
              "boltz2": "none|na", "boltz2_msa": "none|na",
              "smina": "minimizedAffinity_kcalmol|lower"}

def norm_elem(e):
    """PDB/CIF often uppercase element symbols (CL, BR, NA); RDKit needs proper
    case (Cl, Br, Na). Normalize two-letter symbols to Title case."""
    e = e.strip()
    if len(e) == 2:
        return e[0].upper() + e[1].lower()
    return e.upper() if len(e) == 1 else e

def load_smiles(tsv):
    """system_id -> SMILES. The corpus index keys on 'name'; the crystal-tier
    index (experimental_expansion/exp_fold_index.tsv) keys on 'system_id'."""
    out = {}
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        k = r.get("name") or r.get("system_id")
        if k:
            out[k] = r["smiles"]
    return out

def ref_from_smiles(smiles):
    """Reference molecule in DEPOSIT ATOM ORDER: the heavy atoms of the RDKit
    molecule parsed from the largest '.'-separated SMILES fragment, in RDKit
    *input* order, with no canonicalisation.

    This is the ordering the deposited label tables index with `atom_idx`
    (deposit_v3/FIELDS.md, field `atom_idx`), so a store built on it shares one
    atom axis with the tables and `atom_idx` can be used directly as a
    coordinate index.

    Builds up to 2026-08-25 additionally applied
    `CanonicalRankAtoms(breakTies=True)` + `RenumberAtoms` here. Canonical rank
    is a perfectly well-defined ordering, but it is NOT the deposited one: it
    permuted every ligand against the tables (mean d_ca error 2.4-3.1 A, with
    only 1-3 atoms of ~35 left fixed). See reports/atom_order_defect.md.

    The fragment split reproduces the label extractors' expression verbatim
    (`max(smiles.split('.'), key=len)`) so that store and tables agree
    atom-for-atom. CXSMILES coordinate blocks can themselves contain '.', which
    makes that naive split unparseable; where it fails the split is retried on
    the core SMILES only. That branch fires exactly where the extractors also
    failed, i.e. on systems with no table rows, so it cannot desynchronise the
    two artifacts.
    """
    if not smiles:
        return None
    m = Chem.MolFromSmiles(max(smiles.split("."), key=len))
    if m is None:
        core = smiles.split(" |", 1)[0]
        m = Chem.MolFromSmiles(max(core.split("."), key=len))
    if m is None:
        return None
    return Chem.RemoveHs(m)

def ref_from_crystal(crystal_cif, ccd):
    """Reference mol for a crystal system from its own ligand: read the ligand
    heavy atoms, build a mol, perceive bonds, get canonical SMILES, rebuild an
    ordered reference. Crystal ligands are not in the corpus SMILES TSV."""
    sym, xyz = lig_from_cif(crystal_cif, ccd)
    if not sym:
        return None
    rw = Chem.RWMol()
    for e in sym:
        rw.AddAtom(Chem.Atom(e))
    conf = Chem.Conformer(len(sym))
    for i,(x,y,z) in enumerate(xyz):
        conf.SetAtomPosition(i,(float(x),float(y),float(z)))
    m = rw.GetMol(); m.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineConnectivity(m)
        smi = Chem.MolToSmiles(m)
        ref = ref_from_smiles(smi)
        return ref
    except Exception:
        return None

def best_sym_match(ref, pose_xyz, sym_mol, anchor=None):
    """pose coords reordered to ref; among symmetry-equivalent matches pick the
    one best aligned (centered) to anchor, else the first match."""
    matches = sym_mol.GetSubstructMatches(ref, uniquify=False)
    if not matches:
        return None
    if anchor is None:
        return pose_xyz[list(matches[0])]
    ac = anchor - anchor.mean(0)
    best, br = None, 1e18
    for m in matches:
        cand = pose_xyz[list(m)]
        d = (cand - cand.mean(0)) - ac
        r = float((d * d).sum(1).mean())
        if r < br:
            br, best = r, cand
    return best

def cif_pose_to_ref(sym, xyz, ref, anchor=None):
    if len(sym) != ref.GetNumAtoms():
        return None
    rw = Chem.RWMol()
    for e in sym:
        rw.AddAtom(Chem.Atom(e))
    conf = Chem.Conformer(len(sym))
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    m = rw.GetMol(); m.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineConnectivity(m)
        m = AllChem.AssignBondOrdersFromTemplate(ref, m)
    except Exception:
        return None
    return best_sym_match(ref, xyz, m, anchor)

def sdf_pose_to_ref(mol, xyz, ref, anchor=None):
    try:
        m = AllChem.AssignBondOrdersFromTemplate(ref, mol)
    except Exception:
        m = mol
    return best_sym_match(ref, xyz, m, anchor)

def sdf_models(path):
    out = []
    for m in Chem.SDMolSupplier(path, removeHs=True, sanitize=True):
        if m is None:
            continue
        xyz = m.GetConformer().GetPositions().astype(np.float32)
        aff = float(m.GetProp("minimizedAffinity")) if m.HasProp("minimizedAffinity") else None
        out.append((m, xyz, aff))
    return out

def lig_from_cif(path, ccd=None):
    st = gemmi.read_structure(path)
    names = (ccd,) if ccd else LIG_NAMES
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in names:
                    sym = [norm_elem(a.element.name) for a in res if a.element.name != "H"]
                    xyz = [(a.pos.x, a.pos.y, a.pos.z) for a in res if a.element.name != "H"]
                    if xyz:
                        return sym, np.asarray(xyz, np.float32)
        break
    return [], np.zeros((0, 3), np.float32)

def lig_from_pdb(path):
    sym, xyz = [], []
    for ln in open(path):
        if ln.startswith("HETATM"):
            el = norm_elem(ln[76:78].strip() or ln[12:14].strip())
            if el == "H":
                continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))); sym.append(el)
    return sym, np.asarray(xyz, np.float32).reshape(-1, 3)

# OXT is the terminal carboxyl oxygen and is backbone, not side chain. Without
# it here the C-terminal residue of every chain gets a side-chain centroid
# displaced by a median 0.83 A, and a C-terminal glycine acquires a spurious
# one-atom "side chain" 2.4 A from its CA with sc_valid wrongly True.
BACKBONE = ("N", "CA", "C", "O", "OXT")

# Residue gate, identical to the one the label extractor and build_chain_schema.py
# use: the standard twenty, a CA, and at least one heavy atom. Without the name
# check a calcium ion, whose single atom is named CA, is recorded as a residue,
# and the store's residue enumeration diverges from res_row in the tables.
AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR "
         "TRP TYR VAL".split())

def protein_atoms(path):
    """Per residue: the CA, a side-chain centroid, and the backbone carbonyl
    carbon, all full-protein and aligned with one another by position. Plus the
    all-atom heavy-atom list with its dense residue index."""
    st = gemmi.read_structure(path)
    ca, rt, ax, ar, ae = [], [], [], [], []
    sc, sv, pc = [], [], []
    ri = 0
    for chain in st[0]:
        for res in chain:
            if res.name not in AA:
                continue
            c = res.find_atom("CA", "*")
            if c is None:
                continue
            if not any(a.element.name not in ("H", "D") for a in res):
                continue
            cap = (c.pos.x, c.pos.y, c.pos.z)
            ca.append(cap); rt.append(res.name)
            heavy, side = [], []
            for a in res:
                if a.element.name in ("H", "D"):
                    continue
                p = (a.pos.x, a.pos.y, a.pos.z)
                ax.append(p); ar.append(ri); ae.append(a.element.name)
                heavy.append(p)
                if a.name not in BACKBONE:
                    side.append(p)
            # Side-chain bead. Glycine has no side-chain heavy atom, so it falls
            # back to the residue centroid over N, CA, C, O rather than to the CA
            # itself: a bead coincident with a stored coordinate would give a
            # distance-based consumer a degenerate zero-length pair. This departs
            # from cg2all and Rosetta, which place it at the CA.
            if side:
                sc.append(np.mean(side, axis=0)); sv.append(True)
            else:
                sc.append(np.mean(heavy, axis=0) if heavy else cap); sv.append(False)
            # Peptide bead. The backbone carbonyl carbon, a real atom present in
            # every residue including the last, so no mask and no chain-break
            # case. Carries chain direction and peptide-plane orientation.
            cc = res.find_atom("C", "*")
            pc.append((cc.pos.x, cc.pos.y, cc.pos.z) if cc is not None else cap)
            ri += 1
    return (np.asarray(ca, np.float32), rt,
            np.asarray(ax, np.float32).reshape(-1, 3), np.asarray(ar, np.int32), ae,
            np.asarray(sc, np.float32).reshape(-1, 3), np.asarray(sv, bool),
            np.asarray(pc, np.float32).reshape(-1, 3))

def q(xyz, c):
    return np.clip(np.round((xyz - c) * SCALE), -32768, 32767).astype(np.int16)

def gather_roots(sid, A):
    """Explicit per-teacher root list from --roots. Each root is
    {"tpl": path template with {sid}/{pdb}/{k}} (optionally "k": n, to expand
    {k} over range(n)) or {"glob": pattern}. Roots are tried in order and the
    first that yields any file on disk wins for that teacher. Unlike the legacy
    recursive '{sid}*_model_0.pdb' glob these are exact paths, so a numeric
    corpus id cannot pick up a longer id that merely starts with it."""
    pdb = sid.split("_")[0]; s = {}
    for t, roots in A._roots.items():
        for r in roots:
            if "glob" in r:
                f = sorted(glob.glob(r["glob"].format(sid=sid, pdb=pdb)))
            elif "k" in r:
                f = [p for p in (r["tpl"].format(sid=sid, pdb=pdb, k=i) for i in range(r["k"]))
                     if os.path.exists(p)]
            else:
                p = r["tpl"].format(sid=sid, pdb=pdb)
                f = [p] if os.path.exists(p) else []
            if f:
                s[t] = f[:MAX_POSE] if t in ("chai1",) else f
                break
    return s

def gather(sid, A):
    if getattr(A, "_roots", None):
        return gather_roots(sid, A)
    pdb = sid.split("_")[0]; s = {}
    c = sorted(glob.glob(f"{A.chai}/{sid}/*.cif")) or sorted(glob.glob(f"{A.chai_crystal}/{sid}/*.cif"))
    if c: s["chai1"] = c[:MAX_POSE]
    b = (glob.glob(f"{A.boltz}/**/{sid}_model_0.cif", recursive=True) or
         glob.glob(f"{A.boltz}/**/{sid}*_model_0.pdb", recursive=True))
    if b: s["boltz2"] = [b[0]]
    bm = glob.glob(f"{A.boltz_msa}/**/{sid}*_model_0.pdb", recursive=True)
    if bm: s["boltz2_msa"] = [bm[0]]
    sm = sorted(glob.glob(f"{A.smina}/{sid}_*.sdf"))
    if sm: s["smina"] = sm
    cr = glob.glob(f"{A.crystal}/{pdb}*.cif")
    if cr: s["crystal"] = [cr[0]]
    return s

def ligand_poses(t, paths, ref, ccd, anchor):
    """list of (coords_in_ref_order, score); symmetry resolved vs anchor."""
    n = ref.GetNumAtoms(); cap = POSE_CAP[t]; out = []
    if t == "smina":
        pooled = []
        for p in paths:
            for mol, xyz, aff in sdf_models(p):
                rx = sdf_pose_to_ref(mol, xyz, ref, anchor)
                if rx is not None:
                    pooled.append((rx, aff))
        pooled.sort(key=lambda z: z[1] if z[1] is not None else 1e9)
        out = pooled[:cap]
    elif t == "chai1":
        for p in paths[:cap]:
            sym, xyz = lig_from_cif(p, None)
            rx = cif_pose_to_ref(sym, xyz, ref, anchor)
            if rx is not None:
                out.append((rx, None))
    elif t == "crystal":
        sym, xyz = lig_from_cif(paths[0], ccd)
        rx = cif_pose_to_ref(sym, xyz, ref, anchor)
        if rx is not None:
            out.append((rx, None))
    else:
        # boltz2 / boltz2_msa: deposited arms are PDB, crystal_fold is mmCIF
        # (ligand residue LIG1, so read it by LIG_NAMES rather than by CCD).
        sym, xyz = (lig_from_cif(paths[0], None) if paths[0].endswith(".cif")
                    else lig_from_pdb(paths[0]))
        rx = cif_pose_to_ref(sym, xyz, ref, anchor)
        if rx is not None:
            out.append((rx, None))
    return out

def first_anchor(ref, src, ccd, A):
    """Establish a per-system anchor pose (reference-ordered, symmetry arbitrary)
    from the highest-priority available teacher, to disambiguate symmetry."""
    for t in ("crystal", "chai1", "boltz2", "smina", "boltz2_msa"):
        if t not in src:
            continue
        paths = src[t]
        if t == "smina":
            ms = sdf_models(paths[0])
            if ms:
                rx = sdf_pose_to_ref(ms[0][0], ms[0][1], ref, None)
                if rx is not None:
                    return rx
        elif t in ("boltz2", "boltz2_msa"):
            sym, xyz = (lig_from_cif(paths[0], None) if paths[0].endswith(".cif")
                        else lig_from_pdb(paths[0]))
            rx = cif_pose_to_ref(sym, xyz, ref, None)
            if rx is not None:
                return rx
        else:
            sym, xyz = lig_from_cif(paths[0], ccd if t == "crystal" else None)
            rx = cif_pose_to_ref(sym, xyz, ref, None)
            if rx is not None:
                return rx
    return None

def build_system(sid, smiles, ccd, A, out):
    """Returns None on success, else a short reason string naming the point of
    failure. The five early exits are distinct causes and are reported as such:
    no_reference (no usable reference molecule), no_sources (no teacher file on
    disk), no_anchor (no teacher pose could be matched to the reference to fix
    the symmetry frame), no_poses (an anchor existed but no teacher yielded a
    matched pose) and no_valid_poses (poses matched but none had the reference
    atom count)."""
    ref = ref_from_smiles(smiles) if smiles else None
    if ref is None:
        # crystal tier: no TSV smiles; derive reference from the crystal ligand
        cr = gather(sid, A).get("crystal")
        if cr:
            ref = ref_from_crystal(cr[0], ccd)
    if ref is None:
        return "no_reference"
    n = ref.GetNumAtoms()
    src = gather(sid, A)
    if not src:
        return "no_sources"
    anchor = first_anchor(ref, src, ccd, A)
    if anchor is None:
        return "no_anchor"

    per = {}
    for t, paths in src.items():
        p = ligand_poses(t, paths, ref, ccd, anchor)
        if p:
            per[t] = p
    if not per:
        return "no_poses"
    centroid = next(iter(per.values()))[0][0].mean(0).astype(np.float32)

    nt = len(TEACHERS)
    lig = np.zeros((nt, MAX_POSE, n, 3), np.int16)
    val = np.zeros((nt, MAX_POSE), bool)
    sco = np.full((nt, MAX_POSE), np.nan, np.float32)
    for t, poses in per.items():
        ti = TIDX[t]
        for pi, (xyz, sc) in enumerate(poses[:MAX_POSE]):
            if xyz.shape[0] != n:
                continue
            lig[ti, pi] = q(xyz, centroid); val[ti, pi] = True
            if sc is not None:
                sco[ti, pi] = sc
    if not val.any():
        return "no_valid_poses"

    g = out.create_group(sid)
    g.attrs["smiles"] = smiles or ""
    g.attrs["ccd"] = ccd or ""
    g.attrs["centroid_xyz"] = centroid
    g.attrs["scale"] = SCALE
    g.attrs["teachers"] = np.array(TEACHERS, dtype="S12")
    g.attrs["score_kind"] = np.array([SCORE_KIND[t] for t in TEACHERS], dtype="S40")
    g.create_dataset("ligand_coords", data=lig, compression="gzip")
    g.create_dataset("ligand_valid", data=val)
    g.create_dataset("pose_score", data=sco)

    for t, paths in src.items():
        pth = paths[0]
        if not pth.endswith((".cif", ".pdb")):
            continue
        # A receptor is written when a pose exists in ITS frame. For a predicted
        # teacher that means the teacher's own pose, since its receptor and
        # ligand are one prediction. The deposited crystal structure is
        # different: it is the frame of the crystal ligand AND of the docking
        # arm, which docks into a receptor derived from it. So the crystal
        # receptor is written whenever any pose exists, not only when the
        # crystal ligand itself parsed. Without this, 1,341 of 17,368 crystal
        # systems carry a ligand and no experimental receptor.
        if t != "crystal" and t not in per:
            continue
        try:
            ca, rt, ax, ar, ae, sc, sv, pc = protein_atoms(pth)
        except Exception:
            continue
        if not len(ca):
            continue
        gt = g.create_group(f"protein/{t}")
        gt.create_dataset("ca", data=q(ca, centroid), compression="gzip")
        gt.create_dataset("res_types", data=np.array(rt, dtype="S3"), compression="gzip")
        gt.create_dataset("sc_centroid", data=q(sc, centroid), compression="gzip")
        gt.create_dataset("sc_valid", data=sv, compression="gzip")
        gt.create_dataset("pep_c", data=q(pc, centroid), compression="gzip")
        lp = per[t][0][0] if t in per else next(iter(per.values()))[0][0]
        if len(ax):
            d = np.linalg.norm(ax[:, None, :] - lp[None, :, :], axis=2).min(1)
            keep = np.where(d <= SHELL)[0]
            if keep.size:
                gt.create_dataset("shell_coords", data=q(ax[keep], centroid), compression="gzip")
                gt.create_dataset("shell_res_index", data=ar[keep])
                gt.create_dataset("shell_elem", data=np.array([ae[i] for i in keep], dtype="S2"))
    # Receptor fallback. smina docks into an AlphaFold model and writes only an
    # SDF, so the loop above produces no protein group for it. Read that same
    # model here: the pose is in its frame, so no superposition is needed.
    if getattr(A, "af", "") and "smina" in per and "protein/smina" not in g:
        acc = getattr(A, "_s2a", {}).get(sid)
        fp = f"{A.af}/AF-{acc}-F1-model_v6.pdb" if acc else None
        if fp and os.path.exists(fp):
            try:
                ca, rt, ax, ar, ae, sc, sv, pc = protein_atoms(fp)
            except Exception:
                ca = []
            if len(ca):
                gt = g.create_group("protein/smina")
                gt.create_dataset("ca", data=q(ca, centroid), compression="gzip")
                gt.create_dataset("res_types", data=np.array(rt, dtype="S3"),
                                  compression="gzip")
                gt.create_dataset("sc_centroid", data=q(sc, centroid), compression="gzip")
                gt.create_dataset("sc_valid", data=sv, compression="gzip")
                gt.create_dataset("pep_c", data=q(pc, centroid), compression="gzip")
                lp = per["smina"][0][0]
                if len(ax):
                    d = np.linalg.norm(ax[:, None, :] - lp[None, :, :], axis=2).min(1)
                    keep = np.where(d <= SHELL)[0]
                    if keep.size:
                        gt.create_dataset("shell_coords", data=q(ax[keep], centroid),
                                          compression="gzip")
                        gt.create_dataset("shell_res_index", data=ar[keep])
                        gt.create_dataset("shell_elem",
                                          data=np.array([ae[i] for i in keep], dtype="S2"))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--smiles-index", required=True)
    ap.add_argument("--chai", default="/workspace/chai_all")
    ap.add_argument("--chai-crystal", default="/workspace/docking/output/chai_out_exp")
    ap.add_argument("--boltz", default="/workspace/docking/output/boltz_out")
    ap.add_argument("--boltz-msa", default="/workspace/docking/output/boltz_out_msapilot")
    ap.add_argument("--crystal", default="/workspace/datasets/experimental_expansion/structures")
    ap.add_argument("--smina", default="/workspace/docking/output/smina/poses")
    ap.add_argument("--af", default="")
    ap.add_argument("--sys2acc", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    # Opt-in extras. With none of these given the build is byte-for-byte the
    # legacy one; they exist so the v2 stores can name their roots explicitly
    # and resolve corpus SMILES for ids absent from the TSV.
    ap.add_argument("--roots", default=None,
                    help="JSON: {teacher: [{tpl|glob, k?}, ...]}; overrides the default globs")
    ap.add_argument("--smiles-json", default=None,
                    help="JSON inchikey->smiles, used with --inchikey-map")
    ap.add_argument("--inchikey-map", default=None,
                    help="JSON system_id->inchikey; fallback when --smiles-index misses the id")
    ap.add_argument("--reasons", default=None,
                    help="TSV of per-system skip reasons (default: <out>.reasons.tsv)")
    A = ap.parse_args()
    A._roots = json.load(open(A.roots)) if A.roots else None
    A._s2a = json.load(open(A.sys2acc)) if A.sys2acc else {}
    smi = load_smiles(A.smiles_index)
    if A.smiles_json and A.inchikey_map:
        ikmap = json.load(open(A.inchikey_map)); iksmi = json.load(open(A.smiles_json))
        for sid, ik in ikmap.items():
            if sid not in smi and ik in iksmi:
                smi[sid] = iksmi[ik]
    ids = [x.strip() for x in open(A.ids) if x.strip()]
    if A.limit:
        ids = ids[: A.limit]
    reasons = []
    w = s = 0
    with h5py.File(A.out, "w") as out:
        out.attrs["schema_version"] = "2.1"
        out.attrs["scale"] = SCALE
        out.attrs["shell_angstrom"] = SHELL
        out.attrs["teachers"] = np.array(TEACHERS, dtype="S12")
        for sid in ids:
            smiles = smi.get(sid)
            parts = sid.split("_")
            ccd = parts[1] if len(parts) >= 3 else None
            try:
                reason = build_system(sid, smiles, ccd, A, out)
            except Exception as e:
                reason = f"exception:{type(e).__name__}"
                print(f"  {sid}: {type(e).__name__} {e}")
            if reason is None:
                w += 1
            else:
                s += 1; reasons.append((sid, reason))
    rp = A.reasons or (A.out + ".reasons.tsv")
    with open(rp, "w") as fh:
        fh.write("system_id\treason\n")
        for sid, reason in reasons:
            fh.write(f"{sid}\t{reason}\n")
    counts = collections.Counter(r for _, r in reasons)
    with open(rp.replace(".tsv", ".json"), "w") as fh:
        json.dump({"out": A.out, "ids": A.ids, "attempted": len(ids),
                   "written": w, "skipped": s, "reasons": dict(counts)}, fh, indent=1)
    print(f"written {w} skipped {s} -> {A.out}")
    if counts:
        print("  reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

if __name__ == "__main__":
    main()
