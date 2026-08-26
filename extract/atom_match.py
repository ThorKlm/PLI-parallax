"""Map docked-pose atom indices onto the reference (model) atom ordering.

Distance labels are per (residue, ligand_atom), so the atom index in a pose
must be expressed in the SAME ordering used to build ligand_atoms.npz.
SMINA/Vina round-trip through PDBQT and reorder / re-protonate atoms; Boltz-2
has its own ordering. We recover the permutation via the original SMILES as a
bond-order template, which is exactly the AssignBondOrdersFromTemplate use case.

CAVEAT: the "reference" mol below must be constructed identically to the
training featurizer. If ligand_atoms.npz was built from a canonicalised SMILES
or with a specific Hs policy, reuse that code to make ref_mol; do not assume
this loader matches it.
"""
from rdkit import Chem
from rdkit.Chem import AllChem

def ref_mol_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    return Chem.RemoveHs(m)  # heavy-atom reference ordering

def pose_mol_from_sdf(path):
    m = next(iter(Chem.SDMolSupplier(path, sanitize=False, removeHs=True)))
    if m is None:
        raise ValueError(f"no mol in {path}")
    return m

def map_pose_to_ref(ref_mol, pose_mol):
    """Return (perm, ambiguous): perm[pose_i] = ref atom index.
    Bond orders in the pose are reassigned from the reference template, then a
    substructure match yields the index correspondence. Topological symmetry
    (e.g. equivalent ring atoms) is interchangeable for distance labels, so an
    ambiguous flag is returned rather than failing.
    """
    pose_h = Chem.RemoveHs(pose_mol, sanitize=False)
    try:
        pose_fixed = AllChem.AssignBondOrdersFromTemplate(ref_mol, pose_h)
    except Exception:
        pose_fixed = pose_h  # fall back to connectivity-only matching below
    matches = ref_mol.GetSubstructMatches(pose_fixed, uniquify=False,
                                          useChirality=False, maxMatches=64)
    if not matches:
        # relax bond orders entirely: compare single-bond skeletons
        r = _skeleton(ref_mol); p = _skeleton(pose_fixed)
        matches = r.GetSubstructMatches(p, uniquify=False, maxMatches=64)
        if not matches:
            raise ValueError("no atom-graph match between pose and reference")
    if ref_mol.GetNumAtoms() != pose_fixed.GetNumAtoms():
        raise ValueError("heavy-atom count mismatch; wrong compound or stripped atoms")
    return list(matches[0]), len(matches) > 1

def _skeleton(m):
    em = Chem.RWMol(m)
    for b in em.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
    for a in em.GetAtoms():
        a.SetFormalCharge(0); a.SetNoImplicit(True)
    return em.GetMol()

def _selftest():
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    ref = ref_mol_from_smiles(smi)
    pose = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(pose, randomSeed=0)
    # scramble atom order to emulate a PDBQT round-trip
    order = list(range(pose.GetNumAtoms()))[::-1]
    pose = Chem.RenumberAtoms(pose, order)
    perm, amb = map_pose_to_ref(ref, Chem.RemoveHs(pose))
    assert sorted(perm) == list(range(ref.GetNumAtoms())), "perm not a bijection"
    print("selftest ok: n_atoms", ref.GetNumAtoms(), "ambiguous", amb, "perm", perm)

if __name__ == "__main__":
    _selftest()
