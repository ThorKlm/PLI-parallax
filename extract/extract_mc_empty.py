"""Extract residue-ligand distance labels from the multimer empty-MSA Boltz-2 run.

Adapted from extract_exp_msa.py. Two things differ from that script:

  1. Input is Boltz mmCIF (<system_id>/<system_id>_model_0.cif) under
     multimer_empty_msa/, not the single-chain PDB of the msa run.
  2. These are true multimers: Boltz emits one chain per crystal chain and may
     emit them in a permuted order (manifest flag boltz_order_is_identity).
     The deposited res_row enumerates standard-AA CA-bearing residues across all
     chains in CRYSTAL mmCIF chain order, so every predicted residue is remapped
     onto that enumeration via the chain schema's chain_res_offsets before the
     contacts are written. Chain identity is resolved by the mmCIF chain NAME
     (Boltz preserves the input chain id even when it reorders), and every chain
     is checked against the schema's per-chain length and sequence before any
     label is emitted -- a system that fails is recorded as a failure, never
     silently written with a wrong enumeration.

Output teacher tag is boltz2_mc_empty so it cannot collide with the deposited
boltz2 / boltz2_msa / boltz2_r3empty labels.

Run: python extract_mc_empty.py <shard> <nshard>
"""
import sys, os, json
import numpy as np, pandas as pd, gemmi
from rdkit import Chem
sys.path.insert(0, "/workspace")
from run_smina_labels import contacts
from atom_match import map_pose_to_ref

PRED = "/workspace/multimer_empty_msa/out/boltz_results_mc_empty_core/predictions"
IDX  = "/workspace/datasets/experimental_expansion/exp_fold_index_mc.parquet"
SCH  = "/workspace/reports/crystal_chain_schema.parquet"
STAT = "/workspace/reports/multimer_empty_msa_status.tsv"
OUT  = "/workspace/datasets/experimental_expansion/teacher_labels"
SRC  = "boltz2_mc_empty"

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
AA = set(AA3)


class AlignError(Exception):
    """Predicted chain layout does not match the crystal schema."""


def parse_pred(path):
    """Parse a Boltz multimer mmCIF.

    Returns (chains, lig_xyz, elements) where chains is a list of
    (chain_name, seq1, ca_list, heavy_list) in mmCIF iteration order. Only
    standard-AA CA-bearing residues are kept, matching the deposited
    enumeration; everything else is treated as ligand.
    """
    m = gemmi.read_structure(path)[0]
    chains, lig, el = [], [], []
    for ch in m:
        seq, ca, heavy = [], [], []
        for res in ch:
            if res.name in AA:
                hv, cx = [], None
                for at in res:
                    if at.element.name in ("H", "D"): continue
                    p = (at.pos.x, at.pos.y, at.pos.z); hv.append(p)
                    if at.name == "CA": cx = p
                if cx is None or not hv: continue
                seq.append(AA3[res.name]); ca.append(cx)
                heavy.append(np.array(hv, np.float32))
            else:
                for at in res:
                    if at.element.name in ("H", "D"): continue
                    lig.append([at.pos.x, at.pos.y, at.pos.z]); el.append(at.element.name)
        if seq: chains.append((ch.name, "".join(seq), ca, heavy))
    return chains, np.array(lig, np.float32), el


def crystal_remap(chains, emitted_ids, counts, offsets, seqs):
    """Flatten predicted chains into crystal res_row order.

    Returns (ca, heavy, remap) where remap[pred_row] is the deposited res_row.
    Raises AlignError if the predicted chain set, any chain length, or any chain
    sequence disagrees with the crystal schema.
    """
    pos = {c: i for i, c in enumerate(emitted_ids)}
    if len(chains) != len(emitted_ids):
        raise AlignError(f"chain count {len(chains)} != schema {len(emitted_ids)}")
    seen = set()
    for name, _, _, _ in chains:
        if name not in pos: raise AlignError(f"chain '{name}' not in schema {emitted_ids}")
        if name in seen:    raise AlignError(f"duplicate chain '{name}'")
        seen.add(name)
    ca, heavy, remap = [], [], []
    for name, seq, c, h in chains:
        i = pos[name]
        if len(seq) != counts[i]:
            raise AlignError(f"chain '{name}' has {len(seq)} res, schema says {counts[i]}")
        if seq != seqs[i]:
            raise AlignError(f"chain '{name}' sequence differs from crystal schema")
        ca += c; heavy += h
        remap.append(np.arange(counts[i], dtype=np.int64) + offsets[i])
    remap = np.concatenate(remap) if remap else np.zeros(0, np.int64)
    n = int(sum(counts))
    if len(remap) != n or not np.array_equal(np.sort(remap), np.arange(n)):
        raise AlignError("remap is not a permutation of the crystal enumeration")
    return np.array(ca, np.float32), heavy, remap


def canon(lig, el, smi):
    """Reorder pose atoms onto the canonical (SMILES) heavy-atom ordering."""
    frag = max(smi.split('.'), key=len); ref = Chem.MolFromSmiles(frag)
    if ref is None: raise ValueError("bad smiles")
    rel = [a.GetSymbol() for a in ref.GetAtoms()]
    if el == rel and len(lig) == len(rel): return lig, ref.GetNumAtoms()
    if sorted(el) != sorted(rel):
        raise ValueError(f"elem mismatch pose={len(el)} ref={len(rel)}")
    rw = Chem.RWMol()
    for e in el: rw.AddAtom(Chem.Atom(e))
    conf = Chem.Conformer(len(lig))
    for i, (x, y, z) in enumerate(lig): conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    pose = rw.GetMol(); pose.AddConformer(conf)
    perm, _ = map_pose_to_ref(ref, pose)
    out = np.zeros((ref.GetNumAtoms(), 3), np.float32)
    for pi, ri in enumerate(perm): out[ri] = lig[pi]
    return out, ref.GetNumAtoms()


def main():
    shard = int(sys.argv[1]); nshard = int(sys.argv[2])
    idx = pd.read_parquet(IDX).set_index("system_id")
    sch = pd.read_parquet(SCH).set_index("system_id")
    sids = sorted(l.split("\t")[0] for l in open(STAT).read().splitlines()[1:])
    mine = [s for i, s in enumerate(sids) if i % nshard == shard]

    SID, NLIG, NRES, REORD = [], [], [], []
    off = [0]; C_res, C_atom, C_dca, C_dmin = [], [], [], []
    done = miss = fail = 0; failures = []
    for sid in mine:
        path = f"{PRED}/{sid}/{sid}_model_0.cif"
        if not os.path.exists(path):
            miss += 1; failures.append((sid, "missing", "no _model_0.cif")); continue
        try:
            row = idx.loc[sid]; srow = sch.loc[sid]
        except KeyError:
            fail += 1; failures.append((sid, "no_manifest", "absent from index/schema")); continue
        try:
            chains, lig, el = parse_pred(path)
            if len(lig) == 0: raise ValueError("no ligand atoms in prediction")
            ca, heavy, remap = crystal_remap(
                chains, list(row.emitted_chain_ids), list(srow.chain_res_counts),
                list(srow.chain_res_offsets), list(srow.chain_seqs))
            ligc, n_lig = canon(lig, el, row.smiles)
            rr, ai, dca, dmn = contacts(ligc, ca, heavy)
        except AlignError as e:
            fail += 1; failures.append((sid, "align", str(e))); continue
        except Exception as e:
            fail += 1; failures.append((sid, type(e).__name__, str(e)[:200])); continue
        # pred rows -> deposited crystal enumeration, then keep res_row ascending
        rr = remap[np.asarray(rr, np.int64)] if rr else np.zeros(0, np.int64)
        ai = np.asarray(ai, np.int64); dca = np.asarray(dca); dmn = np.asarray(dmn)
        o = np.lexsort((ai, rr))
        C_res += rr[o].tolist(); C_atom += ai[o].tolist()
        C_dca += dca[o].tolist(); C_dmin += dmn[o].tolist()
        SID.append(sid); NLIG.append(n_lig); NRES.append(int(sum(srow.chain_res_counts)))
        REORD.append(not bool(row.boltz_order_is_identity))
        off.append(len(C_res)); done += 1

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/exp_{SRC}_{shard}.npz",
        system_id=np.array(SID), n_lig_atoms=np.array(NLIG, np.int32),
        n_res=np.array(NRES, np.int32), boltz_reordered=np.array(REORD, bool),
        contact_offsets=np.array(off, np.int64), res_row=np.array(C_res, np.int32),
        atom_idx=np.array(C_atom, np.int16), d_ca=np.array(C_dca, np.float16),
        d_min=np.array(C_dmin, np.float16), cutoff=np.float32(15.0),
        source=np.array(SRC))
    with open(f"{OUT}/exp_{SRC}_{shard}.failures.json", "w") as f:
        json.dump([{"system_id": s, "reason": r, "detail": d} for s, r, d in failures], f, indent=1)
    print(f"{SRC} shard {shard}: done={done} miss={miss} fail={fail} "
          f"reordered={sum(REORD)} contacts={len(C_res)}")


if __name__ == "__main__": main()
