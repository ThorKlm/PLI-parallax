"""Dock manifest pockets with SMINA and emit residue-atom distance labels,
checkpointed and resumable. Per pair: dock every candidate pocket, keep the
lowest-affinity pose, remap its atoms to the canonical head ordering
(atom_match), then record for every residue within CUTOFF of any ligand atom
both the CA-to-atom and min-heavy-atom distance. Residue rows index proteins_v10
(PDB CA order). The CSR npz is flushed every FLUSH_EVERY pairs and a restart
resumes by skipping pairs already present. Schema matches boltz_labels.
Run: python run_smina_labels.py /workspace/docking/input \
         /workspace/docking/output/manifest_shuf.tsv \
         /workspace/docking/output/smina --cpu 30 --exhaustiveness 8
"""
import argparse, csv, os, subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from atom_match import map_pose_to_ref, ref_mol_from_smiles

CUTOFF = 15.0
FLUSH_EVERY = 5

def sh(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def prep_receptor(pdb, out_pdbqt):
    if not os.path.exists(out_pdbqt):
        sh(["obabel", pdb, "-O", out_pdbqt, "-xr", "-p", "7.4"])
    return out_pdbqt

def prep_ligand(smiles, out_sdf):
    ref = ref_mol_from_smiles(smiles)
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=0) != 0:
        AllChem.EmbedMolecule(m, randomSeed=0, useRandomCoords=True)
    try: AllChem.MMFFOptimizeMolecule(m)
    except Exception: pass
    Chem.SDWriter(out_sdf).write(m)
    return out_sdf, ref

def dock(rec, lig_sdf, c, s, out_sdf, cpu, ex, nm):
    sh(["smina", "--receptor", rec, "--ligand", lig_sdf,
        "--center_x", f"{c[0]}", "--center_y", f"{c[1]}", "--center_z", f"{c[2]}",
        "--size_x", f"{s[0]}", "--size_y", f"{s[1]}", "--size_z", f"{s[2]}",
        "--exhaustiveness", str(ex), "--num_modes", str(nm), "--cpu", str(cpu),
        "--seed", "0", "--out", out_sdf, "--quiet"])
    poses = [p for p in Chem.SDMolSupplier(out_sdf, removeHs=False, sanitize=False) if p]
    if not poses: return None, None
    return poses[0], float(poses[0].GetProp("minimizedAffinity"))

def parse_protein(pdb):
    res, idx = [], {}
    for ln in open(pdb):
        if not ln.startswith("ATOM"): continue
        if ln[16] not in (" ", "A"): continue
        name = ln[12:16].strip(); elem = (ln[76:78].strip() or name[:1]).upper()
        if elem == "H": continue
        key = (ln[21], ln[22:26].strip())
        if key not in idx:
            idx[key] = len(res); res.append({"ca": None, "heavy": []})
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        r = res[idx[key]]; r["heavy"].append(xyz)
        if name == "CA": r["ca"] = xyz
    ca = np.array([r["ca"] if r["ca"] else r["heavy"][0] for r in res], np.float32)
    heavy = [np.array(r["heavy"], np.float32) for r in res]
    return ca, heavy

def pose_coords_canonical(pose, ref):
    ph = Chem.RemoveHs(pose, sanitize=False)
    perm, _ = map_pose_to_ref(ref, ph)
    conf = ph.GetConformer()
    coords = np.zeros((ref.GetNumAtoms(), 3), np.float32)
    for pose_i, ref_i in enumerate(perm):
        p = conf.GetAtomPosition(pose_i); coords[ref_i] = (p.x, p.y, p.z)
    return coords

def contacts(lig, ca, heavy):
    rr, ai, dca, dmn = [], [], [], []
    for r in range(len(heavy)):
        dm = np.sqrt(((heavy[r][:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(0)
        hit = np.where(dm <= CUTOFF)[0]
        if hit.size == 0: continue
        dc = np.sqrt(((ca[r] - lig[hit]) ** 2).sum(-1))
        rr.extend([r] * hit.size); ai.extend(hit.tolist())
        dca.extend(dc.tolist()); dmn.extend(dm[hit].tolist())
    return rr, ai, dca, dmn

def resume(path):
    if not os.path.exists(path):
        return [], [], [], [], [0], [], [], [], [], set()
    d = np.load(path, allow_pickle=True)
    P_idx = d["pair_idx"].tolist()
    return (P_idx, d["accession"].tolist(), d["pocket_rank"].tolist(),
            d["affinity"].tolist(), d["contact_offsets"].tolist(),
            d["res_row"].tolist(), d["atom_idx"].tolist(),
            d["d_ca"].astype(np.float32).tolist(), d["d_min"].astype(np.float32).tolist(),
            set(P_idx))

def flush(path, P_idx, P_acc, P_rank, P_aff, off, C_res, C_atom, C_dca, C_dmin):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp,
        pair_idx=np.array(P_idx, np.int32), accession=np.array(P_acc),
        pocket_rank=np.array(P_rank, np.int16), affinity=np.array(P_aff, np.float32),
        contact_offsets=np.array(off, np.int64), res_row=np.array(C_res, np.int32),
        atom_idx=np.array(C_atom, np.int16), d_ca=np.array(C_dca, np.float16),
        d_min=np.array(C_dmin, np.float16), cutoff=np.float32(CUTOFF),
        source=np.array("smina"))
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("manifest"); ap.add_argument("out")
    ap.add_argument("--cpu", type=int, default=os.cpu_count())
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--num-modes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cache = os.path.join(a.out, "cache"); poses_dir = os.path.join(a.out, "poses")
    os.makedirs(cache, exist_ok=True); os.makedirs(poses_dir, exist_ok=True)
    out_npz = os.path.join(a.out, "smina_labels.npz")

    jobs = {}
    with open(a.manifest) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            jobs.setdefault(int(row["pair_idx"]), []).append(row)

    (P_idx, P_acc, P_rank, P_aff, off, C_res, C_atom, C_dca, C_dmin, done) = resume(out_npz)
    pairs = [p for p in jobs if p not in done]
    if a.limit: pairs = pairs[:a.limit]
    print(f"resuming: {len(done)} done, {len(pairs)} to do")

    rec_cache, prot_cache = {}, {}
    processed, fail = 0, 0
    for pair in pairs:
        rows = jobs[pair]; acc = rows[0]["accession"]; cid = int(rows[0]["compound_idx"])
        pdb = os.path.join(a.root, "pdbs", f"AF-{acc}-F1-model_v6.pdb")
        try:
            rec = rec_cache.get(acc) or prep_receptor(pdb, os.path.join(cache, f"{acc}.pdbqt"))
            rec_cache[acc] = rec
            lig_sdf, ref = prep_ligand(rows[0]["smiles"], os.path.join(cache, f"cmp_{cid}.sdf"))
        except Exception:
            fail += 1; continue
        best = None
        for r in rows:
            c = (r["cx"], r["cy"], r["cz"]); s = (r["sx"], r["sy"], r["sz"])
            out_sdf = os.path.join(poses_dir, f"{pair}_{r['pocket_rank']}.sdf")
            try:
                pose, aff = dock(rec, lig_sdf, c, s, out_sdf, a.cpu, a.exhaustiveness, a.num_modes)
            except Exception:
                continue
            if pose is not None and (best is None or aff < best[0]):
                best = (aff, pose, int(r["pocket_rank"]))
        if best is None: fail += 1; continue
        try:
            lig = pose_coords_canonical(best[1], ref)
        except Exception:
            fail += 1; continue
        if acc not in prot_cache: prot_cache[acc] = parse_protein(pdb)
        ca, heavy = prot_cache[acc]
        rr, ai, dca, dmn = contacts(lig, ca, heavy)
        C_res += rr; C_atom += ai; C_dca += dca; C_dmin += dmn
        P_idx.append(pair); P_acc.append(acc); P_rank.append(best[2]); P_aff.append(best[0])
        off.append(len(C_res)); processed += 1
        if processed % FLUSH_EVERY == 0:
            flush(out_npz, P_idx, P_acc, P_rank, P_aff, off, C_res, C_atom, C_dca, C_dmin)
            print(f"{processed} new pairs, {len(P_idx)} total, {len(C_res)} contacts, {fail} fail")
    flush(out_npz, P_idx, P_acc, P_rank, P_aff, off, C_res, C_atom, C_dca, C_dmin)
    print(f"done: total={len(P_idx)} new={processed} fail={fail} contacts={len(C_res)} -> {out_npz}")

if __name__ == "__main__":
    main()
