#!/usr/bin/env python3
"""Download ONLY chosen PLINDER holo systems via HTTP range-read zip extraction
(receptor.pdb + ligand SDF per system), extract per-(residue, ligand-heavy-atom)
distances from experimental holo coords (CA-to-atom + min-heavy-atom, <=15 A,
residue index in chain order), and store a CSR npz matching smina_labels.npz.
Caps total downloaded bytes near 1 GB. Read-only on inputs."""
import io, json, zipfile, collections, time
import numpy as np
import pyarrow.fs as pafs
import gemmi

CAP_BYTES = 1_000_000_000          # ~1 GB hard cap on bytes pulled from GCS
CUTOFF = 15.0
BUCKET = 'plinder/2024-06/v2/systems'

# ---- byte-counting seekable adapter over pyarrow GCS NativeFile ----
class Counter:
    total = 0
class Seekable(io.RawIOBase):
    def __init__(self, f):
        self.f = f; self._size = f.size(); self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 1: self.pos += off
        elif whence == 2: self.pos = self._size + off
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self._size - self.pos
        if self.pos >= self._size or n <= 0: return b''
        n = min(n, self._size - self.pos)
        self.f.seek(self.pos); b = self.f.read(n)
        self.pos += len(b); Counter.total += len(b)
        return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

# ---- ligand SDF: parse heavy-atom coords straight from V2000 molblock ----
def ligand_heavy_coords(sdf_bytes):
    text = sdf_bytes.decode('utf-8', 'replace')
    lines = text.splitlines()
    if len(lines) < 4: return None
    counts = lines[3]
    try:
        natoms = int(counts[0:3])
    except ValueError:
        return None
    coords = []
    for k in range(natoms):
        ln = lines[4 + k]
        x = float(ln[0:10]); y = float(ln[10:20]); z = float(ln[20:30])
        elem = ln[31:34].strip()
        if elem == 'H' or elem == 'D':       # heavy atoms only
            continue
        coords.append((x, y, z))
    if not coords: return None
    return np.asarray(coords, dtype=np.float64)

# ---- receptor.pdb: CA coords + per-residue heavy-atom coords, chain order ----
def receptor_residues(pdb_bytes):
    st = gemmi.read_pdb_string(pdb_bytes.decode('utf-8', 'replace'))
    ca = []          # (R,3) CA coords
    heavy = []       # list of (R_i, 3) arrays of residue heavy atoms
    model = st[0]
    for chain in model:
        for res in chain:
            ca_atom = None
            hv = []
            for at in res:
                el = at.element.name
                if el == 'H' or el == 'D':
                    continue
                p = at.pos
                hv.append((p.x, p.y, p.z))
                if at.name == 'CA' and el == 'C':
                    ca_atom = (p.x, p.y, p.z)
            if ca_atom is None or not hv:    # require CA (standard polymer residue)
                continue
            ca.append(ca_atom)
            heavy.append(np.asarray(hv, dtype=np.float64))
    if not ca:
        return None, None
    return np.asarray(ca, dtype=np.float64), heavy

def contacts_for_system(ca, heavy, lig):
    """Return (res_row, atom_idx, d_ca, d_min) for d_min<=CUTOFF."""
    # d_ca: (R,M)
    diff = ca[:, None, :] - lig[None, :, :]
    d_ca = np.sqrt((diff * diff).sum(-1))
    R = ca.shape[0]; M = lig.shape[0]
    d_min = np.empty((R, M), dtype=np.float64)
    for r in range(R):
        h = heavy[r]                                  # (k,3)
        dd = h[:, None, :] - lig[None, :, :]          # (k,M,3)
        d_min[r] = np.sqrt((dd * dd).sum(-1)).min(0)
    mask = d_min <= CUTOFF
    rr, aa = np.nonzero(mask)
    return rr.astype(np.int32), aa.astype(np.int16), \
           d_ca[rr, aa].astype(np.float16), d_min[rr, aa].astype(np.float16)

def main():
    chosen = json.load(open('plinder_cache/chosen_systems.json'))
    by_zip = collections.defaultdict(list)
    for c in chosen:
        by_zip[c['zip']].append(c)
    print(f"{len(chosen)} chosen systems across {len(by_zip)} zips")

    gcs = pafs.GcsFileSystem(anonymous=True)

    # outputs (CSR)
    offsets = [0]
    res_row, atom_idx, d_ca, d_min = [], [], [], []
    sys_meta = []                  # per stored system: id, uniprot, ccd, smiles
    n_done = 0; n_fail = 0; capped = False
    fail_reasons = collections.Counter()
    t0 = time.time()

    for zc in sorted(by_zip):
        if capped: break
        path = f"{BUCKET}/{zc}.zip"
        try:
            seek = Seekable(gcs.open_input_file(path))
            zf = zipfile.ZipFile(seek)
        except Exception as e:
            for c in by_zip[zc]:
                n_fail += 1; fail_reasons['zip_open'] += 1
            continue
        members = set(zf.namelist())
        for c in by_zip[zc]:
            if Counter.total >= CAP_BYTES:
                capped = True; break
            sid = c['system_id']
            rcp = f"{sid}/receptor.pdb"
            # ligand sdf path: ligand_files/<ligand_chain>.sdf — pick any sdf under system
            sdfs = [m for m in members if m.startswith(f"{sid}/ligand_files/") and m.endswith('.sdf')]
            if rcp not in members or not sdfs:
                n_fail += 1; fail_reasons['missing_member'] += 1; continue
            try:
                pdb_bytes = zf.read(rcp)
                lig_parts = [ligand_heavy_coords(zf.read(s)) for s in sorted(sdfs)]
                lig_parts = [l for l in lig_parts if l is not None]
                if not lig_parts:
                    n_fail += 1; fail_reasons['ligand_parse'] += 1; continue
                lig = np.concatenate(lig_parts, 0)        # single-ligand systems (usually 1 sdf)
                ca, heavy = receptor_residues(pdb_bytes)
                if ca is None:
                    n_fail += 1; fail_reasons['receptor_parse'] += 1; continue
                rr, aa, dca, dmin = contacts_for_system(ca, heavy, lig)
                if rr.size == 0:
                    n_fail += 1; fail_reasons['no_contacts'] += 1; continue
            except Exception as e:
                n_fail += 1; fail_reasons[f'exc:{type(e).__name__}'] += 1; continue
            res_row.append(rr); atom_idx.append(aa); d_ca.append(dca); d_min.append(dmin)
            offsets.append(offsets[-1] + rr.size)
            sys_meta.append(c); n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done} systems, {Counter.total/1e6:.1f} MB, "
                      f"{sum(o for o in [offsets[-1]])} contacts, {time.time()-t0:.0f}s")
        # release file handle between zips
        try: seek.f.close()
        except Exception: pass

    # ---- assemble npz ----
    res_row = np.concatenate(res_row) if res_row else np.array([], np.int32)
    atom_idx = np.concatenate(atom_idx) if atom_idx else np.array([], np.int16)
    d_ca = np.concatenate(d_ca) if d_ca else np.array([], np.float16)
    d_min = np.concatenate(d_min) if d_min else np.array([], np.float16)
    offsets = np.asarray(offsets, np.int64)
    out = dict(
        contact_offsets=offsets,
        res_row=res_row, atom_idx=atom_idx, d_ca=d_ca, d_min=d_min,
        cutoff=np.float32(CUTOFF), source=np.str_('experimental'),
        system_id=np.array([c['system_id'] for c in sys_meta]),
        uniprot=np.array([c['uniprot'] for c in sys_meta]),
        ligand_ccd=np.array([c['ccd'] for c in sys_meta]),
        ligand_smiles=np.array([c['smiles'] for c in sys_meta], dtype=object),
        tier=np.array([c['tier'] for c in sys_meta]),
        pdb_id=np.array([c['pdb_id'] for c in sys_meta]),
    )
    np.savez('/workspace/datasets/plinder_experimental_labels.npz', **out)

    stats = dict(
        systems_fetched=n_done, systems_failed=n_fail,
        distinct_proteins=len(set(c['uniprot'] for c in sys_meta)),
        distinct_proteins_tierA=len(set(c['uniprot'] for c in sys_meta if c['tier']=='A')),
        total_contacts=int(offsets[-1]),
        downloaded_bytes=int(Counter.total),
        downloaded_MB=round(Counter.total/1e6, 2),
        capped=capped, cap_bytes=CAP_BYTES,
        tierA=sum(c['tier']=='A' for c in sys_meta),
        tierB=sum(c['tier']=='B' for c in sys_meta),
        fail_reasons=dict(fail_reasons),
        elapsed_s=round(time.time()-t0, 1),
    )
    json.dump(stats, open('plinder_cache/extract_stats.json','w'), indent=2)
    print("\n=== DONE ===")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
