#!/usr/bin/env python3
"""Resumable, threaded fetch of receptor.pdb for every system present in
plinder_experimental_labels.npz. Each receptor.pdb is range-read from its GCS
zip and cached verbatim to plinder_cache/receptors/{system_id}.pdb. Re-running
skips already-cached files. Read-only on /workspace/docking."""
import io, os, sys, json, zipfile, collections, time
import numpy as np
import pyarrow.fs as pafs
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = 'plinder/2024-06/v2/systems'
OUTDIR = 'plinder_cache/receptors'
WORKERS = 12
os.makedirs(OUTDIR, exist_ok=True)

class Seekable(io.RawIOBase):
    def __init__(self, f):
        self.f = f; self._size = f.size(); self.pos = 0; self.nbytes = 0
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
        self.pos += len(b); self.nbytes += len(b)
        return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def process_zip(zc, sysids):
    res = {'zip': zc, 'ok': [], 'fail': collections.Counter()}
    gcs = pafs.GcsFileSystem(anonymous=True)
    try:
        seek = Seekable(gcs.open_input_file(f"{BUCKET}/{zc}.zip"))
        zf = zipfile.ZipFile(seek)
        members = set(zf.namelist())
    except Exception:
        res['fail']['zip_open'] += len(sysids); return res
    for sid in sysids:
        rcp = f"{sid}/receptor.pdb"
        if rcp not in members:
            res['fail']['missing_member'] += 1; continue
        try:
            data = zf.read(rcp)
            with open(f"{OUTDIR}/{sid}.pdb", 'wb') as fh:
                fh.write(data)
            res['ok'].append(sid)
        except Exception as e:
            res['fail'][f'exc:{type(e).__name__}'] += 1
    try: seek.f.close()
    except Exception: pass
    return res

def main():
    d = np.load('plinder_experimental_labels.npz', allow_pickle=True)
    sysids = d['system_id'].tolist()
    chosen = json.load(open('plinder_cache/chosen_systems.json'))
    zmap = {c['system_id']: c['zip'] for c in chosen}
    by_zip = collections.defaultdict(list)
    missing_zip = 0
    for sid in sysids:
        if sid not in zmap: missing_zip += 1; continue
        if os.path.exists(f"{OUTDIR}/{sid}.pdb"): continue
        by_zip[zmap[sid]].append(sid)
    todo_sys = sum(len(v) for v in by_zip.values())
    print(f"{len(sysids)} systems; {missing_zip} without zip; "
          f"{todo_sys} to fetch across {len(by_zip)} zips")
    if not by_zip:
        print("all receptors cached"); return 0
    t0 = time.time(); fails = collections.Counter(); nok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_zip, z, s): z for z, s in by_zip.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); nok += len(r['ok']); fails += r['fail']
            if i % 25 == 0:
                print(f"  {i}/{len(futs)} zips, {nok} ok, {time.time()-t0:.0f}s")
    print(f"done: {nok} fetched, fails={dict(fails)}, {time.time()-t0:.0f}s")
    return 0

if __name__ == '__main__':
    sys.exit(main())
