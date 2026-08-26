#!/usr/bin/env python
"""Merge the resume pass's per-shard manifests into one manifest + a completion report.

The resume pass re-walked every in-scope system: ones already on disk came back as
"skipped" but still carry a copy-selection method and dmin residual, so its eight
manifests together cover the full scope, first-run systems included.
"""
import csv, glob, os, sys, subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")
from extract_crystal_ref_ligands import load_scope, LABEL_DIR, OUT_DIR, RECEPTOR_SUBDIR

REPORT_DIR = "/workspace/reports"
MANIFEST = os.path.join(REPORT_DIR, "manifest_pb_inputs.csv")
TAG = "resume_shard%dof8"

scope = load_scope(LABEL_DIR, "teachers")
scope_ids = list(scope["system_id"])
scope_set = set(scope_ids)

# ---- merge manifests ------------------------------------------------------------------
rows = {}
dupes = 0
for s in range(8):
    p = os.path.join(OUT_DIR, f"manifest_{TAG % s}.csv")
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r["system_id"] in rows:
                dupes += 1
            r["shard"] = s
            rows[r["system_id"]] = r

# ---- failures -------------------------------------------------------------------------
fails = {}
for s in range(8):
    p = os.path.join(REPORT_DIR, f"extract_crystal_ref_ligands_failures_{TAG % s}.csv")
    with open(p) as fh:
        for r in csv.DictReader(fh):
            r["shard"] = s
            fails[r["system_id"]] = r

# ---- verify every manifest row points at real, non-empty files -------------------------
missing_files, empty_files = [], []
for sid, r in rows.items():
    for key in ("ligand_sdf", "receptor_pdb"):
        path = os.path.join(OUT_DIR, r[key])
        if not os.path.exists(path):
            missing_files.append((sid, r[key]))
        elif os.path.getsize(path) == 0:
            empty_files.append((sid, r[key]))

unaccounted = sorted(scope_set - set(rows) - set(fails))

fields = ["system_id", "pdb", "ccd", "chain", "ligand_sdf", "receptor_pdb",
          "select_method", "n_copies", "dmin_residual_A"]
with open(MANIFEST, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for sid in scope_ids:               # scope order, so the manifest is stable
        if sid in rows:
            w.writerow(rows[sid])

# ---- stats -----------------------------------------------------------------------------
res = np.array([float(r["dmin_residual_A"]) for r in rows.values() if r["dmin_residual_A"] != ""])
methods = Counter(r["select_method"] for r in rows.values())
stages = Counter(r["stage"] for r in fails.values())
reasons = Counter((r["stage"], r["reason"].split("(")[0].split(":")[0].strip())
                  for r in fails.values())

n_sdf = len(glob.glob(os.path.join(OUT_DIR, "*_ref.sdf")))
n_rec = len(glob.glob(os.path.join(OUT_DIR, RECEPTOR_SUBDIR, "*.pdb")))
def du(path):
    return subprocess.run(["du", "-sb", path], capture_output=True, text=True).stdout.split()[0]
b_all, b_rec = int(du(OUT_DIR)), int(du(os.path.join(OUT_DIR, RECEPTOR_SUBDIR)))

out = []
def emit(line=""):
    out.append(line)

emit("# Crystal reference-ligand extraction -- completion report")
emit()
emit(f"scope (teachers)            : {len(scope_ids)} systems")
emit(f"manifest rows               : {len(rows)}")
emit(f"failures                    : {len(fails)}")
emit(f"unaccounted (neither)       : {len(unaccounted)}")
emit(f"duplicate manifest rows     : {dupes}")
emit(f"coverage                    : {100.0 * len(rows) / len(scope_ids):.2f}%")
emit()
emit("## Files on disk")
emit(f"ref SDFs                    : {n_sdf}")
emit(f"receptor PDBs               : {n_rec}")
emit(f"manifest rows w/ missing file: {len(missing_files)}")
emit(f"manifest rows w/ empty file : {len(empty_files)}")
emit()
emit("## Disk used (/workspace/pb_inputs)")
emit(f"receptors                   : {b_rec / 1e9:.2f} GB")
emit(f"ligands + cache             : {(b_all - b_rec) / 1e6:.1f} MB")
emit(f"total                       : {b_all / 1e9:.2f} GB")
emit()
emit("## Failures by class")
if not fails:
    emit("(none)")
for (stage, reason), n in reasons.most_common():
    emit(f"{stage:16s} {n:4d}  {reason}")
emit()
emit("by stage: " + ", ".join(f"{k}={v}" for k, v in stages.most_common()))
emit()
emit("## Ligand copy resolved by")
for m, n in methods.most_common():
    emit(f"{m:16s} {n:6d}  ({100.0 * n / len(rows):.1f}%)")
emit()
emit("## dmin_residual_A distribution")
emit(f"rows with a residual        : {len(res)} of {len(rows)} "
     f"({len(rows) - len(res)} single-copy systems have no residual)")
emit(f"max                         : {res.max():.4f} A")
emit(f"median                      : {np.median(res):.4f} A")
emit(f"mean                        : {res.mean():.4f} A")
emit(f"count > 0.01 A              : {int((res > 0.01).sum())} ({100.0 * (res > 0.01).mean():.2f}%)")
emit(f"count > {0.05} A (DMIN_TOL)      : {int((res > 0.05).sum())}")
emit("percentiles                 : " + ", ".join(
    f"p{p}={np.percentile(res, p):.4f}" for p in (50, 90, 99, 99.9)))
emit()
emit("## Per-shard")
emit("shard   rows  failures")
per = Counter(r["shard"] for r in rows.values())
perf = Counter(r["shard"] for r in fails.values())
for s in range(8):
    emit(f"  {s}    {per[s]:5d}     {perf[s]}")
emit()
emit(f"manifest: {MANIFEST}")
if unaccounted:
    emit()
    emit("## Unaccounted system_ids")
    for sid in unaccounted[:50]:
        emit(f"  {sid}")

text = "\n".join(out)
print(text)
with open(os.path.join(REPORT_DIR, "extraction_completion_report.md"), "w") as fh:
    fh.write(text + "\n")

# merged failure CSV, for convenience
with open(os.path.join(REPORT_DIR, "extract_crystal_ref_ligands_failures_all.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["system_id", "pdb", "ccd", "chain", "stage", "reason",
                                       "shard"])
    w.writeheader()
    for sid in sorted(fails):
        w.writerow(fails[sid])
