"""Score a teacher's predicted contacts against PLINDER experimental labels.
Metric: residue-level and pocket-level (5A) Jaccard of contact residues, plus
mean d_min error on shared residues. Reports overall and per-tier (A/B).
Both label sets use proteins_v10 global res_row, so comparison is frame-free.
Usage: calib_score.py <teacher_labels.npz> <teacher_name>
"""
import sys, numpy as np
exp=np.load('/workspace/datasets/plinder_experimental_labels_aligned.npz',allow_pickle=True)
pred=np.load(sys.argv[1],allow_pickle=True)
name=sys.argv[2]
# experimental: keyed by system_id; pred: keyed by pair_idx or system_id
def resmap(d, off, i, cut):
    s,e=off[i],off[i+1]; rr=d["res_row"][s:e]; dm=d["d_min"][s:e].astype(np.float32)
    out={}
    for r,v in zip(rr.tolist(),dm.tolist()):
        if r not in out or v<out[r]: out[r]=v
    return out
exp_off=exp["contact_offsets"]; exp_sys=exp["system_id"].tolist(); exp_tier=exp["tier"].tolist()
exp_idx={s:i for i,s in enumerate(exp_sys)}
# pred keyed by system_id (calibration extraction must store system_id)
pkey = pred["system_id"].tolist() if "system_id" in pred.files else None
if pkey is None:
    print("pred has no system_id; calibration extraction must tag it"); sys.exit(1)
pred_idx={s:i for i,s in enumerate(pkey)}
poff=pred["contact_offsets"]
rows=[]
for sid in exp_sys:
    if sid not in pred_idx: continue
    ei=exp_idx[sid]; pi=pred_idx[sid]
    re=resmap(exp,exp_off,ei,15); rp=resmap(pred,poff,pi,15)
    se,sp=set(re),set(rp)
    if not (se|sp): continue
    j15=len(se&sp)/len(se|sp)
    pe={r for r,v in re.items() if v<=5}; pp={r for r,v in rp.items() if v<=5}
    j5=len(pe&pp)/len(pe|pp) if (pe|pp) else 0.0
    common=se&sp
    derr=np.mean([abs(re[r]-rp[r]) for r in common]) if common else np.nan
    rows.append((sid, exp_tier[ei], j15, j5, derr))
import statistics
def report(subset,label):
    if not subset: print(f"  {label}: n=0"); return
    j15=[r[2] for r in subset]; j5=[r[3] for r in subset]
    de=[r[4] for r in subset if np.isfinite(r[4])]
    print(f"  {label}: n={len(subset)}  resJ15={statistics.mean(j15):.3f}  pocketJ5={statistics.mean(j5):.3f}  d_min_err={statistics.mean(de):.2f}A")
print(f"=== {name} vs experimental ===")
report(rows,"ALL")
report([r for r in rows if r[1]=='A'],"tier A")
report([r for r in rows if r[1]=='B'],"tier B")
