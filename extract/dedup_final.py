
import numpy as np, os
srcs=[("A","/workspace/docking/output/smina/smina_labels_re.npz"),
      ("B","/workspace/smina_from_B/smina_B/smina_labels_re.npz"),
      ("C","/workspace/smina_from_C/smina_C/smina_labels_re.npz"),
      ("gap","/workspace/docking/output/smina_gap_rem/smina_labels_re.npz")]
srcs+=[("rem%d"%i,"/workspace/docking/output/smina_rem_%d/smina_labels_re.npz"%i) for i in range(8)]
srcs+=[("fg%d"%i,"/workspace/docking/output/smina_fg_%d/smina_labels_re.npz"%i) for i in range(8)]
def subset(d, keep):
    pid=d["pair_idx"]; off=d["contact_offsets"]
    rows=[i for i,p in enumerate(pid) if int(p) in keep]
    if not rows:
        return None
    rows_a=np.array(rows)
    lens=off[rows_a+1]-off[rows_a]
    new_off=np.concatenate([[0],np.cumsum(lens)]).astype(np.int64)
    idx=np.concatenate([np.arange(off[r],off[r+1]) for r in rows])
    out={"pair_idx":pid[rows_a],"accession":d["accession"][rows_a],
         "pocket_rank":d["pocket_rank"][rows_a],"affinity":d["affinity"][rows_a],
         "contact_offsets":new_off,"res_row":d["res_row"][idx],
         "atom_idx":d["atom_idx"][idx],"d_ca":d["d_ca"][idx],"d_min":d["d_min"][idx]}
    out["ambiguous"]=d["ambiguous"][rows_a] if "ambiguous" in d.files else np.zeros(len(rows),bool)
    return out
seen=set(); parts=[]; prov=[]
for ti,(tag,f) in enumerate(srcs):
    if not os.path.exists(f): continue
    d=np.load(f,allow_pickle=True)
    keep=set(int(p) for p in d["pair_idx"])-seen
    if not keep: continue
    seen|=keep
    s=subset(d,keep)
    if s is None: continue
    s["src_tag"]=np.full(len(s["pair_idx"]),ti,np.int16)
    parts.append(s); prov.append((tag,len(keep)))
out={}
for k in ["pair_idx","accession","pocket_rank","affinity","ambiguous","src_tag"]:
    out[k]=np.concatenate([p[k] for p in parts])
base=0; offs=[np.array([0])]
for p in parts:
    o=p["contact_offsets"]; offs.append(o[1:]+base); base+=int(o[-1])
out["contact_offsets"]=np.concatenate(offs).astype(np.int64)
for k in ["res_row","atom_idx","d_ca","d_min"]:
    out[k]=np.concatenate([p[k] for p in parts])
out["cutoff"]=np.float32(15.0); out["source"]=np.array("smina")
os.makedirs("/workspace/smina_combined",exist_ok=True)
np.savez_compressed("/workspace/smina_combined/smina_labels_final.npz",**out)
for tag,n in prov: print(f"{tag}: +{n}")
print("FINAL unique pairs",len(seen),"of 31878, gap",31878-len(seen))
