"""Summary tables for the coord_shards_v2 rebuild: attempted/written/skipped
with reason breakdown, teacher-presence distribution, and size."""
import glob, json, os, collections, sys
import h5py, numpy as np

def tier_summary(tier):
    d = f"/workspace/coord_shards_v2/{tier}"
    att = wr = sk = 0
    reasons = collections.Counter()
    for j in sorted(glob.glob(f"{d}/*.reasons.json")):
        m = json.load(open(j))
        att += m["attempted"]; wr += m["written"]; sk += m["skipped"]
        reasons.update(m["reasons"])
    pres = collections.Counter(); ngroups = collections.Counter()
    prot = collections.Counter(); npose = collections.Counter()
    n = 0; T = None
    for f in sorted(glob.glob(f"{d}/*.h5")):
        with h5py.File(f, "r") as h:
            T = [t.decode() for t in h.attrs["teachers"]]
            for s in h:
                n += 1
                v = h[s]["ligand_valid"][:]
                k = 0
                for i, t in enumerate(T):
                    if v[i].any():
                        pres[t] += 1; k += 1; npose[t] += int(v[i].sum())
                ngroups[k] += 1
                for g in h[s].get("protein", {}):
                    prot[g] += 1
    nbytes = sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.h5"))
    return dict(tier=tier, attempted=att, written=wr, skipped=sk,
                reasons=dict(sorted(reasons.items(), key=lambda x: -x[1])),
                systems_in_store=n, teachers=T,
                ligand_present={t: pres[t] for t in T},
                poses_total={t: npose[t] for t in T},
                protein_group={t: prot.get(t, 0) for t in T},
                teacher_groups_per_system={k: ngroups[k] for k in sorted(ngroups)},
                bytes_h5=nbytes)

if __name__ == "__main__":
    out = {t: tier_summary(t) for t in ("corpus", "crystal")}
    json.dump(out, open("/workspace/reports/coord_store_v2_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
