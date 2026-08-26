#!/usr/bin/env python
"""Draw the 400-system sample for the smina_exp_v2 re-run.

Population: the three-teacher-plus-ground-truth core -- the intersection of
system_id across labels_{chai1,boltz2,smina}_crystal_meta.parquet and
labels_crystal_groundtruth_meta.parquet in /workspace/deposit_v3/labels/.

Stratification: proportional (largest-remainder) allocation over metal_class as
assigned by bisy_smina_v2.classify_metal and recorded in the bisy_v2 CSVs, so the
metal_macrocycle class -- the one the current configuration handles worst -- enters
at its population rate rather than being under- or over-sampled.

Reads only; writes /workspace/reports/smina_v2_systems.txt.
"""
import collections, csv, glob, json, os, random, sys

import pandas as pd

LABELS = "/workspace/deposit_v3/labels"
BISY_V2 = "/workspace/datasets/experimental_expansion/bisy_v2"
OUT = "/workspace/reports/smina_v2_systems.txt"
N = 400
SEED = 20260821


def core_systems():
    sets = []
    for name in ("labels_chai1_crystal_meta", "labels_boltz2_crystal_meta",
                 "labels_smina_crystal_meta", "labels_crystal_groundtruth_meta"):
        sets.append(set(pd.read_parquet(f"{LABELS}/{name}.parquet")["system_id"]))
    return set.intersection(*sets)


def metal_classes():
    """system_id -> "" | "metal" | "metal_macrocycle", from the bisy_v2 scoring CSVs.

    The class is a property of the reference ligand, assigned before any RMSD is
    computed, so rows that failed to score still carry a usable class.
    """
    out = {}
    for path in sorted(glob.glob(f"{BISY_V2}/bisy_smina_v2_*.csv")):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                out[row["system_id"]] = row["metal_class"]
    return out


def allocate(counts, total):
    """Largest-remainder proportional allocation."""
    pop = sum(counts.values())
    exact = {k: v * total / pop for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - alloc[k]), k)):
        if sum(alloc.values()) >= total:
            break
        alloc[k] += 1
    return alloc


def main():
    core = core_systems()
    mc = metal_classes()
    missing = [s for s in core if s not in mc]
    if missing:
        print(f"warning: {len(missing)} core systems have no metal_class", file=sys.stderr)

    strata = collections.defaultdict(list)
    for s in sorted(core):
        strata[mc.get(s, "")].append(s)
    counts = {k: len(v) for k, v in strata.items()}
    alloc = allocate(counts, N)

    rng = random.Random(SEED)
    chosen = []
    for cls in sorted(strata):
        pool = sorted(strata[cls])
        take = min(alloc[cls], len(pool))
        chosen.extend(rng.sample(pool, take))
    chosen.sort()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(chosen) + "\n")

    print(f"core (4-way intersection) : {len(core)}")
    for cls in sorted(strata):
        label = cls or "(no metal)"
        print(f"  {label:<18s} pop {counts[cls]:5d} ({100.0*counts[cls]/len(core):5.2f}%)"
              f"   drawn {alloc[cls]:4d} ({100.0*alloc[cls]/N:5.2f}%)")
    print(f"total drawn               : {len(chosen)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
