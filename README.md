# PLI-parallax

Pipeline producing three-dimensional coordinates for protein-ligand systems from
three structure-prediction and docking programs run in four configurations,
together with residue-to-ligand-atom distance labels derived from those
coordinates.

Data: 10.5281/zenodo.21560088 (CC-BY-4.0), canonical
Mirror: https://huggingface.co/datasets/ThorKl/PLI-parallax, byte-identical
Code: this repository (Apache-2.0), archived at Zenodo on release
Paper: ARTICLE_DOI once available

## What this produces

A coordinate store in HDF5 holding, per system and per teacher, the receptor
C-alpha trace, every receptor heavy atom within 15 A of the ligand, a side-chain
centroid and a backbone carbonyl carbon per shell residue, and ligand heavy-atom
coordinates for up to five poses on a fixed teacher axis. Distance tables in
Parquet hold a materialised view of that store at a fixed 15 A cutoff, recording
d_ca, d_min and contact flags at 4, 5 and 8 A. The tables are recomputable from
the store; the store is the primary artifact.

## Layout

    input_prep/   build teacher inputs from pair lists and receptors
    generate/     run each teacher
    extract/      derive distance labels from teacher outputs, merge, dedup
    store/        build and verify the coordinate store
    splits/       construct the leakage-controlled split family
    reliability/  fit and apply the per-system reliability annotation
    docs/         generate the field dictionary and verify quoted figures
    tests/        validation suite for the published deposit
    figures/      manuscript figures and the script that generates them

## Environments

Two, incompatible environments to be used.

    main   torch 2.12.0   Boltz-2, smina, all extraction   requirements.txt
    chai   torch 2.6.0    Chai-1 only                      requirements-chai.txt

Use the correct interpreter per stage. Note that gemmi and rdkit
differ between the two, so ligand perception is not identical across them. The
torch version is a requirement rather than a record: a later version produces
slightly different coordinates from the same input under the same seed.

## Teacher arms

Eight prediction arms across four configurations. Recycling depth was read from
each run hparams.yaml rather than from documentation and differs between arms:
the alignment-conditioned crystal arm ran one recycling step, the corpus pilot
ran three. The per-arm table is in the accompanying Data Descriptor.

    chai1               recycles 1, timesteps 80, ESM, 5 samples, rank-0
    boltz2 single-seq   msa empty, recycles 1, sampling 50, 1 sample
    boltz2 alignment    ColabFold MSA, recycles 1 crystal / 3 corpus pilot
    smina               exhaustiveness 4, num_modes 1

## Install

Linux, glibc 2.39. Python 3.12.
System dependencies: smina (Oct 2019 build, AutoDock Vina 1.1.2 base),
Open Babel 3.1.1, MMseqs2.

    python -m venv main && ./main/bin/pip install -r requirements.txt
    python -m venv chai && ./chai/bin/pip install -r requirements-chai.txt

A few minutes plus teacher model weight downloads.

## Paths

Pipeline scripts use the absolute paths of the machine they ran on. Most stages
need the intermediate prediction outputs, which are not deposited, so they are
not runnable elsewhere. The scripts that read only the published deposit take
the deposit root as an argument, defaulting to the current directory:

    tests/test_deposit_full.py         57 checks against the deposit
    docs/build_field_docs.py           field dictionary and record sets
    docs/verify_quoted_figures_v2.py   figures quoted in the article
    docs/verify_scoring_figures.py     scoring figures quoted in the article

Seven of the checks compare the coordinate store against the structures it was
built from and report as skipped unless PLIP_STRUCTURES, PLIP_AF and
PLIP_SYS2ACC point at them.

## Data

This pipeline produces the dataset archived at Zenodo 10.5281/zenodo.21560088
(CC-BY-4.0), released as a single 6.2 GB tar with the README, the Croissant
metadata and the manifest also provided unarchived, so the record can be
inspected before download. Extract with `tar xf pli_parallax_v1.0.0.tar` into an
empty directory and verify with `sha256sum -c MANIFEST.sha256`, which reports 43
files. A mirror is at https://huggingface.co/datasets/ThorKl/PLI-parallax
carrying the same bytes, so the Zenodo manifest verifies against either copy.
This repository contains code only.

## Cite

See CITATION.cff.

## License

Code Apache-2.0. Data CC-BY-4.0.

## Scope

Three boundaries. The npz shards are converted to the deposited Parquet tables
by the scripts under `extract/`, called in order rather than through a driver.
The corpus pocket annotation comes from a three-detector ensemble and is
deposited as a finished artifact. The reliability fit under `reliability/` runs
against an intermediate agreement table that is not deposited, so the fitted
arrays and the per-system field can be inspected but not regenerated from the
deposit alone.
