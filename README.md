# pli-parallax

Pipeline producing three-dimensional coordinates for protein-ligand systems from
four architecturally distinct structure-prediction and docking programs, together
with residue-to-ligand-atom distance labels derived from those coordinates.

Data: 10.5281/zenodo.21560088 (CC-BY-4.0), byte-canonical
Mirror: https://huggingface.co/datasets/ThorKl/PLI-parallax, re-encoded on upload
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
    docs/         generate the field dictionary, manifest and validation report
    figures/      manuscript figures and the script that generates them
    superseded/   retained for provenance, do not use, see file headers

## Environments

Two, incompatible.

    main   torch 2.12.0   Boltz-2, smina, all extraction   requirements.txt
    chai   torch 2.6.0    Chai-1 only                      requirements-chai.txt

Use the correct interpreter per stage; do not mix. Note that gemmi and rdkit
differ between the two, so ligand perception is not identical across them.

## Teacher arms

Eight prediction arms across four configurations. Recycling depth was read from
each run hparams.yaml rather than from documentation and differs between arms.
The per-arm table is in the accompanying Data Descriptor.

    chai1               recycles 1, timesteps 80, ESM, 5 samples, rank-0
    boltz2 single-seq   msa empty, recycles 1, sampling 50, 1 sample
    boltz2 alignment    ColabFold MSA, recycles 1
    smina               exhaustiveness 8 corpus, 4 crystal, num_modes 1

## Install

Linux, glibc 2.39. Python 3.12.
System dependencies: smina (Oct 2019 build, AutoDock Vina 1.1.2 base),
Open Babel 3.1.1, MMseqs2.

    python -m venv main && ./main/bin/pip install -r requirements.txt
    python -m venv chai && ./chai/bin/pip install -r requirements-chai.txt

A few minutes plus teacher model weight downloads.

## Data

This pipeline produces the dataset archived at Zenodo 10.5281/zenodo.21560088
(CC-BY-4.0). A convenience mirror of the tabular layer is at
https://huggingface.co/datasets/ThorKl/PLI-parallax; its Parquet files are
re-encoded on upload, so checksums differ from the Zenodo manifest while the
content is identical. This repository contains code only.

## Cite

See CITATION.cff.

## License

Code Apache-2.0. Data CC-BY-4.0.

## Not included

Two steps in the published pipeline are not in this repository. No script here
writes the deposited Parquet distance tables from the intermediate npz shards;
that conversion was performed ad hoc, so the deposited tables are not
regenerable from this code alone, although every stage before and after them is.
The corpus-tier pocket annotation was produced by a three-detector ensemble
maintained outside this project and is deposited as a finished artifact.
