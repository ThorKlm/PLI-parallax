#!/usr/bin/env python3
"""Figures for the PLI-Parallax Data Descriptor.

Palette: gold for the two data tiers, blue for the processing this work
performs, grey for sources and deposited output. Within blue the teachers run a
four-step ramp ordered by architectural distance from the physics-based end,
smina lightest through Boltz-2 MSA darkest. Experimental ground truth is black,
since it is a reference rather than a teacher. The ramp is stated in every
caption so it does not read as arbitrary.

Sentence case throughout, product names as their owners write them, field names
monospace lowercase, Angstrom written as a bare A. Panel letters lower-case bold
top-left, which is what the journal asks for in merged multi-panel files.

Values marked PENDING are stand-ins; pending() registers them and
audit_pending() lists them at the end of a run. Missing input arrays raise
rather than falling back to anything synthetic.

    python generate_all_figures.py                 # the seven that ship
    python generate_all_figures.py fig9_splits     # one by name
    python generate_all_figures.py --all           # every variant

    PLIP_FIG_INPUT   directory holding the .npy inputs, default "."
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- style block

COLORS = {
    "given":        "#D9D9D9",
    "tier":         "#FFCC00",
    "produced":     "#6C99D8",
    "smina":        "#A9C9F0",
    "boltz2":       "#6C99D8",
    "chai1":        "#2B5EA7",
    "boltz2_msa":   "#1B3F73",
    "groundtruth":  "#2C2C2C",
    "hatch_orange": "#F5A623",
    "black":        "#2C2C2C",
    "gray":         "#888888",
    "corpus":       "#6C99D8",
    "crystal":      "#1B3F73",
}

FONT_TITLE = 22
FONT_STAGE = 15
FONT_DETAIL = 12
FONT_LABEL = 13
FONT_ANNO = 12
FONT_SMALL = 10

LW_BOX = 2.2
LW_BOX_IO = 1.5
LW_ARROW = 2.2
BOX_PAD = 0.12      # must equal the boxstyle pad, or arrows enter the padding
ARROW_GAP = 0.06    # visible clearance between an arrowhead and a box edge

OUTPUT_DIR = "./figures/"
INPUT_DIR = os.environ.get("PLIP_FIG_INPUT", ".")
RASTER_DPI = 300

PENDING_LOG = []

def pending(label, value):
    """Register a stand-in value and return it. Rendered labels get [!]."""
    PENDING_LOG.append(label)
    return value

def pmark(text):
    return f"{text} [!]"

def audit_pending():
    if not PENDING_LOG:
        print("no pending values")
        return
    print(f"pending values ({len(PENDING_LOG)}):")
    for k in PENDING_LOG:
        print(f"  [!] {k}")

def load_array(name):
    """Inputs are required. A missing file is an error, not a reason to invent
    one; two figures once shipped from synthetic arrays that way."""
    p = os.path.join(INPUT_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found; set PLIP_FIG_INPUT")
    return np.load(p)

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(fig, filename, dpi=RASTER_DPI, tight=True):
    ensure_output_dir()
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(OUTPUT_DIR, f"{filename}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None,
                    facecolor="white", edgecolor="none")
        print(f"Saved: {path}")

def draw_box(ax, x, y, w, h, color, lines, lw=LW_BOX):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle=f"round,pad={BOX_PAD}",
                                facecolor=color, edgecolor="black", linewidth=lw))
    line_h = 0.36
    top = y + (len(lines) - 1) * line_h / 2
    for i, (text, bold, fs) in enumerate(lines):
        ax.text(x, top - i * line_h, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

def draw_io_box(ax, x, y, w, h, color, label, fontsize=FONT_DETAIL):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle=f"round,pad={BOX_PAD}",
                                facecolor=color, edgecolor="black", linewidth=LW_BOX_IO))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)

def draw_arrow(ax, xy_from, xy_to, lw=LW_ARROW, color="black"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="->", mutation_scale=18,
                                 fc=color, ec=color, linewidth=lw))

def connect(ax, x_from, y_from, h_from, x_to, y_to, h_to):
    """Arrow between two boxes, clearing the rounded-box padding at both ends."""
    clear = BOX_PAD + ARROW_GAP
    draw_arrow(ax, (x_from, y_from - h_from/2 - clear), (x_to, y_to + h_to/2 + clear))

def apply_unspec_hatch(artists, index, hatch_color=COLORS["hatch_orange"]):
    try:
        a = artists[index]
        a.set_hatch("///"); a.set_edgecolor(hatch_color); a.set_linewidth(1.5)
    except (IndexError, AttributeError):
        pass

def panel_letter(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FONT_LABEL + 2,
            fontweight="bold", ha="left", va="top")

def clean_axes(ax, ygrid=True):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.3)

def annot_box(ax, x, y, text, ha="right", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FONT_SMALL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="0.8", linewidth=0.8))

# ------------------------------------------------------------------- the data

CORPUS_COVERAGE = [
    ("smina",                    31713, "smina"),
    ("Boltz-2 single-sequence",  23494, "boltz2"),
    ("Chai-1",                   23485, "chai1"),
    ("Boltz-2 MSA (pilot)",        161, "boltz2_msa"),
]
CORPUS_TRIPLE = 23451
CORPUS_UNION = 31746
N_PROTEINS = 906
N_PROTEINS_COFOLD = 678

# smina is the full corrected arm at 11,290 successful of 11,543 attempted, not
# the three-teacher core. CRYSTAL_CORE below is the intersection, drawn as a
# reference line rather than a bar. The two alignment-conditioned arms are
# listed separately and are not summed: the deposited arm ran one recycling step
# on fused chains and emitted PDB, the later arm ran three on true multi-chain
# inputs and emitted mmCIF, and their system sets are disjoint.
CRYSTAL_COVERAGE = [
    ("Ground truth (BioLiP2)",   19350, "groundtruth"),
    ("smina",                    11290, "smina"),
    ("Chai-1",                    9410, "chai1"),
    ("Boltz-2 single-sequence",   8725, "boltz2"),
    ("Boltz-2 MSA (deposited)",   1755, "boltz2_msa"),
]
CRYSTAL_CORE = 7166
CRYSTAL_SMINA_ATTEMPTED = 11543

N_PARQUET = 30
N_SPLITS = 646
N_SPLIT_FAMILIES = 7

PROTEIN_CLUSTERS = {0.30: 886, 0.40: 896, 0.50: 900}
PROTEIN_CLUSTER_SIZES = {1: 886, 2: 10}

# From ligand_cluster_report.json, which is what the split construction used.
# The ligand count is the number entering ligand clustering, which is neither
# the 31,193 distinct InChIKeys in ligand_identity.json nor the 31,034 in the
# smina metadata table.
SCAFFOLD_N = 15173
SCAFFOLD_SINGLETON_FRAC = 0.7068
SCAFFOLD_LARGEST = 274
SCAFFOLD_TOP10_FRAC = 0.0407
SCAFFOLD_TOP50_FRAC = 0.098
N_LIGANDS_CLUSTERED = 31063
BUTINA_N = 7933
TANIMOTO_ABOVE_030 = 0.0011

PB_CONTROL = 0.633
PB_RATES = [
    ("Boltz-2 MSA",              0.532, 1743, "boltz2_msa"),
    ("Chai-1",                   0.388, 9337, "chai1"),
    ("Boltz-2 single-sequence",  0.248, 8653, "boltz2"),
    # Scored under deposited-receptor conditioning so the value sits on the same
    # axis as the cofolders and the crystal control.
    ("smina",                    0.8348, 11289, "smina"),
]

# Ground-truth sanitisation is 0.000 while every teacher fails it at 16.8 to
# 19.5 per cent, so this is a property of predicted output rather than a shared
# perception limit. Fractions are over the rows where the check ran: the two
# geometry checks have 95 systems with no conditioning receptor.
PB_FAILMODES = {
    "min. distance to protein": {"Ground truth": 0.263, "Boltz-2 MSA": 0.289,
                                 "Chai-1": 0.501, "Boltz-2 single-seq": 0.688,
                                 "smina": 0.0792},
    "volume overlap":           {"Ground truth": 0.077, "Boltz-2 MSA": 0.061,
                                 "Chai-1": 0.087, "Boltz-2 single-seq": 0.241,
                                 "smina": 0.0425},
    "InChI convertible":        {"Ground truth": 0.051, "Boltz-2 MSA": 0.195,
                                 "Chai-1": 0.223, "Boltz-2 single-seq": 0.183,
                                 "smina": 0.0750},
    "sanitization":             {"Ground truth": 0.000, "Boltz-2 MSA": 0.195,
                                 "Chai-1": 0.168, "Boltz-2 single-seq": 0.182,
                                 "smina": 0.0349},
}

# n is the resolved count, lower than the arm coverage because symmetry-corrected
# RMSD needs an atom-graph isomorphism to the reference ligand. The npy key must
# name the same arm as the row: boltz2_msa is the deposited alignment-conditioned
# crystal arm, and the multimer arm is a different configuration entirely. Figure
# five recomputes n, median and the two fractions from the arrays and warns where
# they disagree with what is declared here.
RMSD_SUMMARY = [
    ("Boltz-2 aligned",           456,  3.12, 0.428, 0.560, "boltz2_msa"),
    ("Chai-1",                   9519,  5.24, 0.320, 0.491, "chai1"),
    ("Boltz-2 single-sequence",  8820, 11.41, 0.045, 0.185, "boltz2"),
    ("smina",                    6485,  4.15, 0.289, 0.565, "smina"),
]

# Two cohorts, both defensible. CORE is the systems all three arms cover, so the
# three values are directly comparable; OWN is each arm on its own coverage.
# Whichever the figure uses, the caption has to say which.
CONTACT_ACCURACY_CORE = [
    ("smina",                    0.5592, "smina"),
    ("Chai-1",                   0.4442, "chai1"),
    ("Boltz-2 single-sequence",  0.1815, "boltz2"),
]
CONTACT_ACCURACY_OWN = [
    ("smina",                    0.5595, "smina"),
    ("Chai-1",                   0.4443, "chai1"),
    ("Boltz-2 single-sequence",  0.1816, "boltz2"),
]
CONTACT_ACCURACY = CONTACT_ACCURACY_CORE
CONTACT_COHORT = "systems covered by all three arms"

# n is teacher-system rows and sums to 19,739, not systems: each system
# contributes one row per scored teacher.
FOLD_BANDS_UNIT = "teacher-system pairs"
FOLD_BANDS = [
    ("< 1",     3362, 0.6128),
    ("1 to 2",  2534, 0.5406),
    ("2 to 5",  2688, 0.4277),
    (">= 5",   11155, 0.2047),
]
ISOTONIC_MAE = 0.0805        # held-out half of the fit split, as plotted
BASELINE_MAE = 0.1752
CONFORMAL = [(0.90, 0.1526, 0.900), (0.80, 0.1200, 0.800)]

SUPPORT_CRYSTAL = [
    (1, 147966, 0.2175),
    (2,  45268, 0.7265),
    (3,  16200, 0.9356),
]
SUPPORT_CORPUS = [(1, 719080, None), (2, 106481, None), (3, 21762, None)]
UNION_RECALL = 0.8828

FRAME_CONTROL = {
    "protein-ligand max. distance": (1.0000, 0.2843),
    "min. distance to protein":     (0.5528, 0.7665),
    "volume overlap":               (0.9246, 0.8274),
}

# Family means over the shipped split summary tables. c2st_auc is the ligand
# axis, c2st_auc_protein the protein axis, and c2st_auc_protein_length the
# length-only baseline the protein value has to be read against.
SPLIT_FAMILIES = ["C1", "C2p", "C2l", "C3", "C3comp"]
SPLIT_DIAGNOSTICS = {
    "corpus": {
        "C1":     dict(lig=0.5082, prot=0.5001, base=0.4863, cov80=1.0000, local=1.0),
        "C2p":    dict(lig=0.8075, prot=0.5094, base=0.5003, cov80=0.3651, local=1.0),
        "C2l":    dict(lig=0.6183, prot=0.5049, base=0.4912, cov80=1.0000, local=1.0),
        "C3":     dict(lig=0.8241, prot=0.5051, base=0.5003, cov80=0.3563, local=1.0),
        "C3comp": dict(lig=0.8989, prot=0.4919, base=0.4929, cov80=0.3366, local=1.0),
    },
    "crystal": {
        "C1":     dict(lig=0.5230, prot=0.5376, base=0.4991, cov80=1.0000, local=1.0),
        "C2p":    dict(lig=0.6085, prot=0.6068, base=0.5714, cov80=0.9153, local=1.0),
        "C2l":    dict(lig=0.9618, prot=0.6580, base=0.5067, cov80=1.0000, local=1.0),
        "C3":     dict(lig=0.9686, prot=0.7798, base=0.5735, cov80=0.7839, local=1.0),
        "C3comp": dict(lig=0.9283, prot=0.7591, base=0.5697, cov80=0.8180, local=1.0),
    },
}
SPLIT_COUNTS = {"corpus": 375, "crystal": 250, "temporal": 6, "lead optimisation": 15}

# Core against non-core, both within the crystal tier. Not the same split as
# covered-by-cofolders against excluded-by-the-length-cap, which gives 404
# and 1,101.
APPLICABILITY = [
    ("Residues (median)",            414,   1344),
    ("Ligand heavy atoms (median)",   22,     44),
    ("Chains (median)",                2,      4),
    ("Metal-containing (%)",         5.5,   19.0),
    ("Interface binder (%)",        14.6,   32.4),
]
ZERO_CORE_LIGANDS = [("SF4", 535), ("CLA", 375), ("CO", 142), ("NAI", 87),
                     ("DD6", 84), ("CHL", 74), ("BCL", 46), ("CU1", 43),
                     ("B12", 43), ("FDA", 30)]

# ----------------------------------------------------------------- figure one

def create_figure_1_workflow():
    """Construction workflow. Two semantic colours: blue for given data, gold
    for what this work produces. Bands evenly spaced with the stack raised so
    the legend clears the output box; arrows clear the rounded-box padding at
    both ends."""
    fig = plt.figure(figsize=(12, 10.5))
    ax = fig.add_axes([0.04, 0.01, 0.92, 0.98])
    ax.set_xlim(0, 12); ax.set_ylim(1.6, 13.0); ax.axis("off")
    cx, xl, xr = 6.0, 3.2, 8.8
    bw, bh, sh_io = 5.0, 1.25, 0.58
    h_teach = bh + 0.20
    h_out = sh_io * 1.60
    # Layout computed rather than hardcoded, so the inter-box gap is always
    # positive and equal and the arrows cannot invert when a height changes.
    TOP, BOTTOM = 12.10, 2.60
    heights = [sh_io, bh, h_teach, bh, h_out]
    gap = ((TOP - BOTTOM) - sum(heights) - 2 * BOX_PAD * len(heights)) / (len(heights) - 1)
    centres, y = [], TOP
    for h in heights:
        y -= BOX_PAD + h / 2
        centres.append(y)
        y -= h / 2 + BOX_PAD + gap
    y_src, y_tier, y_teach, y_lab, y_out = centres

    ax.text(cx, 12.7, "PLI-Parallax construction", ha="center", va="center",
            fontsize=FONT_TITLE, fontweight="bold")
    draw_io_box(ax, cx, y_src, 11.0, sh_io, COLORS["given"],
                "BioLiP2  |  AlphaFold DB v6  |  PDB  |  BindingDB  |  ChEMBL  |  Bernett protein universe",
                fontsize=FONT_DETAIL + 1)
    connect(ax, cx - 0.4, y_src, sh_io, xl, y_tier, bh)
    connect(ax, cx + 0.4, y_src, sh_io, xr, y_tier, bh)
    draw_box(ax, xl, y_tier, bw, bh, COLORS["tier"], [
        ("Corpus tier", True, FONT_STAGE),
        ("non-cognate pairs on AlphaFold receptors", False, FONT_DETAIL),
        (f"{CORPUS_UNION:,} systems without an experimental complex", False, FONT_DETAIL),
    ])
    draw_box(ax, xr, y_tier, bw, bh, COLORS["tier"], [
        ("Crystal tier", True, FONT_STAGE),
        ("experimental complexes from BioLiP2", False, FONT_DETAIL),
        (f"{CRYSTAL_COVERAGE[0][1]:,} systems with ground truth", False, FONT_DETAIL),
    ])
    connect(ax, xl, y_tier, bh, xl, y_teach, h_teach)
    connect(ax, xr, y_tier, bh, xr, y_teach, h_teach)
    draw_box(ax, cx, y_teach, 11.0, h_teach, COLORS["produced"], [
        ("Four teacher configurations on shared systems", True, FONT_STAGE),
        ("Chai-1 (ESM, 1 recycle)   Boltz-2 single-sequence   Boltz-2 with MSA   smina docking",
         False, FONT_DETAIL),
        ("per-arm settings in Table 1", False, FONT_DETAIL),
    ])
    connect(ax, cx, y_teach, h_teach, cx, y_lab, bh)
    draw_box(ax, cx, y_lab, 8.0, bh, COLORS["produced"], [
        ("Residue-to-ligand-atom labels", True, FONT_STAGE),
        ("d_ca, d_min and contact flags at 4, 5 and 8 A within 15 A", False, FONT_DETAIL),
        (f"three-teacher core: {CORPUS_TRIPLE:,} corpus, {CRYSTAL_CORE:,} crystal",
         False, FONT_DETAIL),
    ])
    connect(ax, cx, y_lab, bh, cx, y_out, h_out)
    draw_io_box(ax, cx, y_out, 11.0, h_out, COLORS["given"],
                f"{N_PARQUET} Parquet tables  |  HDF5 coordinate store  |  "
                f"{N_SPLITS} split configurations\n"
                "per-system reliability  |  per-residue support  |  "
                "manifest, Croissant, field dictionary",
                fontsize=FONT_DETAIL)
    ax.legend(handles=[
        mpatches.Patch(facecolor=COLORS["given"], edgecolor="black", lw=1,
                       label="Source and deposited output"),
        mpatches.Patch(facecolor=COLORS["tier"], edgecolor="black", lw=1,
                       label="Data tier"),
        mpatches.Patch(facecolor=COLORS["produced"], edgecolor="black", lw=1,
                       label="Processing step"),
    ], loc="upper center", fontsize=FONT_SMALL + 1, frameon=True, fancybox=True,
        ncol=3, bbox_to_anchor=(0.5, 0.055))
    return fig

# ----------------------------------------------------------------- figure two

def create_figure_2_coverage():
    """Per-teacher coverage and the intersection that defines the core."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, rows, core, core_lbl, letter) in zip(axes, [
        ("Corpus tier", CORPUS_COVERAGE, CORPUS_TRIPLE, "three-teacher core", "a"),
        ("Crystal tier", CRYSTAL_COVERAGE, CRYSTAL_CORE, "three teachers and\nground truth", "b"),
    ]):
        rows = sorted(rows, key=lambda r: r[1])
        names = [r[0] for r in rows]; counts = [r[1] for r in rows]
        colors = [COLORS[r[2]] for r in rows]
        bars = ax.barh(names, counts, color=colors, edgecolor="black", linewidth=0.6)
        for b, c in zip(bars, counts):
            ax.text(b.get_width() * 1.03, b.get_y() + b.get_height()/2, f"{c:,}",
                    va="center", ha="left", fontsize=FONT_SMALL)
        ax.axvline(core, color=COLORS["hatch_orange"], linestyle="--", linewidth=1.2,
                   label=f"{core_lbl}\n(at {core:,})")
        ax.set_xscale("log"); ax.set_xlim(80, 200000)
        ax.set_xlabel("Systems (log scale)", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_LABEL, fontweight="bold", pad=10)
        ax.tick_params(labelsize=FONT_SMALL)
        ax.legend(loc="lower right", fontsize=FONT_SMALL - 1, frameon=True,
                  fancybox=True, framealpha=0.5, borderpad=0.6,
                  bbox_to_anchor=(1.0, 0.02))
        clean_axes(ax, ygrid=False); ax.grid(axis="x", alpha=0.3)
        panel_letter(ax, letter)
    fig.tight_layout()
    return fig

# --------------------------------------------------------------- figure three

def _panel_protein_clusters(ax, letter="a"):
    prot = load_array("protein_cluster_sizes.npy")
    vals, cnts = np.unique(prot, return_counts=True)
    ax.bar(vals, cnts, color=COLORS["chai1"], edgecolor="black", linewidth=0.6, width=0.6)
    for v, c in zip(vals, cnts):
        ax.text(v, c * 1.08, f"{c:,}", ha="center", va="bottom", fontsize=FONT_SMALL)
    ax.set_yscale("log"); ax.set_xticks(vals)
    ax.set_xlabel("Proteins per cluster", fontsize=FONT_LABEL)
    ax.set_ylabel("Clusters", fontsize=FONT_LABEL)
    ax.set_title(f"Protein clusters, {PROTEIN_CLUSTERS[0.40]} from {N_PROTEINS}\n"
                 "MMseqs2, 40% identity, coverage 0.8",
                 fontsize=FONT_LABEL, fontweight="bold", pad=10)
    annot_box(ax, 0.97, 0.94, f"{PROTEIN_CLUSTERS[0.30]} clusters at 30%\n"
                              f"{PROTEIN_CLUSTERS[0.50]} clusters at 50%")
    clean_axes(ax); panel_letter(ax, letter, x=-0.16, y=1.16)

def _panel_scaffold_ccdf(ax, letter="b"):
    scaf = load_array("scaffold_cluster_sizes.npy")
    sizes = np.sort(scaf)
    x = np.unique(sizes)
    ccdf = np.array([(sizes >= v).mean() for v in x])
    ax.step(x, ccdf, where="post", color=COLORS["chai1"], linewidth=2.0)
    ax.axhline(1.0 - SCAFFOLD_SINGLETON_FRAC, color=COLORS["hatch_orange"],
               linestyle="--", linewidth=1.2,
               label=f"{100*SCAFFOLD_SINGLETON_FRAC:.1f}% of scaffolds are singletons")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Scaffold cluster size, n ligands", fontsize=FONT_LABEL)
    ax.set_ylabel("Fraction of scaffolds\nwith at least n ligands", fontsize=FONT_LABEL)
    ax.set_title(f"Ligand scaffolds, {SCAFFOLD_N:,} from {N_LIGANDS_CLUSTERED:,}\n"
                 f"largest cluster {SCAFFOLD_LARGEST}",
                 fontsize=FONT_LABEL, fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=FONT_SMALL - 1, frameon=True,
              fancybox=True, framealpha=0.8)
    annot_box(ax, 0.97, 0.62, f"ten largest scaffolds hold {100*SCAFFOLD_TOP10_FRAC:.1f}% of ligands\n"
                              f"fifty largest hold {100*SCAFFOLD_TOP50_FRAC:.1f}%\n"
                              f"{100*TANIMOTO_ABOVE_030:.2f}% of ligand pairs exceed\n"
                              f"Tanimoto 0.30")
    clean_axes(ax); panel_letter(ax, letter, x=-0.20, y=1.16)

def create_figure_3_entity_space():
    """Non-redundancy of both entity axes. The left panel is a bar chart because
    the distribution takes two values; the right is a complementary cumulative
    distribution, which reads the singleton fraction and the tail directly
    without the binning artefacts a log-binned histogram introduces."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    _panel_protein_clusters(axes[0], "a")
    _panel_scaffold_ccdf(axes[1], "b")
    fig.tight_layout()
    return fig

def create_figure_3_protein_only():
    """Protein axis alone, for a layout where the scaffold numbers sit in the
    text instead of a panel."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    _panel_protein_clusters(ax, "")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- figure four

def create_figure_4_posebusters():
    """Physical validity per teacher against the crystal control. The control is
    mandatory: experimental ligands also fail under this check configuration, so
    a teacher rate is only interpretable relative to it."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    rows = sorted(PB_RATES, key=lambda r: r[1])
    names = [r[0] for r in rows]
    rates = [r[1] for r in rows]; ns = [r[2] for r in rows]
    colors = [COLORS[r[3]] for r in rows]
    bars = ax.barh(names, rates, color=colors, edgecolor="black", linewidth=0.6)
    for b, r, n in zip(bars, rates, ns):
        ax.text(b.get_width() + 0.012, b.get_y() + b.get_height()/2,
                f"{r:.3f}  (n={n:,})", va="center", ha="left", fontsize=FONT_SMALL)
    ax.axvline(PB_CONTROL, color=COLORS["black"], linestyle="--", linewidth=0.9,
               alpha=0.5)
    ax.text(PB_CONTROL + 0.02, 0.03, f"crystal control {PB_CONTROL:.3f}",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=FONT_SMALL)
    ax.set_xlim(0, 1.15); ax.set_xlabel("PoseBusters valid fraction", fontsize=FONT_LABEL)
    ax.set_title("Physical validity", fontsize=FONT_LABEL, fontweight="bold", pad=10)
    ax.tick_params(labelsize=FONT_SMALL)
    clean_axes(ax, ygrid=False); ax.grid(axis="x", alpha=0.3); panel_letter(ax, "a")
    ax = axes[1]
    checks = list(PB_FAILMODES.keys())
    groups = ["Ground truth", "Boltz-2 MSA", "Chai-1", "Boltz-2 single-seq", "smina"]
    gcol = [COLORS["groundtruth"], COLORS["boltz2_msa"], COLORS["chai1"],
            COLORS["boltz2"], COLORS["smina"]]
    glabel = list(groups)
    y = np.arange(len(checks)); h = 0.16
    for i, (g, c, lbl) in enumerate(zip(groups, gcol, glabel)):
        ax.barh(y + (i - 2) * h, [PB_FAILMODES[k][g] for k in checks], height=h,
                color=c, edgecolor="black", linewidth=0.4, label=lbl)
    ax.set_yticks(y); ax.set_yticklabels(checks, fontsize=FONT_SMALL)
    ax.set_xlabel("Fraction failing", fontsize=FONT_LABEL)
    ax.set_title("Dominant failure modes", fontsize=FONT_LABEL, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_SMALL - 1, frameon=True, fancybox=True,
              framealpha=0.8, loc="upper right")
    clean_axes(ax, ygrid=False); ax.grid(axis="x", alpha=0.3); panel_letter(ax, "b")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- figure five

def create_figure_5_rmsd(rmsd_dir="rmsd_arrays"):
    """Symmetry-corrected ligand RMSD as cumulative distributions, so the
    fraction below any threshold is readable directly. The contact-accuracy
    panel is not optional: the two metrics order the teachers differently and
    RMSD alone would be read as a total ordering."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axes[0]
    for label, n_dec, med_dec, f2_dec, f5_dec, ckey in RMSD_SUMMARY:
        v = np.sort(load_array(os.path.join(rmsd_dir, f"{ckey}.npy")))
        n, med = v.size, float(np.median(v))
        f2, f5 = float((v < 2).mean()), float((v < 5).mean())
        if (n, round(med, 2), round(f2, 3), round(f5, 3)) != \
           (n_dec, round(med_dec, 2), round(f2_dec, 3), round(f5_dec, 3)):
            print(f"note: {ckey} array gives n={n} median={med:.2f} "
                  f"<2A={f2:.3f} <5A={f5:.3f}, declared "
                  f"{n_dec} {med_dec} {f2_dec} {f5_dec}")
        lbl = f"{label}\n(n={n:,})" if len(label) > 14 else f"{label} (n={n:,})"
        ax.plot(v, np.arange(1, n + 1) / n, color=COLORS[ckey], linewidth=2.0,
                label=lbl, linestyle="-")
    for x in (2.0, 5.0):
        ax.axvline(x, color=COLORS["gray"], linestyle=":", linewidth=1.0)
    ax.set_xscale("log"); ax.set_xlim(0.3, 60); ax.set_ylim(0, 1.0)
    ax.set_xticks([2, 5], minor=True)
    ax.set_xticklabels(["2.0", "5.0"], minor=True, fontsize=FONT_SMALL)
    ax.tick_params(axis="x", which="minor", length=4, pad=2, colors=COLORS["gray"])
    ax.set_xlabel("Symmetry-corrected ligand RMSD (A, log scale)", fontsize=FONT_LABEL)
    ax.set_ylabel("Cumulative fraction of systems", fontsize=FONT_LABEL)
    ax.set_title("Pose accuracy\non the crystal tier", fontsize=FONT_LABEL,
                 fontweight="bold", pad=8)
    ax.legend(fontsize=FONT_SMALL, frameon=True, fancybox=True, framealpha=0.7,
              loc="upper left")
    clean_axes(ax); panel_letter(ax, "a", x=-0.13, y=1.20)
    ax = axes[1]
    labs = [l for l, _, _ in CONTACT_ACCURACY][::-1]
    vals = [v for _, v, _ in CONTACT_ACCURACY][::-1]
    cols = [COLORS[c] for _, _, c in CONTACT_ACCURACY][::-1]
    bars = ax.barh(labs, vals, color=cols, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.012, b.get_y() + b.get_height()/2, f"{v:.4f}",
                va="center", ha="left", fontsize=FONT_SMALL)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("Mean contact-set agreement\nwith ground truth", fontsize=FONT_LABEL)
    ax.set_title("Contact accuracy\norders differently", fontsize=FONT_LABEL,
                 fontweight="bold", pad=8)
    ax.tick_params(labelsize=FONT_SMALL)
    clean_axes(ax, ygrid=False); ax.grid(axis="x", alpha=0.3)
    panel_letter(ax, "b", x=-0.30, y=1.20)
    fig.tight_layout()
    return fig

# ----------------------------------------------------------------- figure six

def create_figure_6_reliability(calib_path="calibration_pairs.npy"):
    """Calibration of the shipped per-system reliability annotation."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    pairs = load_array(calib_path)
    pred, obs = pairs[:, 0], pairs[:, 1]
    edges = np.linspace(0, 1, 11)
    idx = np.digitize(pred, edges) - 1
    bx = [pred[idx == i].mean() for i in range(10) if (idx == i).sum() > 20]
    by = [obs[idx == i].mean() for i in range(10) if (idx == i).sum() > 20]
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle=":", linewidth=1.4)
    ax.plot(bx, by, "o-", color=COLORS["chai1"], linewidth=2.0, markersize=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted label accuracy", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed", fontsize=FONT_LABEL)
    ax.set_title(f"Isotonic calibration\nMAE {ISOTONIC_MAE:.4f} against "
                 f"{BASELINE_MAE:.4f} for the mean",
                 fontsize=FONT_LABEL, fontweight="bold", pad=10)
    clean_axes(ax); panel_letter(ax, "a")
    ax = axes[1]
    nom = [c[0] for c in CONFORMAL]; emp = [c[2] for c in CONFORMAL]
    hw = [c[1] for c in CONFORMAL]
    x = np.arange(len(nom)); w = 0.35
    ax.bar(x - w/2, nom, w, color=COLORS["smina"], edgecolor="black", linewidth=0.6,
           label="Nominal")
    ax.bar(x + w/2, emp, w, color=COLORS["chai1"], edgecolor="black", linewidth=0.6,
           label="Empirical")
    for xi, (e, h) in enumerate(zip(emp, hw)):
        ax.text(xi + w/2, e + 0.015, f"{e:.3f}\nhalf-width {h:.4f}", ha="center",
                va="bottom", fontsize=FONT_SMALL)
    ax.set_xticks(x); ax.set_xticklabels([f"{n:.0%}" for n in nom])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Coverage", fontsize=FONT_LABEL)
    ax.set_xlabel("Nominal coverage level", fontsize=FONT_LABEL)
    ax.set_title("Split-conformal coverage", fontsize=FONT_LABEL, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_SMALL, frameon=True, fancybox=True, loc="lower right")
    clean_axes(ax); panel_letter(ax, "b")
    ax = axes[2]
    labs = [f[0] for f in FOLD_BANDS]; ns = [f[1] for f in FOLD_BANDS]
    accs = [f[2] for f in FOLD_BANDS]
    shades = [COLORS["boltz2_msa"], COLORS["chai1"], COLORS["boltz2"], COLORS["smina"]]
    bars = ax.bar(labs, accs, color=shades, edgecolor="black", linewidth=0.6)
    for b, a, n in zip(bars, accs, ns):
        ax.text(b.get_x() + b.get_width()/2, a + 0.012, f"{a:.4f}\n{n:,} pairs",
                ha="center", va="bottom", fontsize=FONT_SMALL)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Pocket RMSD band (A)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean label accuracy", fontsize=FONT_LABEL)
    ax.set_title("Fold-quality tier separation", fontsize=FONT_LABEL, fontweight="bold", pad=10)
    clean_axes(ax); panel_letter(ax, "c")
    fig.tight_layout()
    return fig

# --------------------------------------------------------------- figure seven

def create_figure_7_frame_control():
    """Receptor-conditioning control. A cofolder pose and the deposited crystal
    receptor sit in different frames, so pairing them measures a frame offset
    rather than physical validity."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    checks = list(FRAME_CONTROL.keys())
    pred = [FRAME_CONTROL[c][0] for c in checks]
    dep = [FRAME_CONTROL[c][1] for c in checks]
    y = np.arange(len(checks)); h = 0.34
    ax.barh(y - h/2, pred, h, color=COLORS["chai1"], edgecolor="black",
            linewidth=0.6, label="Conditioned on predicted protein")
    ax.barh(y + h/2, dep, h, color=COLORS["smina"], edgecolor="black",
            linewidth=0.6, label="Conditioned on deposited receptor")
    for yi, (a, b) in enumerate(zip(pred, dep)):
        ax.text(a + 0.012, yi - h/2, f"{a:.4f}", va="center", fontsize=FONT_SMALL)
        ax.text(b + 0.012, yi + h/2, f"{b:.4f}", va="center", fontsize=FONT_SMALL)
    ax.set_yticks(y); ax.set_yticklabels(checks, fontsize=FONT_SMALL)
    ax.set_xlim(0, 1.18); ax.set_xlabel("Pass rate", fontsize=FONT_LABEL)
    ax.set_title("Receptor conditioning control (n = 199)", fontsize=FONT_LABEL,
                 fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_SMALL, frameon=True, fancybox=True, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    clean_axes(ax, ygrid=False); ax.grid(axis="x", alpha=0.3)
    fig.subplots_adjust(bottom=0.30)
    return fig

# --------------------------------------------------------------- figure eight

def _panel_support(ax, rows, title, letter):
    k = [str(r[0]) for r in rows]; n = [r[1] for r in rows]
    shades = [COLORS["smina"], COLORS["boltz2"], COLORS["chai1"]]
    bars = ax.bar(k, n, color=shades, edgecolor="black", linewidth=0.6)
    total = sum(n)
    for b, v, r in zip(bars, n, rows):
        txt = f"{v:,}\n{100*v/total:.1f}%"
        if r[2] is not None:
            txt += f"\nprecision {r[2]:.4f}"
        ax.text(b.get_x() + b.get_width()/2, v * 1.06, txt, ha="center",
                va="bottom", fontsize=FONT_SMALL)
    ax.set_yscale("log"); ax.set_ylim(top=max(n) * 25)
    ax.set_xlabel("Teachers asserting the contact", fontsize=FONT_LABEL)
    ax.set_ylabel("Residue rows", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_LABEL, fontweight="bold", pad=10)
    clean_axes(ax)
    if letter:
        panel_letter(ax, letter)

def create_figure_8_support():
    """Per-residue teacher support and the precision it buys, both tiers."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _panel_support(axes[0], SUPPORT_CRYSTAL, "Crystal tier", "a")
    _panel_support(axes[1], SUPPORT_CORPUS, "Corpus tier", "b")
    annot_box(axes[0], 0.97, 0.97, f"union recall {UNION_RECALL:.4f}")
    annot_box(axes[1], 0.97, 0.97, "precision is not defined here:\n"
                                   "the corpus has no ground truth")
    fig.tight_layout()
    return fig

def create_figure_8_crystal_only():
    """Crystal tier alone. The corpus panel carries a distribution with no
    precision attached, so it is worth dropping where space is tight."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    _panel_support(ax, SUPPORT_CRYSTAL, "Per-residue teacher support, crystal tier", "")
    annot_box(ax, 0.97, 0.97, f"union recall {UNION_RECALL:.4f}")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- figure nine

def create_figure_9_splits():
    """Distributional separation across the split families, measured rather than
    assumed. Panel a is the ligand axis, panel b the protein axis against the
    protein-length baseline it has to be read against: a protein-axis AUC near
    the baseline means the fold assignment did not make proteins distinguishable
    beyond what length alone already does."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(SPLIT_FAMILIES)); w = 0.36

    ax = axes[0]
    for i, tier in enumerate(("corpus", "crystal")):
        v = [SPLIT_DIAGNOSTICS[tier][f]["lig"] for f in SPLIT_FAMILIES]
        b = ax.bar(x + (i - 0.5) * w, v, w, color=COLORS[tier], edgecolor="black",
                   linewidth=0.6, label=f"{tier} ({SPLIT_COUNTS[tier]} configurations)")
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width()/2, vv + 0.012, f"{vv:.2f}",
                    ha="center", va="bottom", fontsize=FONT_SMALL - 1)
    ax.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=1.2)
    ax.text(len(SPLIT_FAMILIES) - 0.4, 0.515, "chance", fontsize=FONT_SMALL,
            color=COLORS["gray"], ha="right")
    ax.set_xticks(x); ax.set_xticklabels(SPLIT_FAMILIES)
    ax.set_ylim(0.4, 1.08)
    ax.set_xlabel("Split family", fontsize=FONT_LABEL)
    ax.set_ylabel("Ligand two-sample classifier AUC", fontsize=FONT_LABEL)
    ax.set_title("Ligand axis", fontsize=FONT_LABEL, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_SMALL, frameon=True, fancybox=True, loc="upper left")
    clean_axes(ax); panel_letter(ax, "a")

    ax = axes[1]
    for i, tier in enumerate(("corpus", "crystal")):
        v = [SPLIT_DIAGNOSTICS[tier][f]["prot"] for f in SPLIT_FAMILIES]
        base = [SPLIT_DIAGNOSTICS[tier][f]["base"] for f in SPLIT_FAMILIES]
        b = ax.bar(x + (i - 0.5) * w, v, w, color=COLORS[tier], edgecolor="black",
                   linewidth=0.6, label=f"{tier}")
        ax.plot(x + (i - 0.5) * w, base, "_", color=COLORS["hatch_orange"],
                markersize=16, markeredgewidth=2.2,
                label="length-only baseline" if i == 0 else None)
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width()/2, vv + 0.012, f"{vv:.2f}",
                    ha="center", va="bottom", fontsize=FONT_SMALL - 1)
    ax.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(SPLIT_FAMILIES)
    ax.set_ylim(0.4, 1.08)
    ax.set_xlabel("Split family", fontsize=FONT_LABEL)
    ax.set_ylabel("Protein two-sample classifier AUC", fontsize=FONT_LABEL)
    ax.set_title("Protein axis, against its baseline", fontsize=FONT_LABEL,
                 fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_SMALL, frameon=True, fancybox=True, loc="upper left")
    clean_axes(ax); panel_letter(ax, "b")

    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ table aid

def emit_applicability_table():
    """Table 5 as LaTeX booktabs, generated so it cannot drift from the figures."""
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Property & Three-teacher core & Non-core \\", r"\midrule"]
    for prop, core, non in APPLICABILITY:
        fmt = "{:,.1f}" if isinstance(core, float) else "{:,}"
        lines.append(f"{prop} & {fmt.format(core)} & {fmt.format(non)} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{3}{l}{\emph{Ligand classes with zero core fraction}} \\"]
    for ccd, n in ZERO_CORE_LIGANDS:
        lines.append(f"\\quad {ccd} & 0 & {n:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)

# ------------------------------------------------------------------------ cli

FIGURES = {
    "fig1_workflow":        create_figure_1_workflow,
    "fig2_coverage":        create_figure_2_coverage,
    "fig3_entity_space":    create_figure_3_entity_space,
    "fig3_protein_only":    create_figure_3_protein_only,
    "fig4_posebusters":     create_figure_4_posebusters,
    "fig5_rmsd":            create_figure_5_rmsd,
    "fig6_reliability":     create_figure_6_reliability,
    "fig7_frame_control":   create_figure_7_frame_control,
    "fig8_support":         create_figure_8_support,
    "fig8_crystal_only":    create_figure_8_crystal_only,
    "fig9_splits":          create_figure_9_splits,
}

# The set that ships. fig2 and fig9 are the swap under discussion, fig7 and the
# single-panel variants are held in reserve.
DEFAULT_SET = ["fig1_workflow", "fig2_coverage", "fig3_entity_space",
               "fig4_posebusters", "fig5_rmsd", "fig6_reliability",
               "fig8_support"]

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--all"]:
        want = list(FIGURES)
    elif args:
        want = args
    else:
        want = DEFAULT_SET
    for name in want:
        if name not in FIGURES:
            print(f"unknown figure: {name}")
            continue
        fig = FIGURES[name]()
        save_figure(fig, name)
        plt.close(fig)
    ensure_output_dir()
    with open(os.path.join(OUTPUT_DIR, "table5_applicability.tex"), "w") as fh:
        fh.write(emit_applicability_table() + "\n")
    print(f"Saved: {OUTPUT_DIR}table5_applicability.tex")
    print()
    audit_pending()
