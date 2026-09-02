#!/usr/bin/env python3
"""PLI-Parallax Data Descriptor figure generation.

Style inherited from THEOBROMA generate_figures_v135.py: FONT_* and LW_*
constants, save_figure, draw_box / draw_io_box / draw_arrow,
apply_unspec_hatch, and the axis-hygiene conventions (top/right spines hidden,
y-grid at alpha 0.3, set_axisbelow, frameon+fancybox legends, thousands
separators). The THEOBROMA icon is not carried over: Scientific Data titles
cannot carry dataset brand names and a descriptor should not carry a logo.

PALETTE
    Gold marks the two data tiers, blue the processing this work performs, grey
    the sources and the deposited output. Within blue, teachers run a four-step
    ramp ordered by architectural distance from the physics-based end: smina
    lightest, Boltz-2 single-sequence, Chai-1, Boltz-2 MSA darkest. The ramp is
    stated in every caption so it does not read as arbitrary. Experimental
    ground truth is black, since it is a reference rather than a teacher.

CAPITALISATION
    Sentence case throughout: first word and proper nouns only. Product names
    as their owners write them (Chai-1, Boltz-2, smina, BioLiP2, AlphaFold,
    MMseqs2). Field names monospace lowercase. Angstrom written as a bare A.

VALUE PROVENANCE
    REAL      verified against the deposit or a logged analysis run.
    PENDING   a stand-in chosen so the panel renders at final size and shape.
              Every one is suffixed [!] in any rendered label and collected by
              audit_pending(). Swap the value, re-run, done.

Every figure accepts an input path, checks for it, and falls back to a
documented synthetic array when absent, so the figure set can be laid out and
the manuscript finished while generation is still running.

Target: Nature Scientific Data. Panel letters lower-case bold top-left, since
the journal wants merged multi-panel files with panels labelled a, b, c.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- style block

COLORS = {
    # semantic roles
    "given":        "#D9D9D9",   # sources and deposited outputs
    "tier":         "#FFCC00",   # both data tiers, one colour by design
    "produced":     "#6C99D8",   # processing steps, i.e. what this work does
    # teacher ramp, light to dark by architectural distance from physics
    "smina":        "#A9C9F0",
    "boltz2":       "#6C99D8",
    "chai1":        "#2B5EA7",
    "boltz2_msa":   "#1B3F73",
    "groundtruth":  "#2C2C2C",
    # accents
    "hatch_orange": "#F5A623",
    "black":        "#2C2C2C",
    "gray":         "#888888",
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
INPUT_DIR = "."
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
        print("no pending values; figures are submission-ready")
        return
    print(f"PENDING VALUES ({len(PENDING_LOG)}) -- resolve before submission:")
    for k in PENDING_LOG:
        print(f"  [!] {k}")

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
    """Standard annotation box, used identically in every figure."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FONT_SMALL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="0.8", linewidth=0.8))

# ------------------------------------------------------------------- the data

CORPUS_COVERAGE = [                      # REAL, verified against deposit_v3
    ("smina",                    31713, "smina"),
    ("Boltz-2 single-sequence",  23494, "boltz2"),
    ("Chai-1",                   23485, "chai1"),
    ("Boltz-2 MSA (pilot)",        161, "boltz2_msa"),
]
CORPUS_TRIPLE = 23451
CORPUS_UNION = 31746
N_PROTEINS = 906
N_PROTEINS_COFOLD = 678
N_LIGANDS = 31064

CRYSTAL_COVERAGE = [
    ("Ground truth (BioLiP2)",   19350, "groundtruth"),
    ("Chai-1",                    9410, "chai1"),
    ("Boltz-2 single-sequence",   8725, "boltz2"),
    ("smina (corrected)",         7161, "smina"),
    ("Boltz-2 MSA",               1755, "boltz2_msa"),
]
CRYSTAL_CORE = 7161
CRYSTAL_MSA_PROJECTED = pending("crystal Boltz-2 MSA coverage after fold", 6750)

PROTEIN_CLUSTERS = {0.30: 886, 0.40: 896, 0.50: 900}
PROTEIN_CLUSTER_SIZES = {1: 886, 2: 10}
SCAFFOLD_N = 15172
SCAFFOLD_SINGLETON_FRAC = 0.7068
SCAFFOLD_LARGEST = 274
SCAFFOLD_TOP10_FRAC = 0.041
SCAFFOLD_TOP50_FRAC = 0.098
BUTINA_N = 7933
TANIMOTO_ABOVE_030 = 0.0011

PB_CONTROL = 0.633
PB_RATES = [
    ("Boltz-2 MSA",              0.536, 1755, "boltz2_msa"),
    ("Chai-1",                   0.393, 9409, "chai1"),
    ("Boltz-2 single-sequence",  0.254, 8725, "boltz2"),
    ("Boltz-2 multimer, empty",  0.254, 3655, "boltz2"),
    ("smina (corrected)",
     pending("smina PoseBusters PB-valid rate, corrected arm", 0.31), 7161, "smina"),
]
# Panel b lists the same teachers as panel a. smina's failure fractions are from
# the defective arm and are marked pending rather than omitted, since a silently
# absent teacher is worse than a flagged one.
PB_FAILMODES = {
    "min. distance to protein": {"Ground truth": 0.263, "Boltz-2 MSA": 0.289,
                                 "Chai-1": 0.501, "Boltz-2 single-seq": 0.688,
                                 "smina": pending("smina min-distance failure fraction", 0.55)},
    "volume overlap":           {"Ground truth": 0.077, "Boltz-2 MSA": 0.061,
                                 "Chai-1": 0.087, "Boltz-2 single-seq": 0.241,
                                 "smina": pending("smina volume-overlap failure fraction", 0.32)},
    "InChI convertible":        {"Ground truth": 0.051, "Boltz-2 MSA": 0.195,
                                 "Chai-1": 0.223, "Boltz-2 single-seq": 0.183,
                                 "smina": pending("smina InChI failure fraction", 0.074)},
    "sanitization":             {"Ground truth": 0.000, "Boltz-2 MSA": 0.195,
                                 "Chai-1": 0.168, "Boltz-2 single-seq": 0.182,
                                 "smina": pending("smina sanitization failure fraction", 0.070)},
}

RMSD_SUMMARY = [
    ("Boltz-2 MSA",              456, 3.12, 0.428, 0.550, "boltz2_msa"),
    ("Chai-1",                  9519, 5.24, 0.320, 0.492, "chai1"),
    ("Boltz-2 single-sequence", 8820, 11.41, 0.045, 0.184, "boltz2"),
    ("smina (corrected)", 6485,
     4.15,
     0.289,
     0.565, "smina"),
]
CONTACT_ACCURACY = [                      # ordered as measured
    ("smina (corrected)",        0.5590, "smina"),
    ("Chai-1",                   0.4651, "chai1"),
    ("Boltz-2 single-sequence",  0.1862, "boltz2"),
]

FOLD_BANDS = [
    ("< 1",     3362, 0.6128),
    ("1 to 2",  2534, 0.5406),
    ("2 to 5",  2688, 0.4277),
    (">= 5",   11155, 0.2047),
]
ISOTONIC_MAE = 0.0814
BASELINE_MAE = 0.1734
CONFORMAL = [(0.90, 0.1526, 0.900), (0.80, 0.1200, 0.800)]

SUPPORT_CRYSTAL = [(1, 147966, 0.2175), (2, 45268, 0.7265), (3, 16200, 0.9356)]
SUPPORT_CORPUS = [(1, 719817, None), (2, 107124, None), (3, 21901, None)]
UNION_RECALL = 0.8828

FRAME_CONTROL = {
    "protein-ligand max. distance": (1.0000, 0.2843),
    "min. distance to protein":     (0.5528, 0.7665),
    "volume overlap":               (0.9246, 0.8274),
}

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
                "BioLiP2  |  AlphaFold DB v6  |  PDB  |  BindingDB  |  Bernett protein universe",
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
        ("per-arm settings in Table 2", False, FONT_DETAIL),
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
                "18 Parquet tables  |  HDF5 coordinate store  |  375 split configurations\n"
                "per-system reliability  |  per-residue support  |  manifest, Croissant, field dictionary",
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

def create_figure_3_entity_space(protein_clusters_path="protein_cluster_sizes.npy",
                                 scaffold_sizes_path="scaffold_cluster_sizes.npy"):
    """Non-redundancy of both entity axes. The left panel is a bar chart because
    the distribution takes two values; the right is a complementary cumulative
    distribution, which reads the singleton fraction and the tail directly
    without the binning artefacts a log-binned histogram introduces."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    p = os.path.join(INPUT_DIR, protein_clusters_path)
    if os.path.exists(p):
        prot = np.load(p)
    else:
        print(f"WARNING: {protein_clusters_path} missing; reconstructing from known counts")
        prot = np.array([1] * PROTEIN_CLUSTER_SIZES[1] + [2] * PROTEIN_CLUSTER_SIZES[2])
    ax = axes[0]
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
    clean_axes(ax); panel_letter(ax, "a", x=-0.16, y=1.16)
    s = os.path.join(INPUT_DIR, scaffold_sizes_path)
    if os.path.exists(s):
        scaf = np.load(s)
    else:
        print(f"WARNING: {scaffold_sizes_path} missing; synthetic power law matched to known moments")
        rng = np.random.default_rng(11)
        n_single = int(SCAFFOLD_N * SCAFFOLD_SINGLETON_FRAC)
        tail = np.clip((rng.pareto(1.15, SCAFFOLD_N - n_single) + 1).astype(int),
                       2, SCAFFOLD_LARGEST)
        scaf = np.concatenate([np.ones(n_single, int), tail])
    ax = axes[1]
    sizes = np.sort(scaf)
    x = np.unique(sizes)
    ccdf = np.array([(sizes >= v).mean() for v in x])
    ax.step(x, ccdf, where="post", color=COLORS["chai1"], linewidth=2.0)
    ax.axhline(1.0 - SCAFFOLD_SINGLETON_FRAC, color=COLORS["hatch_orange"],
               linestyle="--", linewidth=1.2,
               label=f"{100*(1-SCAFFOLD_SINGLETON_FRAC):.1f}% have 2 or more,\ni.e. {100*SCAFFOLD_SINGLETON_FRAC:.1f}% are singletons")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Scaffold cluster size, n ligands", fontsize=FONT_LABEL)
    ax.set_ylabel("Fraction of scaffolds\nwith at least n ligands", fontsize=FONT_LABEL)
    ax.set_title(f"Ligand scaffolds, {SCAFFOLD_N:,} from {N_LIGANDS:,}\n"
                 f"largest cluster {SCAFFOLD_LARGEST}",
                 fontsize=FONT_LABEL, fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=FONT_SMALL - 1, frameon=True,
              fancybox=True, framealpha=0.8)
    annot_box(ax, 0.97, 0.62, f"top 10 scaffolds {100*SCAFFOLD_TOP10_FRAC:.1f}%\n"
                              f"top 50 scaffolds {100*SCAFFOLD_TOP50_FRAC:.1f}%\n"
                              f"{100*TANIMOTO_ABOVE_030:.2f}% of pairs above T 0.30")
    clean_axes(ax); panel_letter(ax, "b", x=-0.20, y=1.16)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- figure four

def create_figure_4_posebusters():
    """Physical validity per teacher against the crystal self-docking control.
    The control is mandatory: 21% of experimental crystal ligands also fail
    under this check configuration, so a teacher rate is only interpretable
    relative to it."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    rows = sorted(PB_RATES, key=lambda r: r[1])
    names = [pmark(r[0]) if "smina" in r[0] else r[0] for r in rows]
    rates = [r[1] for r in rows]; ns = [r[2] for r in rows]
    colors = [COLORS[r[3]] for r in rows]
    bars = ax.barh(names, rates, color=colors, edgecolor="black", linewidth=0.6)
    for b, r, n in zip(bars, rates, ns):
        ax.text(b.get_width() + 0.012, b.get_y() + b.get_height()/2,
                f"{r:.3f}  (n={n:,})", va="center", ha="left", fontsize=FONT_SMALL)
    ax.axvline(PB_CONTROL, color=COLORS["black"], linestyle="--", linewidth=0.9,
               alpha=0.5)
    ax.text(PB_CONTROL + 0.02, 0.03, f"crystal self-docking\ncontrol {PB_CONTROL:.3f}",
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
    glabel = [g if g != "smina" else pmark(g) for g in groups]
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
    for label, n, med, f2, f5, ckey in RMSD_SUMMARY:
        path = os.path.join(INPUT_DIR, rmsd_dir, f"{ckey}.npy")
        if os.path.exists(path):
            v = np.load(path)
        else:
            print(f"WARNING: {path} missing; lognormal matched to median and P(<2 A)")
            rng = np.random.default_rng(abs(hash(ckey)) % 2**31)
            sigma = max((np.log(med) - np.log(2.0)) / 0.9, 0.25)
            v = np.clip(rng.lognormal(np.log(med), sigma, n), 0.15, 80)
        v = np.sort(v)
        lbl = f"{label}\n(n={n:,})" if len(label) > 14 else f"{label} (n={n:,})"
        if "smina" in label:
            lbl = pmark(lbl)
        ax.plot(v, np.arange(1, len(v)+1)/len(v), color=COLORS[ckey], linewidth=2.0,
                label=lbl, linestyle="--" if "smina" in label else "-")
    for x in (2.0, 5.0):
        ax.axvline(x, color=COLORS["gray"], linestyle=":", linewidth=1.0)
    ax.set_xscale("log"); ax.set_xlim(0.3, 60); ax.set_ylim(0, 1.0)
    ax.set_xticks([2, 5], minor=True)
    ax.set_xticklabels(["2.0", "5.0"], minor=True, fontsize=FONT_SMALL)
    ax.tick_params(axis="x", which="minor", length=4, pad=2,
                   colors=COLORS["gray"])
    ax.set_xlabel("Symmetry-corrected ligand RMSD (A, log scale)", fontsize=FONT_LABEL)
    ax.set_ylabel("Cumulative fraction of systems", fontsize=FONT_LABEL)
    ax.set_title("Pose accuracy\non the crystal tier", fontsize=FONT_LABEL,
                 fontweight="bold", pad=8)
    ax.legend(fontsize=FONT_SMALL, frameon=True, fancybox=True, framealpha=0.7,
              loc="upper left")
    clean_axes(ax); panel_letter(ax, "a", x=-0.13, y=1.20)
    ax = axes[1]
    labs = [pmark(l) if "smina" in l else l for l, _, _ in CONTACT_ACCURACY][::-1]
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
    p = os.path.join(INPUT_DIR, calib_path)
    if os.path.exists(p):
        pairs = np.load(p); pred, obs = pairs[:, 0], pairs[:, 1]
    else:
        print(f"WARNING: {calib_path} missing; synthetic pairs matched to MAE {ISOTONIC_MAE}")
        rng = np.random.default_rng(3)
        pred = np.clip(rng.beta(2.0, 4.0, 4000), 0.01, 0.99)
        obs = np.clip(pred + rng.normal(0, ISOTONIC_MAE * 1.25, pred.size), 0, 1)
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
        ax.text(b.get_x() + b.get_width()/2, a + 0.012, f"{a:.4f}\nn={n:,}",
                ha="center", va="bottom", fontsize=FONT_SMALL)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Pocket RMSD band (A)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean label accuracy", fontsize=FONT_LABEL)
    ax.set_title("Fold-quality tier separation", fontsize=FONT_LABEL, fontweight="bold", pad=10)
    clean_axes(ax); panel_letter(ax, "c")
    fig.tight_layout()
    return fig

# --------------------------------------------------------- optional figure 7

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

# --------------------------------------------------------- optional figure 8

def create_figure_8_support():
    """Per-residue teacher support and the precision it buys, both tiers."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (rows, title, letter) in zip(axes, [
        (SUPPORT_CRYSTAL, "Crystal tier", "a"),
        (SUPPORT_CORPUS, "Corpus tier", "b"),
    ]):
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
        clean_axes(ax); panel_letter(ax, letter)
    annot_box(axes[0], 0.97, 0.97, f"union recall {UNION_RECALL:.4f}")
    annot_box(axes[1], 0.97, 0.97, "precision is not defined here:\n"
                                   "the corpus has no ground truth")
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
    "fig1_workflow": create_figure_1_workflow,
    "fig2_coverage": create_figure_2_coverage,
    "fig3_entity_space": create_figure_3_entity_space,
    "fig4_posebusters": create_figure_4_posebusters,
    "fig5_rmsd": create_figure_5_rmsd,
    "fig6_reliability": create_figure_6_reliability,
    "fig7_frame_control": create_figure_7_frame_control,
    "fig8_support": create_figure_8_support,
}

if __name__ == "__main__":
    import sys
    want = sys.argv[1:] or list(FIGURES)
    for name in want:
        if name not in FIGURES:
            print(f"unknown figure: {name}"); continue
        fig = FIGURES[name]()
        save_figure(fig, name)
        plt.close(fig)
    ensure_output_dir()
    with open(os.path.join(OUTPUT_DIR, "table5_applicability.tex"), "w") as fh:
        fh.write(emit_applicability_table() + "\n")
    print(f"Saved: {OUTPUT_DIR}table5_applicability.tex")
    print()
    audit_pending()
