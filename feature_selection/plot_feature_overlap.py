import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NAVY   = "#1E2761"
ACCENT = "#4361ee"
BOTH   = "#2ecc71"
BG     = "#F8F9FF"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = BG

FEATURE_NAMES = [
    "mean radius",        "mean texture",       "mean perimeter",    "mean area",
    "mean smoothness",    "mean compactness",    "mean concavity",    "mean concave pts",
    "mean symmetry",      "mean fractal dim",    "radius SE",         "texture SE",
    "perimeter SE",       "area SE",             "smoothness SE",     "compactness SE",
    "concavity SE",       "concave pts SE",      "symmetry SE",       "fractal dim SE",
    "worst radius",       "worst texture",       "worst perimeter",   "worst area",
    "worst smoothness",   "worst compactness",   "worst concavity",   "worst concave pts",
    "worst symmetry",     "worst fractal dim"
]

original = {6, 7, 13, 15, 17, 19, 20, 21, 22, 23, 24, 27}
ablation = {1, 2, 3, 6, 7, 10, 12, 19, 22, 23, 24}

# Only keep features selected in at least one run
visible = sorted(original | ablation)
n_visible = len(visible)

def category(i):
    in_orig = i in original
    in_abl  = i in ablation
    if in_orig and in_abl:   return 3
    elif in_orig:             return 1
    else:                     return 2

color_map = {1: NAVY, 2: ACCENT, 3: BOTH}

fig, ax = plt.subplots(figsize=(max(10, n_visible * 0.9), 3))
ax.set_facecolor(BG)

for col, feat_idx in enumerate(visible):
    val = category(feat_idx)
    color = color_map[val]
    for row in range(2):
        # for row 0 (original): only draw if in original
        # for row 1 (ablation): only draw if in ablation
        in_this_row = (feat_idx in original) if row == 1 else (feat_idx in ablation)
        cell_color = color if in_this_row else "#F8F9FF"
        rect = mpatches.FancyBboxPatch(
            (col + 0.05, row + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=cell_color, edgecolor="white", linewidth=1.5
        )
        ax.add_patch(rect)
        if in_this_row:
            ax.text(col + 0.5, row + 0.5, str(feat_idx),
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

ax.set_xlim(0, n_visible)
ax.set_ylim(0, 2)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Ablation", "Original"], fontsize=13)
ax.set_xticks([i + 0.5 for i in range(n_visible)])
ax.set_xticklabels([FEATURE_NAMES[i] for i in visible],
                   rotation=45, ha="right", fontsize=9)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0)
ax.spines[:].set_visible(False)

overlap = original & ablation
legend_patches = [
    mpatches.Patch(color=NAVY,   label=f"Original only ({len(original - ablation)})"),
    mpatches.Patch(color=ACCENT, label=f"Ablation only ({len(ablation - original)})"),
    mpatches.Patch(color=BOTH,   label=f"Both ({len(overlap)})"),
]
ax.legend(handles=legend_patches, loc="upper right",
          bbox_to_anchor=(1, 1.45), fontsize=12, ncol=3, frameon=False)

ax.set_title(
    f"Feature Selection Overlap  |  Shared: {sorted(overlap)}",
    fontsize=13, fontweight="bold", color=NAVY, pad=35
)

plt.tight_layout()
plt.savefig("feature_overlap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_overlap.png")