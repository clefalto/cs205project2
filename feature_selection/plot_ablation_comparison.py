import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import sys

NAVY   = "#1E2761"
ACCENT = "#4361ee"
GRAY   = "#9CA3AF"
BG     = "#F8F9FF"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = BG


def read_lines(filepath):
    with open(filepath, encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\r\n") for line in f]


def parse_forward_steps(lines):
    steps = []
    pattern = re.compile(r"^best accuracy was ([\d.]+) with feature\(s\) \{([^}]+)\}")
    for line in lines:
        m = pattern.match(line)
        if m:
            acc = float(m.group(1)) * 100
            n = len(m.group(2).split(","))
            steps.append((n, acc))
    return steps


def parse_baseline(lines):
    pattern = re.compile(r"we get an accuracy of ([\d.]+)")
    for line in lines:
        m = pattern.search(line)
        if m:
            return float(m.group(1)) * 100
    return None


def plot_comparison(orig_file, ablation_file, outfile="ablation_comparison.png"):
    orig_lines = read_lines(orig_file)
    abl_lines  = read_lines(ablation_file)

    orig_steps = [(n, a) for n, a in parse_forward_steps(orig_lines) if n <= 12]
    abl_steps  = [(n, a) for n, a in parse_forward_steps(abl_lines)  if n <= 12]
    baseline   = parse_baseline(orig_lines)

    orig_xs = [s[0] for s in orig_steps]
    orig_ys = [s[1] for s in orig_steps]
    abl_xs  = [s[0] for s in abl_steps]
    abl_ys  = [s[1] for s in abl_steps]

    orig_peak_idx = orig_ys.index(max(orig_ys))
    abl_peak_idx  = abl_ys.index(max(abl_ys))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(orig_xs, orig_ys, color=NAVY, linewidth=2.5, marker="o",
            markersize=7, markerfacecolor=NAVY, zorder=3,
            label="Original (30 features)")

    ax.plot(abl_xs, abl_ys, color=ACCENT, linewidth=2.5, marker="o",
            markersize=7, markerfacecolor=ACCENT, zorder=3,
            linestyle="--", label="Ablation (excl. features 27, 13, 21)")

    if baseline:
        ax.axhline(baseline, color=GRAY, linewidth=1.2, linestyle=":",
                   label=f"Baseline ({baseline:.1f}%)")

    ax.plot(orig_xs[orig_peak_idx], orig_ys[orig_peak_idx], "o",
            color=NAVY, markersize=13, markeredgecolor="white",
            markeredgewidth=2, zorder=5)
    ax.annotate(f"{orig_ys[orig_peak_idx]:.1f}%",
                xy=(orig_xs[orig_peak_idx], orig_ys[orig_peak_idx]),
                xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=13, color=NAVY, fontweight="bold")

    ax.plot(abl_xs[abl_peak_idx], abl_ys[abl_peak_idx], "o",
            color=ACCENT, markersize=13, markeredgecolor="white",
            markeredgewidth=2, zorder=5)
    ax.annotate(f"{abl_ys[abl_peak_idx]:.1f}%",
                xy=(abl_xs[abl_peak_idx], abl_ys[abl_peak_idx]),
                xytext=(0, -20), textcoords="offset points",
                ha="center", fontsize=13, color=ACCENT, fontweight="bold")

    ax.set_xlabel("Number of Features", fontsize=13)
    ax.set_ylabel("LOO-CV Accuracy (%)", fontsize=13)
    ax.set_title("Forward Selection: Original vs. Ablation Study",
                 fontsize=13, fontweight="bold", color=NAVY, pad=15)

    all_xs = sorted(set(orig_xs) | set(abl_xs))
    ax.set_xticks(all_xs)
    ax.tick_params(axis="both", labelsize=13)
    all_ys = orig_ys + abl_ys
    ax.set_ylim(min(all_ys) - 2, 100)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(fontsize=13, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


if len(sys.argv) != 3:
    print("Usage: python plot_ablation_comparison.py original_log.txt ablation_log.txt")
    sys.exit(1)

plot_comparison(sys.argv[1], sys.argv[2])