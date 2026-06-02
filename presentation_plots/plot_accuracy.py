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
    """Read lines, stripping Windows carriage returns."""
    with open(filepath, encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\r\n") for line in f]


def parse_baseline(lines):
    pattern = re.compile(r"we get an accuracy of ([\d.]+)")
    for line in lines:
        m = pattern.search(line)
        if m:
            return float(m.group(1)) * 100
    return None


def parse_overall_best_n(lines):
    """Get the number of features in the overall best subset."""
    pattern = re.compile(r"Overall best accuracy was ([\d.]+) with feature subset \{([^}]+)\}")
    for line in lines:
        m = pattern.search(line)
        if m:
            return len(m.group(2).split(","))
    return None


def parse_log(filepath):
    lines = read_lines(filepath)
    baseline = parse_baseline(lines)
    best_n = parse_overall_best_n(lines)

    all_steps = []
    pattern = re.compile(r"^best accuracy was ([\d.]+) with feature\(s\) \{([^}]+)\}")
    for line in lines:
        m = pattern.match(line)
        if m:
            acc = float(m.group(1)) * 100
            n = len(m.group(2).split(","))
            all_steps.append((n, acc))

    if not all_steps:
        return baseline, []

    counts = [s[0] for s in all_steps]
    is_forward = len(counts) > 1 and counts[1] > counts[0]

    if is_forward and best_n:
        # Keep one step per depth 1,2,...,best_n in order
        clean = []
        expected = 1
        for n, acc in all_steps:
            if n == expected:
                clean.append((n, acc))
                expected += 1
            if expected > best_n:
                break
        steps = clean
    else:
        # Backward: deduplicate by feature count, keep first occurrence
        seen = {}
        for n, acc in all_steps:
            if n not in seen:
                seen[n] = acc
        steps = sorted(seen.items(), key=lambda x: x[0], reverse=True)

    return baseline, steps


def plot_forward(baseline, steps, outfile="forward_accuracy.png"):
    xs = [s[0] for s in steps]
    ys = [s[1] for s in steps]
    peak_idx = ys.index(max(ys))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(xs, ys, color=ACCENT, linewidth=2.5, marker="o",
            markersize=8, markerfacecolor=ACCENT, zorder=3)

    for i, (x, y) in enumerate(zip(xs, ys)):
        offset = 13 if i % 2 == 0 else -20
        ax.annotate(f"{y:.2f}%", xy=(x, y), xytext=(0, offset),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#1a1a2e")

    if baseline:
        ax.axhline(baseline, color=GRAY, linewidth=1.2, linestyle="--",
                   label=f"Baseline ({baseline:.2f}%)")
        ax.legend(fontsize=11)

    ax.plot(xs[peak_idx], ys[peak_idx], "o", color=ACCENT,
            markersize=13, markeredgecolor=NAVY, markeredgewidth=2.5, zorder=4)

    ax.set_xlabel("Number of Features", fontsize=13)
    ax.set_ylabel("LOO-CV Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs. Number of Features — Forward Selection",
                 fontsize=15, fontweight="bold", color=NAVY, pad=15)
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(min(ys) - 4, 101)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_backward(baseline, steps, outfile="backward_accuracy.png"):
    xs = [s[0] for s in steps]
    ys = [s[1] for s in steps]

    if baseline and xs[0] != 30:
        xs = [30] + xs
        ys = [baseline] + ys

    peak_idx = ys.index(max(ys))

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(xs, ys, color=NAVY, linewidth=2.5, marker="o",
            markersize=8, markerfacecolor=NAVY, zorder=3)

    for i, (x, y) in enumerate(zip(xs, ys)):
        offset = 13 if i % 2 == 0 else -20
        ax.annotate(f"{y:.2f}%", xy=(x, y), xytext=(0, offset),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#1a1a2e")

    if baseline:
        ax.axhline(baseline, color=GRAY, linewidth=1.2, linestyle="--",
                   label=f"Baseline ({baseline:.2f}%)")
        ax.legend(fontsize=11)

    ax.plot(xs[peak_idx], ys[peak_idx], "o", color=NAVY,
            markersize=13, markeredgecolor=ACCENT, markeredgewidth=2.5, zorder=4)

    ax.set_xlabel("Number of Features Remaining", fontsize=13)
    ax.set_ylabel("LOO-CV Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs. Number of Features — Backward Elimination",
                 fontsize=15, fontweight="bold", color=NAVY, pad=15)
    ax.set_xticks(xs)
    ax.invert_xaxis()
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(min(ys) - 2, 101)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_comparison(fwd_baseline, fwd_steps, bwd_steps, outfile="accuracy_comparison.png"):
    fwd_best = max(s[1] for s in fwd_steps)
    bwd_best = max(s[1] for s in bwd_steps)
    baseline = fwd_baseline

    # Forward Selection: keep your original logic or match peak
    fwd_n = max(s[0] for s in fwd_steps if s[1] == fwd_best)
    
    # FIX: Get the feature count of the final state where the algorithm actually stopped.
    # Instead of min(), we look at the last item in the chronological steps list 
    # where the maximum accuracy was sustained.
    bwd_n = [s[0] for s in bwd_steps if s[1] == bwd_best][-1]

    labels = ["Baseline\n(all 30 features)",
              f"Forward Selection\n({fwd_n} features)",
              f"Backward Elimination\n({bwd_n} features)"]
    values = [baseline, fwd_best, bwd_best]
    colors = [GRAY, ACCENT, NAVY]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color="#1a1a2e")

    ax.set_ylabel("LOO-CV Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy: Baseline vs. Feature-Selected",
                 fontsize=15, fontweight="bold", color=NAVY, pad=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(min(values) - 3, 101)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


# Main
if len(sys.argv) != 3:
    print("Usage: python plot_accuracy.py forward_log.txt backward_log.txt")
    sys.exit(1)

fwd_file = sys.argv[1]
bwd_file = sys.argv[2]

fwd_baseline, fwd_steps = parse_log(fwd_file)
bwd_baseline, bwd_steps = parse_log(bwd_file)

print(f"Forward  — baseline: {fwd_baseline:.2f}%, best: {max(s[1] for s in fwd_steps):.2f}%, steps found: {len(fwd_steps)}")
print(f"Backward — baseline: {bwd_baseline:.2f}%, best: {max(s[1] for s in bwd_steps):.2f}%, steps found: {len(bwd_steps)}")

plot_forward(fwd_baseline, fwd_steps)
plot_backward(bwd_baseline, bwd_steps)
plot_comparison(fwd_baseline, fwd_steps, bwd_steps)