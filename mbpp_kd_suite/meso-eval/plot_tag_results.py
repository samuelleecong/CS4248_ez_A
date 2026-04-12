"""Generate figures from eval_tag_breakdown.py output.

Produces:
  1. overall_mrr.png       — bar chart of overall MRR per method
  2. by_difficulty.png     — grouped bar chart: MRR per difficulty × method
  3. by_skill.png            — horizontal bar chart: top tags, best method vs control vs teacher
  4. tag_gap_heatmap.png   — heatmap: (method × tag) MRR delta over control

Usage:
    uv run python plot_tag_results.py --input eval_tag_results.json --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "control":      "#9e9e9e",
    "score":        "#64b5f6",
    "hnp":          "#4fc3f7",
    "embed":        "#81c784",
    "bimga":        "#e57373",
    "teacher":      "#ffd54f",
}

DIFFICULTY_ORDER = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]


def method_label(run_name: str) -> str:
    if run_name == "__teacher__":
        return "teacher"
    n = run_name.replace("s1_", "").replace("s2_", "").replace("s3_", "").replace("s4_", "")
    return n


def method_color(label: str) -> str:
    for key, color in METHOD_COLORS.items():
        if key in label.lower():
            return color
    return "#bdbdbd"


def pick_best_per_method(results: dict) -> dict[str, dict]:
    """From all runs, pick the best MRR per method family."""
    families: dict[str, tuple[str, float, dict]] = {}
    for run_name, data in results.items():
        if "error" in data or "overall" not in data:
            continue
        mrr = data["overall"]["MRR"]
        label = method_label(run_name)

        # Family = coarse method name
        if "__teacher__" in run_name:
            family = "teacher"
        elif "control" in label:
            family = "control"
        elif "score" in label:
            family = "score_distill"
        elif "hnp" in label or "hard_neg" in label:
            family = "hard_neg_pair"
        elif "embed" in label:
            family = "embed_distill"
        elif "bimga" in label and "uniform" not in label and "query_only" not in label:
            family = "bimga"
        elif "bimga_uniform" in label:
            family = "bimga_uniform"
        elif "bimga_query_only" in label:
            family = "bimga_query_only"
        else:
            family = label

        if family not in families or mrr > families[family][1]:
            families[family] = (run_name, mrr, data)

    return {family: data for family, (_, _, data) in families.items()}


# ── Figure 1: Overall MRR bar chart ──────────────────────────────────────────

def plot_overall(best: dict[str, dict], output_dir: Path) -> None:
    order = ["control", "score_distill", "hard_neg_pair", "embed_distill", "bimga", "teacher"]
    names, mrrs, r1s, r10s = [], [], [], []
    for family in order:
        if family not in best:
            continue
        o = best[family]["overall"]
        names.append(family.replace("_", "\n"))
        mrrs.append(o["MRR"])
        r1s.append(o["Recall@1"])
        r10s.append(o["Recall@10"])

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, mrrs, width, label="MRR", color="#5c85d6")
    ax.bar(x,         r1s,  width, label="R@1", color="#6ec26e")
    ax.bar(x + width, r10s, width, label="R@10", color="#e07070")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Overall retrieval metrics by method (best config per family)")
    ax.legend()
    ax.set_ylim(0, min(1.0, max(r10s) * 1.2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    out = output_dir / "overall_mrr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: MRR by difficulty ───────────────────────────────────────────────

def plot_by_difficulty(best: dict[str, dict], output_dir: Path) -> None:
    families = [f for f in ["control", "score_distill", "embed_distill", "bimga", "teacher"] if f in best]
    all_diffs = DIFFICULTY_ORDER

    # Filter to difficulties that appear in at least one method
    present_diffs = [d for d in all_diffs if any(
        d in best[f].get("by_difficulty", {}) for f in families
    )]
    if not present_diffs:
        print("  No difficulty breakdown data — skipping by_difficulty.png")
        return

    x = np.arange(len(present_diffs))
    width = 0.15
    offsets = np.linspace(-(len(families) - 1) / 2, (len(families) - 1) / 2, len(families)) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for family, offset in zip(families, offsets):
        by_diff = best[family].get("by_difficulty", {})
        vals = [by_diff.get(d, {}).get("MRR", float("nan")) for d in present_diffs]
        color = method_color(family)
        ax.bar(x + offset, vals, width * 0.9, label=family.replace("_", " "), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in present_diffs], fontsize=9)
    ax.set_ylabel("MRR")
    ax.set_title("MRR by difficulty level")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    out = output_dir / "by_difficulty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Skill gain line chart ──────────────────────────────────────────

def plot_tag_gain_lines(best: dict[str, dict], output_dir: Path, top_n: int = 20) -> None:
    """MRR gain over control per method, one line per method, tags on X-axis.

    Tags sorted by frequency (n) so X-axis goes from common → rare.
    Strips difficulty signal — isolates what each KD method adds.
    """
    family_order = ["score_distill", "hard_neg_pair", "embed_distill", "bimga", "teacher"]
    families = [f for f in family_order if f in best and best[f].get("by_skill")]
    control_tags = best.get("control", {}).get("by_skill", {})
    if not control_tags or not families:
        print("  Insufficient data — skipping tag_gain_lines.png")
        return

    # Sort tags by frequency (most common first), take top_n
    all_tags = {
        tag: data["n"]
        for tag, data in control_tags.items()
    }
    tags = sorted(all_tags.keys(), key=lambda t: all_tags[t], reverse=True)[:top_n]
    if not tags:
        return

    x = np.arange(len(tags))
    markers = ["o", "s", "^", "D", "P"]

    fig, ax = plt.subplots(figsize=(max(12, len(tags) * 0.7), 5))

    for family, marker in zip(families, markers):
        family_tags = best[family].get("by_skill", {})
        gains = []
        for tag in tags:
            if tag in family_tags and tag in control_tags:
                gains.append(family_tags[tag]["MRR"] - control_tags[tag]["MRR"])
            else:
                gains.append(float("nan"))
        color = method_color(family)
        ax.plot(x, gains, marker=marker, label=family.replace("_", " "),
                color=color, linewidth=1.8, markersize=6)

    ax.axhline(0, color="#999", linestyle="--", linewidth=1, label="control (baseline)")

    # Annotate n= below x-axis labels
    ax.set_xticks(x)
    tag_labels = [f"{t}\n(n={all_tags[t]})" for t in tags]
    ax.set_xticklabels(tag_labels, rotation=35, ha="right", fontsize=8)

    ax.set_ylabel("MRR gain over control")
    ax.set_title("KD gain by skill type")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("+%.3f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "tag_gain_lines.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: MRR gain heatmap (method × tag) ────────────────────────────────

def plot_tag_heatmap(best: dict[str, dict], output_dir: Path, top_n: int = 20) -> None:
    families = [f for f in ["score_distill", "hard_neg_pair", "embed_distill", "bimga"] if f in best]
    control_tags = best.get("control", {}).get("by_skill", {})
    if not control_tags or not families:
        print("  Insufficient data — skipping tag_gap_heatmap.png")
        return

    # Tags present in all families and control, sorted by bimga MRR or first available
    ref_family = "bimga" if "bimga" in best else families[-1]
    ref_tags = best[ref_family].get("by_skill", {})
    tags = sorted(ref_tags.keys(), key=lambda t: ref_tags[t]["MRR"], reverse=True)[:top_n]
    if not tags:
        return

    # Build delta matrix: (family × tag), delta = method MRR - control MRR
    matrix = np.full((len(families), len(tags)), np.nan)
    for fi, family in enumerate(families):
        family_tags = best[family].get("by_skill", {})
        for ti, tag in enumerate(tags):
            if tag in family_tags and tag in control_tags:
                matrix[fi, ti] = family_tags[tag]["MRR"] - control_tags[tag]["MRR"]

    cell_w, cell_h = 1.1, 0.7  # inches per cell
    fig_w = max(10, len(tags) * cell_w + 3)
    fig_h = max(3, len(families) * cell_h + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 0.1
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f.replace("_", " ") for f in families], fontsize=10)

    # Annotate cells
    for fi in range(len(families)):
        for ti in range(len(tags)):
            val = matrix[fi, ti]
            if not np.isnan(val):
                ax.text(ti, fi, f"{val:+.3f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="MRR delta vs control", shrink=0.6)
    ax.set_title(f"MRR gain over control by skill type")
    fig.tight_layout()
    out = output_dir / "tag_gap_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Teacher margin vs BiMGA–embed_distill delta ────────────────────

def plot_margin_correlation(best: dict[str, dict], output_dir: Path) -> None:
    """Two-panel chart testing the hypothesis:
       low teacher margin → BiMGA loses to embed_distill.

    Top panel: average teacher margin per skill / difficulty (bar).
    Bottom panel: BiMGA minus embed_distill MRR delta for the same groups.
    If the bars move in opposite directions the hypothesis holds.
    """
    teacher = best.get("__teacher__") or best.get("teacher")
    bimga = best.get("bimga")
    embed = best.get("embed_distill")
    if not teacher or not bimga or not embed:
        print("  Need teacher + bimga + embed_distill — skipping margin_correlation.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Hypothesis: low teacher confidence → BiMGA loses to embed_distill",
        fontsize=12, fontweight="bold"
    )

    for col, (field, margin_key, mrr_key, title) in enumerate([
        ("skill",       "margin_by_skill",       "by_skill",       "By skill type"),
        ("difficulty",  "margin_by_difficulty",  "by_difficulty",  "By difficulty"),
    ]):
        teacher_margins = teacher.get(margin_key, {})
        bimga_mrr       = bimga.get(mrr_key, {})
        embed_mrr       = embed.get(mrr_key, {})

        # Only groups present in all three
        groups = [g for g in teacher_margins if g in bimga_mrr and g in embed_mrr]
        if field == "difficulty":
            order = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]
            groups = [g for g in order if g in groups]
        else:
            # Sort by teacher margin ascending so low-confidence groups appear first
            groups = sorted(groups, key=lambda g: teacher_margins[g]["mean"])

        if not groups:
            continue

        x = np.arange(len(groups))
        margins = [teacher_margins[g]["mean"] for g in groups]
        deltas  = [bimga_mrr[g]["MRR"] - embed_mrr[g]["MRR"] for g in groups]
        labels  = [g.replace(" ", "\n") for g in groups]

        # Top panel: teacher margin per group
        ax_top = axes[0][col]
        bars = ax_top.bar(x, margins, color="#ffd54f", edgecolor="#e0a800", linewidth=0.8)
        ax_top.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        for bar, v in zip(bars, margins):
            ax_top.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(labels, fontsize=8)
        ax_top.set_ylabel("Avg teacher margin")
        ax_top.set_title(f"Teacher confidence — {title}")
        ax_top.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Bottom panel: BiMGA - embed_distill delta
        ax_bot = axes[1][col]
        colors = [METHOD_COLORS["bimga"] if d >= 0 else METHOD_COLORS["embed"] for d in deltas]
        bars = ax_bot.bar(x, deltas, color=colors, edgecolor="#555", linewidth=0.5)
        ax_bot.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        for bar, v in zip(bars, deltas):
            ax_bot.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.001 if v >= 0 else -0.003),
                        f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(labels, fontsize=8)
        ax_bot.set_ylabel("BiMGA − embed_distill MRR")
        ax_bot.set_title(f"BiMGA advantage — {title}")
        ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.3f"))

        # Annotate correlation direction
        if len(margins) > 2:
            corr = float(np.corrcoef(margins, deltas)[0, 1])
            ax_bot.text(0.98, 0.05, f"r = {corr:+.2f}",
                        transform=ax_bot.transAxes, ha="right", fontsize=9,
                        color="#333",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout()
    out = output_dir / "margin_correlation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-skill eval results")
    parser.add_argument("--input", default="eval_tag_results.json")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--top-n-tags", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run eval_tag_breakdown.py first.")
        return

    with input_path.open() as f:
        results = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = pick_best_per_method(results)
    print(f"Method families found: {list(best.keys())}")

    print("\nGenerating figures ...")
    plot_overall(best, output_dir)
    plot_by_difficulty(best, output_dir)
    plot_tag_gain_lines(best, output_dir, top_n=args.top_n_tags)
    plot_tag_heatmap(best, output_dir, top_n=args.top_n_tags)
    plot_margin_correlation(best, output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
