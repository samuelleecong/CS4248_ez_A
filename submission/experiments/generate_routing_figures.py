"""Generate paper-ready routing figures with matplotlib.

Usage:
    cd mbpp_kd_suite
    .venv/Scripts/python.exe ../submission/experiments/generate_routing_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.0,
})

# ── Colors ──
C = {
    'bimga': '#2a7f62',
    'uniform': '#3a8a72',
    'embed': '#3366a5',
    'score': '#c4880b',
    'hnp': '#b04848',
    'soft': '#cc5500',
    'hard': '#7777aa',
    'oracle': '#222222',
    'baseline': '#aaaaaa',
    'control': '#cccccc',
}

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / 'artifacts' / 'results_summary.json'
ROUTE_PATH = Path('artifacts/paper_experiments/20260402_015143/merge_and_route_results.json')

# Try both paths
if ROUTE_PATH.exists():
    with open(ROUTE_PATH) as f:
        route_data = json.load(f)
else:
    alt = OUT_DIR / 'artifacts' / 'merge_and_route_results.json'
    with open(alt) as f:
        route_data = json.load(f)


# ═══════════════════════════════════════════════
# FIGURE 1: Routing Strategy Comparison
# ═══════════════════════════════════════════════
def fig1_strategy_comparison():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ks = [4, 8, 12, 16]
    best_single = 0.298
    hard = [route_data['routing'][f'k={k}']['hard_routing_mrr'] for k in ks]
    soft = [route_data['routing'][f'k={k}']['soft_routing_mrr'] for k in ks]
    oracle = route_data['routing']['k=4']['oracle_mrr']

    x = np.arange(len(ks))
    w = 0.3

    bars_hard = ax.bar(x - w/2, hard, w, color=C['hard'], label='Hard routing', zorder=3)
    bars_soft = ax.bar(x + w/2, soft, w, color=C['soft'], label='Soft routing', zorder=3)

    # Baseline dashed line
    ax.axhline(y=best_single, color=C['baseline'], linestyle='--', linewidth=0.8,
               label=f'Best single model ({best_single:.3f})', zorder=2)

    # Oracle dashed line
    ax.axhline(y=oracle, color=C['oracle'], linestyle=':', linewidth=0.8,
               label=f'Oracle ({oracle:.3f})', zorder=2)

    # Value labels on soft routing bars
    for i, v in enumerate(soft):
        ax.text(x[i] + w/2, v + 0.002, f'{v:.3f}', ha='center', va='bottom',
                fontsize=7, fontfamily='monospace', color=C['soft'])

    # Value labels on hard routing bars
    for i, v in enumerate(hard):
        ax.text(x[i] - w/2, v + 0.002, f'{v:.3f}', ha='center', va='bottom',
                fontsize=7, fontfamily='monospace', color=C['hard'])

    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Test MRR')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0.26, 0.40)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#ddd')
    ax.set_title('(a) Routing Strategy Comparison')

    fig.savefig(OUT_DIR / 'fig_routing_comparison.pdf')
    fig.savefig(OUT_DIR / 'fig_routing_comparison.png')
    plt.close(fig)
    print('Saved fig_routing_comparison.pdf/png')


# ═══════════════════════════════════════════════
# FIGURE 2: Cluster Routing Map (k=8)
# ═══════════════════════════════════════════════
def fig2_cluster_map():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    clusters = route_data['routing']['k=8']['clusters']
    n = len(clusters)

    model_colors = {
        'bimga': C['bimga'],
        'bimga_uniform': C['uniform'],
        'embed_distill': C['embed'],
        'score_distill': C['score'],
        'hard_neg_pair': C['hnp'],
    }

    ids = sorted(clusters.keys(), key=lambda x: int(x))
    sizes = [clusters[i]['size'] for i in ids]
    mrrs = [clusters[i]['val_mrr'] for i in ids]
    models = [clusters[i]['best_model'] for i in ids]
    colors = [model_colors.get(m, '#999') for m in models]

    y = np.arange(n)

    bars = ax.barh(y, mrrs, color=colors, height=0.65, zorder=3)

    # Size annotations on the right
    for i in range(n):
        ax.text(mrrs[i] + 0.015, y[i], f'n={sizes[i]}', va='center',
                fontsize=7, color='#555')

    ax.set_yticks(y)
    ax.set_yticklabels([f'$C_{{{i}}}$' for i in ids])
    ax.set_xlabel('Validation MRR')
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='x', alpha=0.3, zorder=1)
    ax.invert_yaxis()
    ax.set_title('(b) Per-Cluster Best Model Assignment (k = 8)')

    # Legend for model colors
    from matplotlib.patches import Patch
    seen = {}
    for m, c in zip(models, colors):
        if m not in seen:
            seen[m] = c
    handles = [Patch(facecolor=c, label=m) for m, c in seen.items()]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9, edgecolor='#ddd',
              fontsize=7)

    fig.savefig(OUT_DIR / 'fig_cluster_map.pdf')
    fig.savefig(OUT_DIR / 'fig_cluster_map.png')
    plt.close(fig)
    print('Saved fig_cluster_map.pdf/png')


# ═══════════════════════════════════════════════
# FIGURE 3: Model Complementarity
# ═══════════════════════════════════════════════
def fig3_complementarity():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    data = [
        ('Oracle\n(per-query best)', 0.381, C['oracle']),
        ('Soft routing\n(k = 4)', 0.310, C['soft']),
        ('bimga_uniform', 0.298, C['uniform']),
        ('bimga', 0.297, C['bimga']),
        ('embed_distill', 0.282, C['embed']),
        ('hard_neg_pair', 0.268, C['hnp']),
        ('score_distill', 0.266, C['score']),
        ('Control\n(no KD)', 0.198, C['control']),
    ]

    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [d[2] for d in data]

    y = np.arange(len(data))
    bars = ax.barh(y, values, color=colors, height=0.6, zorder=3)

    # Value labels
    for i, v in enumerate(values):
        ax.text(v + 0.004, y[i], f'{v:.3f}', va='center',
                fontsize=7, fontfamily='monospace', color='#444')

    # Baseline dashed line at 0.298
    ax.axvline(x=0.298, color='#999', linestyle='--', linewidth=0.7, zorder=2)

    # Bracket: oracle gap
    bx = 0.395
    y_oracle = 0
    y_single = 2
    ax.annotate('', xy=(bx, y_oracle), xytext=(bx, y_single),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=0.8))
    ax.text(bx + 0.005, (y_oracle + y_single) / 2, '+0.083\noracle gap',
            va='center', fontsize=6.5, color='#555')

    # Bracket: soft routing gain
    bx2 = 0.37
    y_soft = 1
    ax.annotate('', xy=(bx2, y_soft), xytext=(bx2, y_single),
                arrowprops=dict(arrowstyle='<->', color=C['soft'], lw=0.8))
    ax.text(bx2 + 0.004, (y_soft + y_single) / 2, '+0.012',
            va='center', fontsize=6.5, color=C['soft'], fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Test MRR')
    ax.set_xlim(0.15, 0.44)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='x', alpha=0.3, zorder=1)
    ax.invert_yaxis()
    ax.set_title('(c) Model Complementarity: Why Routing Works')

    fig.savefig(OUT_DIR / 'fig_complementarity.pdf')
    fig.savefig(OUT_DIR / 'fig_complementarity.png')
    plt.close(fig)
    print('Saved fig_complementarity.pdf/png')


# ═══════════════════════════════════════════════
# FIGURE 4: Merging Results
# ═══════════════════════════════════════════════
def fig4_merging():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    merge_data = route_data['merging']
    # Group by pair, plot as lines
    pairs = {}
    for r in merge_data:
        key = f"{r['model_a']}+{r['model_b']}"
        if key not in pairs:
            pairs[key] = []
        pairs[key].append((r['alpha'], r['MRR']))

    pair_colors = {
        'bimga+embed_distill': C['embed'],
        'bimga+hard_neg_pair': C['hnp'],
        'bimga+score_distill': C['score'],
        'bimga_uniform+embed_distill': C['uniform'],
        'embed_distill+hard_neg_pair': '#8866aa',
    }

    for pair_name, points in pairs.items():
        points.sort()
        alphas = [p[0] for p in points]
        mrrs = [p[1] for p in points]
        short = pair_name.replace('_distill', '').replace('_neg_pair', '')
        color = pair_colors.get(pair_name, '#999')
        ax.plot(alphas, mrrs, 'o-', markersize=3.5, color=color, label=short, linewidth=1.0)

    # Baseline
    ax.axhline(y=0.2973, color=C['baseline'], linestyle='--', linewidth=0.7,
               label='bimga alone (0.297)')

    ax.set_xlabel(r'Interpolation ratio $\alpha$ (0 = model A, 1 = model B)')
    ax.set_ylabel('Test MRR')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.24, 0.305)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='#ddd', fontsize=7, ncol=2)
    ax.set_title('(d) Model Merging: Weight Interpolation')

    fig.savefig(OUT_DIR / 'fig_merging.pdf')
    fig.savefig(OUT_DIR / 'fig_merging.png')
    plt.close(fig)
    print('Saved fig_merging.pdf/png')


if __name__ == '__main__':
    fig1_strategy_comparison()
    fig2_cluster_map()
    fig3_complementarity()
    fig4_merging()
    print(f'\nAll figures saved to: {OUT_DIR}')
