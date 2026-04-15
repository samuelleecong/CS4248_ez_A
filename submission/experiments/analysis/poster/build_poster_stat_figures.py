from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1] / 'significance'
OUT = Path(__file__).resolve().parent

ORDER = [
    'control',
    'score_distill',
    'hard_neg_pair',
    'embed_distill',
    'bimga_uniform',
    'bimga',
]

LABELS = {
    'control': 'Control',
    'score_distill': 'ScoreDistill',
    'hard_neg_pair': 'PairDistill',
    'embed_distill': 'EmbedDistill',
    'bimga_uniform': 'BiMGA-U',
    'bimga': 'BiMGA',
}

COLORS = {
    'control': '#9aa3ad',
    'score_distill': '#c9852d',
    'hard_neg_pair': '#c44e52',
    'embed_distill': '#3f7fbf',
    'bimga_uniform': '#58a29b',
    'bimga': '#2f855a',
}

STATUS_TO_NUM = {
    'significant_worse': -1,
    'not_significant': 0,
    'significant_better': 1,
}

sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'savefig.dpi': 220,
})


def load_data():
    run_summary = pd.read_csv(BASE / 'saturated_run_summary.csv')
    rr = pd.read_csv(BASE / 'pairwise_rr_permutation.csv')
    mc = pd.read_csv(BASE / 'pairwise_mcnemar.csv')
    margin = pd.read_csv(BASE / 'pairwise_margin_wilcoxon.csv')
    per_query = pd.read_csv(BASE / 'saturated_per_query.csv')
    return run_summary, rr, mc, margin, per_query


def order_df(df: pd.DataFrame, method_col: str = 'method') -> pd.DataFrame:
    cat = pd.Categorical(df[method_col], categories=ORDER, ordered=True)
    return df.assign(_order=cat).sort_values('_order').drop(columns='_order')




def fmt_score(val: float | int | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'N/A'
    if abs(float(val)) < 5e-4:
        val = 0.0
    return f'{float(val):.3f}'

def save_both(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = OUT / f'{stem}.png'
    svg = OUT / f'{stem}.svg'
    fig.savefig(png, bbox_inches='tight', facecolor='white')
    fig.savefig(svg, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return png, svg


def plot_retrieval(run_summary: pd.DataFrame):
    df = run_summary[['method', 'replay_mrr', 'replay_recall@1', 'replay_recall@10']].copy()
    df = order_df(df)
    metrics = [
        ('replay_mrr', 'Symmetric MRR'),
        ('replay_recall@1', 'Recall@1'),
        ('replay_recall@10', 'Recall@10'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    methods = [LABELS[m] for m in df['method']]
    x = np.arange(len(df))
    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].to_numpy()
        bars = ax.bar(x, vals, color=[COLORS[m] for m in df['method']], edgecolor='black', linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(x, methods, rotation=25, ha='right')
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_ylabel('Score')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.007, fmt_score(val), ha='center', va='bottom', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.suptitle('Pillar 1: Final Symmetric Retrieval Quality on TACO (n=1000 queries)', y=1.04)
    fig.tight_layout()
    return save_both(fig, 'poster_fig01_retrieval_quality')


def plot_alignment(run_summary: pd.DataFrame):
    df = run_summary[['method', 'replay_asym_mrr', 'replay_doc_cosine']].copy()
    df = order_df(df)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharex=True)
    methods = [LABELS[m] for m in df['method']]
    x = np.arange(len(df))

    panels = [
        ('replay_asym_mrr', 'Published Asymmetric MRR', 'Asymmetric MRR'),
        ('replay_doc_cosine', 'Published Document Cosine', 'Doc cosine to teacher'),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        vals = df[col].to_numpy()
        colors = [COLORS[m] for m in df['method']]
        bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), color=colors, edgecolor='black', linewidth=0.5)
        for idx, (bar, val) in enumerate(zip(bars, vals)):
            if np.isnan(val):
                bar.set_facecolor('#f3f3f3')
                bar.set_hatch('//')
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, 'N/A', ha='center', va='bottom', fontsize=10, color='#666666')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, fmt_score(val), ha='center', va='bottom', fontsize=10)
        ax.set_title(title)
        ax.set_xticks(x, methods, rotation=25, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(np.nan_to_num(vals, nan=0.0)) * 1.25 + 0.03)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.suptitle('Pillar 2: Teacher-Space Diagnostics from Published HF Metrics', y=1.04)
    fig.text(0.5, -0.02, 'These are aggregate diagnostics only. Paired significance is unavailable because the saturated HF repos do not publish the per-query fine-tuned teacher targets.', ha='center', fontsize=10, color='#555555')
    fig.tight_layout()
    return save_both(fig, 'poster_fig02_teacher_space_diagnostics')


def plot_margin(per_query: pd.DataFrame):
    df = per_query[['method', 'sym_margin']].copy()
    df['method_label'] = df['method'].map(LABELS)
    df = df[df['method'].isin(ORDER)]
    summary = (
        df.groupby('method', as_index=False)
        .agg(mean_margin=('sym_margin', 'mean'), neg_margin_rate=('sym_margin', lambda s: (s < 0).mean()))
    )
    summary = order_df(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={'width_ratios': [1.35, 1.0]})

    sns.boxplot(
        data=df,
        x='method',
        y='sym_margin',
        hue='method',
        order=ORDER,
        hue_order=ORDER,
        palette=COLORS,
        width=0.58,
        showfliers=False,
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(0, color='black', linewidth=1.0, linestyle='--')
    axes[0].set_title('Per-query hardest-negative margin distribution')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Margin = score(correct) - score(hardest wrong)')
    axes[0].set_xticks(np.arange(len(ORDER)))
    axes[0].set_xticklabels([LABELS[m] for m in ORDER], rotation=25, ha='right')
    axes[0].spines[['top', 'right']].set_visible(False)

    x = np.arange(len(summary))
    bars = axes[1].bar(x, summary['neg_margin_rate'], color=[COLORS[m] for m in summary['method']], edgecolor='black', linewidth=0.5)
    axes[1].set_title('Negative-margin rate by method')
    axes[1].set_ylabel('Fraction of queries with margin < 0')
    axes[1].set_xticks(x, [LABELS[m] for m in summary['method']], rotation=25, ha='right')
    axes[1].set_ylim(0, 1.0)
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)
    for bar, neg_rate, mean_margin in zip(bars, summary['neg_margin_rate'], summary['mean_margin']):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            neg_rate + 0.02,
            f'{fmt_score(neg_rate)}\nmean={fmt_score(mean_margin)}',
            ha='center', va='bottom', fontsize=9,
        )

    fig.suptitle('Pillar 3: Hardest-Negative Margin Behaviour', y=1.02)
    fig.tight_layout()
    return save_both(fig, 'poster_fig03_margin_behaviour')


def plot_significance(rr: pd.DataFrame, mc: pd.DataFrame, margin: pd.DataFrame):
    focus_order = ['control', 'score_distill', 'hard_neg_pair', 'embed_distill', 'bimga_uniform']

    rr_b = rr[(rr['method_a'] == 'bimga') & (rr['method_b'].isin(focus_order))].copy()
    rr_b['method_b'] = pd.Categorical(rr_b['method_b'], categories=focus_order, ordered=True)
    rr_b = rr_b.sort_values('method_b')

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.6), gridspec_kw={'width_ratios': [1.4, 1.0]})

    y = np.arange(len(rr_b))
    colors = [COLORS['bimga'] if s == 'significant_better' else '#b7bcc2' for s in rr_b['status']]
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1.0)
    axes[0].barh(y, rr_b['mean_delta_rr'], color=colors, edgecolor='black', linewidth=0.5, height=0.55)
    xerr = np.vstack([
        rr_b['mean_delta_rr'] - rr_b['ci_low'],
        rr_b['ci_high'] - rr_b['mean_delta_rr'],
    ])
    axes[0].errorbar(rr_b['mean_delta_rr'], y, xerr=xerr, fmt='none', ecolor='black', capsize=4, linewidth=1.1)
    axes[0].set_yticks(y, [f'BiMGA vs {LABELS[m]}' for m in rr_b['method_b']])
    axes[0].set_xlabel('Mean Δ reciprocal rank')
    axes[0].set_title('BiMGA retrieval gain with 95% bootstrap CI')
    axes[0].invert_yaxis()
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].grid(axis='x', linestyle='--', alpha=0.35)
    for yi, row in zip(y, rr_b.itertuples(index=False)):
        axes[0].text(
            row.ci_high + 0.003,
            yi,
            f"adj p={row.adjusted_p_value:.4f}",
            va='center',
            fontsize=9,
            color='#444444',
        )

    def get_status(df: pd.DataFrame, metric: str, baseline: str) -> str:
        row = df[(df['method_a'] == 'bimga') & (df['method_b'] == baseline)]
        if metric is not None:
            row = row[row['metric'] == metric]
        return row.iloc[0]['status']

    heat = pd.DataFrame(index=[LABELS[m] for m in focus_order], columns=['MRR', 'R@1', 'R@10', 'Margin'])
    for baseline in focus_order:
        heat.loc[LABELS[baseline], 'MRR'] = get_status(rr, None, baseline)
        heat.loc[LABELS[baseline], 'R@1'] = get_status(mc, 'sym_correct_at_1', baseline)
        heat.loc[LABELS[baseline], 'R@10'] = get_status(mc, 'sym_correct_at_10', baseline)
        heat.loc[LABELS[baseline], 'Margin'] = get_status(margin, 'sym_margin', baseline)

    heat_num = heat.replace(STATUS_TO_NUM).astype(int)
    cmap = ListedColormap(['#d6604d', '#d9d9d9', '#1a9850'])
    sns.heatmap(
        heat_num,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        cbar=False,
        annot=heat.replace({
            'significant_better': '+',
            'not_significant': 'ns',
            'significant_worse': '-',
        }),
        fmt='',
        linewidths=1,
        linecolor='white',
        ax=axes[1],
        annot_kws={'fontsize': 12, 'fontweight': 'bold'},
    )
    axes[1].set_title('Paired statistical verdicts for BiMGA')
    axes[1].set_xlabel('Test pillar')
    axes[1].set_ylabel('Baseline')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].tick_params(axis='y', rotation=0)
    legend_handles = [
        Patch(facecolor='#1a9850', edgecolor='none', label='significant_better'),
        Patch(facecolor='#d9d9d9', edgecolor='none', label='not_significant'),
        Patch(facecolor='#d6604d', edgecolor='none', label='significant_worse'),
    ]
    axes[1].legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    fig.suptitle('Pillar 4: Statistical Evidence for BiMGA Against Saturated Baselines', y=1.03)
    fig.tight_layout()
    return save_both(fig, 'poster_fig04_bimga_significance')




def plot_rr_margin_distributions(per_query: pd.DataFrame):
    focus = ['score_distill', 'hard_neg_pair', 'bimga']
    df = per_query[per_query['method'].isin(focus)].copy()
    df['method_label'] = df['method'].map(LABELS)

    summary = (
        df.groupby('method', as_index=False)
        .agg(
            rr_mean=('sym_reciprocal_rank', 'mean'),
            rr_median=('sym_reciprocal_rank', 'median'),
            margin_mean=('sym_margin', 'mean'),
            margin_median=('sym_margin', 'median'),
        )
    )
    summary['method'] = pd.Categorical(summary['method'], categories=focus, ordered=True)
    summary = summary.sort_values('method')

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6))

    violin_kwargs = dict(inner=None, linewidth=1.0, cut=0, saturation=0.95)

    sns.violinplot(
        data=df,
        x='method',
        y='sym_reciprocal_rank',
        order=focus,
        hue='method',
        hue_order=focus,
        palette=COLORS,
        dodge=False,
        legend=False,
        ax=axes[0],
        **violin_kwargs,
    )
    sns.boxplot(
        data=df,
        x='method',
        y='sym_reciprocal_rank',
        order=focus,
        width=0.22,
        showcaps=True,
        showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.0, 'zorder': 3},
        whiskerprops={'color': 'black', 'linewidth': 1.0},
        medianprops={'color': 'black', 'linewidth': 1.5},
        ax=axes[0],
    )
    axes[0].set_title('Per-query reciprocal rank distribution')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Reciprocal rank per query')
    axes[0].set_xticks(np.arange(len(focus)), [LABELS[m] for m in focus])
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)

    sns.violinplot(
        data=df,
        x='method',
        y='sym_margin',
        order=focus,
        hue='method',
        hue_order=focus,
        palette=COLORS,
        dodge=False,
        legend=False,
        ax=axes[1],
        **violin_kwargs,
    )
    sns.boxplot(
        data=df,
        x='method',
        y='sym_margin',
        order=focus,
        width=0.22,
        showcaps=True,
        showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.0, 'zorder': 3},
        whiskerprops={'color': 'black', 'linewidth': 1.0},
        medianprops={'color': 'black', 'linewidth': 1.5},
        ax=axes[1],
    )
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.0)
    axes[1].set_title('Per-query hardest-negative margin distribution')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('Margin = score(correct) - score(best wrong)')
    axes[1].set_xticks(np.arange(len(focus)), [LABELS[m] for m in focus])
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)

    for idx, row in enumerate(summary.itertuples(index=False)):
        axes[0].text(idx, 0.995, f"mean={row.rr_mean:.3f}\nmed={row.rr_median:.3f}", ha='center', va='top', fontsize=10)
        axes[1].text(idx, axes[1].get_ylim()[1] * 0.95, f"mean={row.margin_mean:.3f}\nmed={row.margin_median:.3f}", ha='center', va='top', fontsize=10)

    fig.suptitle('Pillar 5: Distribution View of Global Rank vs Local Margin', y=1.03)
    fig.text(0.5, 0.01, 'Left: MRR is the average of these per-query reciprocal-rank values. Right: margin only looks at the correct code versus the single strongest wrong code.', ha='center', fontsize=11, color='#444444')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save_both(fig, 'poster_fig05_rr_margin_distributions')




def plot_overlay_histograms(per_query: pd.DataFrame):
    focus = ['score_distill', 'hard_neg_pair', 'bimga']
    df = per_query[per_query['method'].isin(focus)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2))

    rr_bins = np.linspace(0.0, 1.0, 51)
    margin_vals = df['sym_margin'].to_numpy()
    m_lo = float(np.floor(margin_vals.min() * 20) / 20)
    m_hi = float(np.ceil(margin_vals.max() * 20) / 20)
    margin_bins = np.linspace(m_lo, m_hi, 50)

    for method in focus:
        sub = df[df['method'] == method]
        label = LABELS[method]
        color = COLORS[method]
        axes[0].hist(
            sub['sym_reciprocal_rank'],
            bins=rr_bins,
            histtype='step',
            linewidth=2.2,
            color=color,
            label=label,
        )
        axes[1].hist(
            sub['sym_margin'],
            bins=margin_bins,
            histtype='step',
            linewidth=2.2,
            color=color,
            label=label,
        )

    axes[0].set_title('Overlayed count distribution: reciprocal rank')
    axes[0].set_xlabel('Reciprocal rank per query')
    axes[0].set_ylabel('Number of queries')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)

    axes[1].set_title('Overlayed count distribution: hardest-negative margin')
    axes[1].set_xlabel('Margin = score(correct) - score(best wrong)')
    axes[1].set_ylabel('Number of queries')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.0)
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)

    handles = [Patch(facecolor='none', edgecolor=COLORS[m], linewidth=2, label=LABELS[m]) for m in focus]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    fig.suptitle('Pillar 6: Direct Count Overlay Across Methods', y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return save_both(fig, 'poster_fig06_overlay_counts')


def build_markdown(figures: dict[str, tuple[Path, Path]]):
    md = OUT / 'POSTER_STAT_FIGURES.md'
    text = f"""# Poster Statistical Figures

These figures summarize the saturated TACO evaluation for the six final Hugging Face checkpoints. They are poster-oriented views over the exact replay and significance outputs in `/submission/experiments/analysis/significance`.

## Figure 1. Final Symmetric Retrieval Quality

![Figure 1](./{figures['retrieval'][0].name})

This plot compares the final symmetric retrieval metrics across all six saturated methods. The x-axis is the training method and the y-axis is the metric value. The key message is that BiMGA is the strongest final retriever on all three headline metrics, with BiMGA-uniform as the closest variant.

## Figure 2. Teacher-Space Diagnostics

![Figure 2](./{figures['alignment'][0].name})

This plot shows the published aggregate teacher-space diagnostics from each HF repo: asymmetric MRR and document cosine. The x-axis is the method and the y-axis is the diagnostic value. The key message is that alignment-based methods retain strong compatibility with teacher document space, while score-only and pairwise methods do not.

## Figure 3. Hardest-Negative Margin Behaviour

![Figure 3](./{figures['margin'][0].name})

The left panel plots the per-query hardest-negative margin distribution; the y-axis is `score(correct) - score(hardest wrong)` and the x-axis is the method. The right panel shows the fraction of queries with negative margin. The key message is that PairDistill and ScoreDistill sharpen the strict local margin more than BiMGA, but that local gain does not translate into the best overall retrieval.

## Figure 4. Paired Statistical Evidence for BiMGA

![Figure 4](./{figures['significance'][0].name})

The left panel is a forest plot of mean reciprocal-rank deltas for BiMGA against each baseline, with 95% bootstrap confidence intervals and Holm-adjusted permutation-test p-values. The right panel is a compact significance matrix over the paired tests used in the report: reciprocal rank, Recall@1, Recall@10, and hardest-negative margin. The key message is that BiMGA is significantly better than EmbedDistill, PairDistill, ScoreDistill, and Control on reciprocal-rank retrieval, while the BiMGA vs BiMGA-uniform gap is not significant.

## Figure 5. Reciprocal-Rank and Margin Distributions

![Figure 5](./{figures['rr_margin_dist'][0].name})

The left panel shows the per-query reciprocal-rank distribution for `ScoreDistill`, `PairDistill`, and `BiMGA`. Since `MRR` is just the average of reciprocal rank across queries, this is the most direct distribution view behind the final MRR numbers. The right panel shows the per-query hardest-negative margin distribution for the same three methods. The key message is that `ScoreDistill` and `PairDistill` have slightly better local margin distributions, but `BiMGA` still has the strongest reciprocal-rank distribution overall.

## Figure 6. Overlayed Count Distributions

![Figure 6](./{figures['overlay_counts'][0].name})

This figure puts the three methods on the same axes. The left panel overlays the reciprocal-rank count distributions, so you can directly see which method has more queries at low or high reciprocal-rank values. The right panel does the same for hardest-negative margin. This is the clearest visual for answering: where exactly does BiMGA have more high-quality ranking outcomes, and where exactly do ScoreDistill and PairDistill have more favorable local margins?

## Files

- Retrieval: `{figures['retrieval'][0].name}`, `{figures['retrieval'][1].name}`
- Alignment: `{figures['alignment'][0].name}`, `{figures['alignment'][1].name}`
- Margin: `{figures['margin'][0].name}`, `{figures['margin'][1].name}`
- Significance: `{figures['significance'][0].name}`, `{figures['significance'][1].name}`
- RR vs margin distributions: `{figures['rr_margin_dist'][0].name}`, `{figures['rr_margin_dist'][1].name}`
- Overlay counts: `{figures['overlay_counts'][0].name}`, `{figures['overlay_counts'][1].name}`
"""
    md.write_text(text, encoding='utf-8')
    return md


def main():
    run_summary, rr, mc, margin, per_query = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        'retrieval': plot_retrieval(run_summary),
        'alignment': plot_alignment(run_summary),
        'margin': plot_margin(per_query),
        'significance': plot_significance(rr, mc, margin),
        'rr_margin_dist': plot_rr_margin_distributions(per_query),
        'overlay_counts': plot_overlay_histograms(per_query),
    }
    md = build_markdown(figures)
    print(md)
    for pair in figures.values():
        print(pair[0])
        print(pair[1])


if __name__ == '__main__':
    main()
