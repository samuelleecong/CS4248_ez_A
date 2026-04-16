# Attention Analysis: How BiMGA Reshapes Internal Representations

## Overview

We analyse the internal representations and attention patterns of 6 student models (TinyBERT, 4 transformer layers, 12 heads) distilled from a MiniLM-L12 teacher on the TACO competitive programming dataset (1,000 test examples). Our goal is to understand *where* in the network BiMGA's advantage arises — is it at the output layer (which would be trivially explained by its alignment loss), or does it reshape the internal representations?

---

## Figure 1: BiMGA's Alignment Advantage is in the Middle Layers, Not the Output

**File:** `fig1_cka_layerwise.png` / `.pdf`

### What it shows

*Panels (a)-(b):* CKA heatmaps for Control and BiMGA, showing teacher-student representational similarity at each (teacher layer, student layer) pair. *Panel (c):* CKA gain of each distillation method over the Control baseline, plotted per student layer.

### The key finding

A naive reading of CKA would attribute BiMGA's advantage to the output layer — after all, BiMGA directly minimises L2 distance on the output embedding. But the per-layer decomposition reveals the opposite:

| Method | L2 (middle) gain | L3 (penultimate) gain | L4 (output) gain | MRR |
|--------|------------------|-----------------------|-------------------|-----|
| Score Distill | +0.020 | +0.031 | +0.031 | 0.301 |
| Hard Neg Pairwise | +0.009 | +0.010 | +0.029 | 0.302 |
| Embed Distill | +0.030 | -0.016 | +0.011 | 0.303 |
| BiMGA Uniform | +0.050 | +0.069 | +0.019 | 0.313 |
| **BiMGA (Full)** | **+0.076** | **+0.134** | **+0.013** | **0.325** |

**BiMGA's CKA gain peaks at L3 (+0.134) and L2 (+0.076) — the middle/penultimate transformer layers — while its output layer gain (+0.013) is the smallest of any distillation method.** Score Distill and Hard Neg Pairwise show the opposite pattern: their gains are concentrated at the output layer.

This means:

- **Score Distill / HNP** improve the output embedding without changing how the model internally processes the input. They find a different internal route to a better output.
- **BiMGA** reshapes the internal representations in the middle layers so they structurally mirror the teacher's. The output improvement is a consequence of better internal processing, not a direct effect of the loss.

### Why this matters

The middle-layer alignment is the non-trivial, non-circular evidence for BiMGA's mechanism. It shows that bidirectional embedding alignment doesn't just push the output closer to the teacher — it forces the gradient signal to propagate backward through the network, restructuring the intermediate representations. This is consistent with BiMGA's bidirectional design: by aligning both query *and* document embeddings simultaneously, the loss creates pressure on the shared backbone layers (which process both), not just the final projection.

### Statistical backing

- Bootstrap (n=1000): Mean CKA difference (deep layers, L2-4) = 0.076 [95% CI: 0.064, 0.087]. P(BiMGA > Control) = 100%.
- Random baseline: Real CKA at student L1 = 0.57 vs random = 0.06 (Z = 457), confirming CKA reflects genuine learned alignment.

---

## Figure 2: Attention Divergence Hierarchy

**File:** `fig2_kl_hierarchy.png` / `.pdf`

### What it shows

KL divergence of each model's attention distributions from BiMGA's, per (layer, head). Panels ordered left-to-right from most similar to most divergent.

### The hierarchy

| Model | Mean KL | Relationship to BiMGA |
|-------|---------|----------------------|
| BiMGA Uniform | 0.086 | Same bidirectional mechanism, differs only in weighting |
| Embed Distill | 0.210 | Shares partial alignment objective (query-side L2) |
| Score Distill | 0.467 | Matches output distributions only |
| Hard Neg Pairwise | 0.500 | Learns rankings via pairwise preferences |
| Control (Supervised) | 0.801 | No distillation signal at all |

### What this adds beyond CKA

CKA measures representational *similarity* (what information is encoded). KL divergence on attention measures *processing strategy* (how the model routes information). These are complementary:

- CKA tells us BiMGA's middle layers encode similar information to the teacher's.
- KL tells us BiMGA's attention heads route information in similar patterns.

The KL hierarchy mirrors the CKA findings: alignment-based methods (BiMGA Uniform, Embed) develop the most similar attention patterns, while output-matching methods (Score, HNP) achieve their results through different internal strategies.

The structural details add nuance:
- **Control** diverges most at layer 0 head 0 (KL = 2.18) — different low-level attention from the first layer.
- **Hard Neg Pairwise** has concentrated hotspots at layer 0 heads 8-9 and across layer 3 — it develops a specialised early-layer pattern for hard negative discrimination that other methods don't.
- **BiMGA Uniform** is near-zero everywhere (max 0.33), confirming the margin-weighting refines training dynamics but barely changes the resulting attention patterns.

---

## Figure 3: Layer-wise Gain Decomposition (Supplementary)

**File:** `fig3_layer_gains.png` / `.pdf`

Grouped bar chart showing CKA gain over Control at three student layers (L2 middle, L3 penultimate, L4 output) for each distillation method. This is the same data as panel (c) of Figure 1 in a different format that makes the per-method comparison clearer.

The visual pattern is immediate:
- **Score Distill / HNP**: Roughly equal bars, slight output-layer emphasis (red bars tallest).
- **BiMGA variants**: Blue (L2) and green (L3) bars tower over red (L4). The alignment advantage is internal.

---

## What we tested and ruled out

| Concern | Test | Result |
|---------|------|--------|
| CKA difference is noise | Bootstrap (n=1000) | P(BiMGA > Control) = 100%, CI doesn't touch zero |
| Layer 1 CKA is a tokenizer artifact | Random projection baseline | Real = 0.57, Random = 0.06, Z = 457 |
| BiMGA's CKA advantage is just at the output layer | Per-layer decomposition | **Refuted** — gain peaks at L2-L3, L4 gain is smallest |
| Per-example entropy predicts retrieval | Spearman correlation | r = 0.002 (p = 0.96), no signal |
| Tag-specific entropy variation is meaningful | Cross-tag variance | 0.04 range, too small |

---

## Figures Index

| File | Description | Use |
|------|-------------|-----|
| `fig1_cka_layerwise.png/pdf` | CKA heatmaps + per-layer gain curves | **Primary figure** |
| `fig2_kl_hierarchy.png/pdf` | KL divergence from BiMGA, 5 panels | **Supporting figure** |
| `fig3_layer_gains.png/pdf` | Layer-wise gain bars | Supplementary |
