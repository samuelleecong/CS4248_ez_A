# Experiment TODO

## Ablation Study

Separate BiMGA's two contributions to show each matters independently. Run at best settings (aw=10, dw=50 and aw=10, dw=100).

| # | Ablation | Query align | Doc align | Margin weight | Implementation |
|:-:|----------|:-----------:|:---------:|:-------------:|----------------|
| A1 | embed_distill (baseline) | uniform | -- | -- | existing method |
| A2 | bidirectional uniform | uniform | uniform | -- | new: `bimga_uniform` — BiMGA loss without sigmoid weighting (w_i = 1.0) |
| A3 | margin query-only | margin | -- | -- | new: `bimga_query_only` — margin-weighted query align, no doc align |
| A4 | **BiMGA (full)** | margin | margin | margin | existing `bimga` |

**Key questions:**
- A2 vs A1: Does document alignment help independently of margin weighting?
- A3 vs A1: Does margin weighting help independently of document alignment?
- A4 vs A2: Does margin weighting help on top of bidirectional alignment?
- If A2 > A4: margin weighting is hurting (explains the aw=10, dw=100 result)

## Multiple Seeds

Run best configs with seeds 42, 123, 456, 789, 1337 to get confidence intervals. Priority configs:
- BiMGA at aw=10, dw=50
- embed_distill at aw=10, dw=100
- control_supervised

## Different Student-Teacher Pairs

Test generalization beyond TinyBERT-4L → MiniLM-L6:
- MiniLM-L6-v2 → all-mpnet-base-v2 (your teammate's TACO setup from the March 26 HF upload)
- TinyBERT-4L → all-mpnet-base-v2 (larger teacher gap)

## Investigate High-Weight Failure

At aw=10, dw=100, embed_distill beats BiMGA by 0.011. Run A2 (bidirectional uniform) at this setting:
- If A2 > BiMGA: margin weighting suppresses useful signal at high weights → consider annealing
- If A2 < embed_distill: doc alignment itself hurts at high weights (unlikely given other results)
