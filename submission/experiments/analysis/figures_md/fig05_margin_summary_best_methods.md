# Figure 5: Margin summary for representative methods

**Purpose:** Summarize score separation with both average margin strength and failure rate.

![Figure 5: Margin summary for representative methods](../figures/fig05_margin_summary_best_methods.png)

## How to read this
The left panel shows mean margin with a bootstrap confidence interval. The right panel shows how often the positive margin is zero or negative.

## What it shows
The least negative average margin belongs to `s1_hnp_dw100_pw10` (`-0.1246`), while the worst negative-margin rate is `s1_control_bs32` (`0.867`). BiMGA does not win on mean margin, but it does reduce the negative-margin rate relative to the other KD baselines.

## Why it matters for the paper
This turns per-query margin behavior into a compact paper-ready comparison and shows that score separation and symmetric retrieval quality are related but not identical signals.
