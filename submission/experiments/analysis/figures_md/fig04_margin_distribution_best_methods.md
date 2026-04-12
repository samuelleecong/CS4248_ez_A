# Figure 4: Margin distributions for representative methods

**Purpose:** Compare how clearly each representative method separates the positive code from the hardest negative.

![Figure 4: Margin distributions for representative methods](../figures/fig04_margin_distribution_best_methods.png)

## How to read this
Wider distributions shifted upward are better. In this full-corpus setting the hardest-negative margin is often still negative, so the useful comparison is whether one method is less negative than another and whether it produces fewer extreme failures.

## What it shows
Among the representative runs, `s1_hnp_dw100_pw10` has the least negative median margin at `-0.1223`, but the top-MRR BiMGA variants still outperform it overall on retrieval quality.

## Why it matters for the paper
This is the main quantitative view for Analysis #2: score separation helps explain model behavior, but it does not replace the document-quality story from symmetric retrieval.
