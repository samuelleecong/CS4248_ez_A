# Figure 2: Document cosine vs symmetric MRR

**Purpose:** Test whether better student document embeddings explain better symmetric retrieval.

![Figure 2: Document cosine vs symmetric MRR](../figures/fig02_doc_cosine_vs_sym_mrr.png)

## How to read this
Each point is one replayed HF run. The x-axis is how close student doc embeddings are to teacher docs. The y-axis is symmetric test MRR.

## What it shows
Across 29 replayed runs, higher document cosine tracks higher symmetric MRR (Pearson r=0.814).

## Why it matters for the paper
This is the core quantitative evidence for Analysis #1: document quality is not just a side metric; it tracks retrieval quality.
