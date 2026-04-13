# Poster Captions

## Figure A (`poster_fig_A_cka`)

- **CKA** measures whether two layers organise examples the same way — high CKA means the same examples cluster together in both.
- **(a)-(b):** CKA between every teacher layer (rows) and student layer (columns). Brighter = more similar.
- **(c):** CKA gain over Control per student layer.
- **Finding:** BiMGA's gain peaks at middle layers L2-L3 (+0.076, +0.134), not the output (+0.013). Score Distill and HNP show the opposite. BiMGA reshapes internal representations; output-matching methods only improve the final embedding. Bootstrap: P(BiMGA > Control) = 100%, CI [0.064, 0.087].

## Figure B (`poster_fig_B_teacher_kl`)

- **KL divergence** measures how differently a student attends to input tokens compared to the teacher — lower means more teacher-like. Each student layer is matched to its most similar teacher layer.
- **(a):** Overall KL from teacher. BiMGA (Full) closest (1.626), Control most divergent (2.071). All CIs non-overlapping (bootstrap n=1,000).
- **(b):** Per-layer KL. Alignment methods (BiMGA, Embed) decrease sharply with depth — deeper layers become progressively more teacher-like. Output-matching methods (Score, HNP, Control) plateau.
