# CS4248 Assignment Checklist

This checklist is tailored to the current MBPP text-to-code retrieval project and the final report requirements.

## 1. Project Framing

- [ ] Lock the final project scope in one sentence.
- [ ] Decide the main story: best retrieval quality, or quality-efficiency tradeoff.
- [ ] Write a clear problem statement with input, output, and task definition.
- [ ] State why this is an NLP problem, not just generic machine learning.
- [ ] Write 2-3 concrete project contributions.

## 2. Dataset and Task Setup

- [ ] Describe MBPP clearly.
- [ ] State which splits are used for train, validation, and test.
- [ ] Explain the retrieval setup: query is natural language, target is code.
- [ ] Explain any preprocessing or filtering decisions.
- [ ] Mention any limitations in the dataset or task formulation.
- [ ] Decide whether to include functional evaluation using MBPP tests.

## 3. Models and Baselines

- [ ] Include a random baseline.
- [ ] Include a TF-IDF lexical baseline.
- [ ] Include the pretrained dense embedding models tested.
- [ ] Include the standard MNR finetuning stage.
- [ ] Include the hard-negative finetuning stage.
- [ ] Clearly distinguish original project code from external libraries and pretrained models.

## 4. Evaluation Plan

- [ ] Report primary ranking metrics: `MRR`, `Recall@1`, `Recall@5`, `Recall@10`.
- [ ] Report supporting metrics: `MAP@10`, `nDCG@10`.
- [ ] Decide whether `Precision@10` stays in the main report or appendix.
- [ ] Do not rely on `F1` unless the task definition is changed to make it meaningful.
- [ ] Include baseline-to-model comparisons, not just final scores.
- [ ] Include statistical comparison or confidence intervals for the key deltas.
- [ ] Add micro-level error analysis with concrete examples.
- [ ] If using the efficiency framing, measure latency, throughput, and model size.
- [ ] If possible, add a functional retrieval metric using MBPP test cases.

## 5. Experiments

- [ ] Document the full experiment matrix that was run.
- [ ] State the selected best pretrained backbone and why it was selected.
- [ ] State the sweep configurations that were tried.
- [ ] Explain why standard MNR was used.
- [ ] Explain why hard-negative mining was added.
- [ ] Record any experiments that did not work and summarize what was learned.
- [ ] Decide which failed or less useful branches go into the appendix instead of the main report.

## 6. Analysis and Discussion

- [ ] Write at least 2-3 discussion questions.
- [ ] Include one discussion question that is clearly NLP-specific.
- [ ] Explain why dense retrievers outperform lexical baselines.
- [ ] Explain when finetuning helps and when gains are small.
- [ ] Explain any differences between small and large models.
- [ ] If using the efficiency story, identify the best cost-performance model.
- [ ] Include at least 3-5 concrete failure examples.
- [ ] Include limitations and realistic future improvements.

## 7. Figures and Tables

- [ ] One main table comparing baselines, pretrained models, and finetuned models.
- [ ] One plot showing methodology-level improvement.
- [ ] One plot showing per-model before/after finetuning.
- [ ] If using the efficiency story, one Pareto-style plot for quality vs cost.
- [ ] Make sure every figure has a readable caption and can stand alone.
- [ ] Do not put critical findings only in appendices.

## 8. Report Writing

- [ ] Keep the main report body within 8 A4 pages.
- [ ] Keep the abstract within 100-200 words.
- [ ] Use the recommended section flow:
- [ ] Introduction
- [ ] Related Work / Background
- [ ] Corpus Analysis & Method
- [ ] Experiments
- [ ] Discussion
- [ ] Conclusion
- [ ] Make sure the introduction states motivation, task, and contributions clearly.
- [ ] Make sure the related work section cites relevant prior work and model sources.
- [ ] Make sure the experiments section explains metrics and baselines.
- [ ] Make sure the discussion section interprets results rather than repeating them.
- [ ] Make sure the conclusion includes limitations and future work.

## 9. Reproducibility

- [ ] Ensure the repo is organized and documented.
- [ ] Ensure the main experiment command works from the documented path.
- [ ] Ensure the plot generation command is documented.
- [ ] Ensure final results live in the expected localized folder under `experiments/kai/`.
- [ ] Ensure the README explains how to run the project.
- [ ] Ensure the notebook is not the only way to reproduce results.
- [ ] Ensure generated artifacts are ignored by git.

## 10. References and Policy

- [ ] Add complete academic references for models, datasets, and methods.
- [ ] Cite external repositories, blogs, or documentation where appropriate.
- [ ] Add the Statement of Independent Work section.
- [ ] Add AI usage disclosure.
- [ ] Keep an audit trail of prompts and AI-assisted work if needed.
- [ ] Add an ethical statement if it helps explain data, compute, or deployment concerns.

## 11. Final Submission Check

- [ ] Regenerate final metrics and plots if anything changed.
- [ ] Recheck the best numbers quoted in the report against the CSV outputs.
- [ ] Recheck captions, section numbering, and formatting.
- [ ] Recheck page count.
- [ ] Export the final PDF.
- [ ] Verify the PDF is legible and correctly formatted.
- [ ] Verify the repository branch contains the intended files only.
- [ ] Submit the PDF to Canvas before the deadline.

## 12. Nice-to-Have Upgrades

- [ ] Add functional retrieval evaluation using MBPP tests.
- [ ] Add latency benchmarking for all models.
- [ ] Add compact-vs-large model tradeoff analysis.
- [ ] Add ablation analysis for hard-negative training.
- [ ] Add a small human-judged relevance study for sampled queries.
