"""Find test queries where one model gets R@1 (symmetric) and the other does not.

Either load from **local** sweep dirs (with ``model/``) or from **Hugging Face** repos listed in
``submission/experiments/README.md``.

  cd mbpp_kd_suite
  uv run python ../submission/person_c/cherry_pick_examples.py \\
    --hf-repo-a cs4248-nlp/paper-s1-bimga-dw50-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143 \\
    --hf-repo-b cs4248-nlp/paper-s1-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143 \\
    --dataset-name BEE-spoke-data/TACO-hf --per-side 5

Writes JSON (+ Markdown) under ``submission/person_c/outputs/``.

**macOS / CPU:** if ``uv run`` fails on CUDA-only torch wheels, sync deps from PyPI::

  cd mbpp_kd_suite && uv sync --no-sources

Then run with ``.venv/bin/python`` (see command below).

**Hugging Face (no local ``model/`` dirs):**

  cd mbpp_kd_suite
  .venv/bin/python ../submission/person_c/cherry_pick_examples.py \\
    --hf-repo-a cs4248-nlp/paper-s1-bimga-dw50-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143 \\
    --hf-repo-b cs4248-nlp/paper-s1-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143 \\
    --dataset-name BEE-spoke-data/TACO-hf --per-side 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PERSON_C = Path(__file__).resolve().parent
if str(_PERSON_C) not in sys.path:
    sys.path.insert(0, str(_PERSON_C))

from mbpp_kd_suite.config import TrainConfig
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.metrics import paired_ranks, score_metrics_from_embeddings
from mbpp_kd_suite.modeling import encode_student_texts
from mbpp_kd_suite.runtime import apply_device_runtime_optimizations, pick_device, set_seed

from suite_student_loader import load_suite_student, load_suite_student_from_hub, resolve_run_paths


def _truncate(s: str, max_chars: int) -> str:
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _symmetric_eval_mrr(q: torch.Tensor, d: torch.Tensor) -> float:
    return score_metrics_from_embeddings(q, d)["MRR"]


def _pick_examples(
    scores: np.ndarray,
    ranks: np.ndarray,
    *,
    win_mask: np.ndarray,
    k: int,
    queries: list[str],
    codes: list[str],
    label: str,
    max_query_chars: int,
    max_code_chars: int,
) -> list[dict]:
    """Rows where win_mask is True (e.g. rank==1 for this model)."""
    slug = label.replace(" ", "_")
    idxs = np.flatnonzero(win_mask)
    out: list[dict] = []
    for i in idxs[:k]:
        i = int(i)
        row = scores[i]
        pred_j = int(np.argmax(row))
        rank = int(ranks[i])
        out.append(
            {
                "test_index": i,
                "query": _truncate(queries[i], max_query_chars),
                "gold_code": _truncate(codes[i], max_code_chars),
                f"{slug}_symmetric_rank_of_gold": rank,
                f"{slug}_top1_predicted_code": _truncate(codes[pred_j], max_code_chars),
                f"{slug}_top1_is_gold": pred_j == i,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cherry-pick symmetric R@1 disagreements between two students.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional phase-1 checkpoint.pt (reads dataset_name from adjacent config.json)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset id (default: from checkpoint config, else BEE-spoke-data/TACO-hf)",
    )
    parser.add_argument(
        "--hf-repo-a",
        default=None,
        help="Hugging Face repo id for model A (e.g. cs4248-nlp/paper-s1-bimga-...)",
    )
    parser.add_argument(
        "--hf-repo-b",
        default=None,
        help="Hugging Face repo id for model B (e.g. cs4248-nlp/paper-s1-embed-...)",
    )
    parser.add_argument("--run-a", type=Path, default=None, help="Local sweep run directory for model A")
    parser.add_argument("--run-b", type=Path, default=None, help="Local sweep run directory for model B")
    parser.add_argument("--label-a", default="BiMGA", help="Short name for run-a in the output")
    parser.add_argument("--label-b", default="embed_distill", help="Short name for run-b in the output")
    parser.add_argument("--per-side", type=int, default=5, help="Max examples per direction (a wins vs b wins)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-query-chars", type=int, default=320)
    parser.add_argument("--max-code-chars", type=int, default=500)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_PERSON_C / "outputs" / "cherry_pick_examples.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=_PERSON_C / "outputs" / "cherry_pick_examples.md",
    )
    args = parser.parse_args()

    use_hub = bool(args.hf_repo_a and args.hf_repo_b)
    use_local = bool(args.run_a and args.run_b)
    if use_hub == use_local:
        parser.error("Specify exactly one of: (--hf-repo-a and --hf-repo-b) or (--run-a and --run-b).")

    set_seed(args.seed)
    device = pick_device()

    original_cfg: dict = {}
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint.expanduser().resolve()
        run_dir_original = ckpt_path.parent.parent
        config_path = run_dir_original / "config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                original_cfg = json.load(f)

    dataset_name = args.dataset_name or original_cfg.get("dataset_name", "BEE-spoke-data/TACO-hf")
    student_model_id = original_cfg.get("student_model", "huawei-noah/TinyBERT_General_4L_312D")
    teacher_model = original_cfg.get("teacher_model", "sentence-transformers/all-MiniLM-L6-v2")

    cfg = TrainConfig(
        teacher_model=teacher_model,
        student_model=student_model_id,
        dataset_name=dataset_name,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    apply_device_runtime_optimizations(cfg=cfg, device=device)

    dataset = load_retrieval_dataset(dataset_name=dataset_name, taco_val_size=1000, seed=args.seed)
    data = dataset_dict_to_splits(dataset)
    test = data.test
    queries = test.queries
    codes = test.codes
    n = len(queries)
    if n != len(codes):
        raise ValueError("Test split length mismatch")

    if use_hub:
        assert args.hf_repo_a and args.hf_repo_b
        student_a, tok_a = load_suite_student_from_hub(args.hf_repo_a, device=device)
        student_b, tok_b = load_suite_student_from_hub(args.hf_repo_b, device=device)
    else:
        assert args.run_a and args.run_b
        _, model_parent_a = resolve_run_paths(args.run_a.expanduser().resolve())
        _, model_parent_b = resolve_run_paths(args.run_b.expanduser().resolve())
        student_a, tok_a = load_suite_student(model_parent_a, device=device)
        student_b, tok_b = load_suite_student(model_parent_b, device=device)

    qa = encode_student_texts(
        student_a,
        tok_a,
        queries,
        "query",
        cfg.max_query_length,
        cfg.eval_batch_size,
        device,
        "cherry_q_a",
    )
    da = encode_student_texts(
        student_a,
        tok_a,
        codes,
        "document",
        cfg.max_code_length,
        cfg.eval_batch_size,
        device,
        "cherry_d_a",
    )
    qb = encode_student_texts(
        student_b,
        tok_b,
        queries,
        "query",
        cfg.max_query_length,
        cfg.eval_batch_size,
        device,
        "cherry_q_b",
    )
    db = encode_student_texts(
        student_b,
        tok_b,
        codes,
        "document",
        cfg.max_code_length,
        cfg.eval_batch_size,
        device,
        "cherry_d_b",
    )

    scores_a = (qa @ da.T).numpy()
    scores_b = (qb @ db.T).numpy()
    ranks_a = paired_ranks(scores_a)
    ranks_b = paired_ranks(scores_b)

    mrr_a = _symmetric_eval_mrr(qa, da)
    mrr_b = _symmetric_eval_mrr(qb, db)

    # a wins @1, b does not
    a_only = (ranks_a == 1) & (ranks_b > 1)
    # b wins @1, a does not
    b_only = (ranks_b == 1) & (ranks_a > 1)

    n_a = int(np.sum(a_only))
    n_b = int(np.sum(b_only))

    ex_a = _pick_examples(
        scores_a,
        ranks_a,
        win_mask=a_only,
        k=args.per_side,
        queries=queries,
        codes=codes,
        label=args.label_a,
        max_query_chars=args.max_query_chars,
        max_code_chars=args.max_code_chars,
    )
    ex_b = _pick_examples(
        scores_b,
        ranks_b,
        win_mask=b_only,
        k=args.per_side,
        queries=queries,
        codes=codes,
        label=args.label_b,
        max_query_chars=args.max_query_chars,
        max_code_chars=args.max_code_chars,
    )

    slug_a = args.label_a.replace(" ", "_")
    slug_b = args.label_b.replace(" ", "_")

    # Merge rows: primary model fields from _pick_examples + other model ranks/top1
    merged_a: list[dict] = []
    for row in ex_a:
        i = row["test_index"]
        merged_a.append(
            {
                **row,
                f"{slug_b}_symmetric_rank_of_gold": int(ranks_b[i]),
                f"{slug_b}_top1_predicted_code": _truncate(
                    codes[int(np.argmax(scores_b[i]))], args.max_code_chars
                ),
            }
        )

    merged_b: list[dict] = []
    for row in ex_b:
        i = row["test_index"]
        merged_b.append(
            {
                **row,
                f"{slug_a}_symmetric_rank_of_gold": int(ranks_a[i]),
                f"{slug_a}_top1_predicted_code": _truncate(
                    codes[int(np.argmax(scores_a[i]))], args.max_code_chars
                ),
            }
        )

    payload = {
        "dataset": dataset_name,
        "test_size": n,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "symmetric_mrr": {args.label_a: mrr_a, args.label_b: mrr_b},
        "disagreement_counts": {
            f"only_{slug_a}_recall_at_1": n_a,
            f"only_{slug_b}_recall_at_1": n_b,
        },
        f"examples_only_{slug_a}_r1": merged_a,
        f"examples_only_{slug_b}_r1": merged_b,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown appendix-friendly table
    lines: list[str] = []
    lines.append("# Cherry-picked symmetric retrieval examples\n")
    lines.append(f"- Dataset: `{dataset_name}` (test n={n})\n")
    lines.append(
        f"- Symmetric MRR: **{args.label_a}** {mrr_a:.4f}, **{args.label_b}** {mrr_b:.4f}\n"
    )
    lines.append(f"- Queries where only {args.label_a} has R@1: **{n_a}**; only {args.label_b}: **{n_b}**\n")

    def _section(title: str, rows: list[dict], primary: str, other: str) -> None:
        lines.append(f"\n## {title}\n")
        ps = primary.replace(" ", "_")
        os = other.replace(" ", "_")
        for j, r in enumerate(rows, 1):
            lines.append(f"\n### Example {j} (test index {r['test_index']})\n")
            lines.append("\n**Query**\n\n```\n" + r["query"] + "\n```\n")
            lines.append("\n**Gold code (truncated)**\n\n```\n" + r["gold_code"] + "\n```\n")
            lines.append(
                f"\n- **{primary}**: gold rank = {r.get(f'{ps}_symmetric_rank_of_gold')}, "
                f"top-1 matches gold = {r.get(f'{ps}_top1_is_gold')}\n"
            )
            lines.append(
                f"\n**{primary} top-1 prediction (truncated)**\n\n```\n"
                + str(r.get(f"{ps}_top1_predicted_code", ""))
                + "\n```\n"
            )
            lines.append(f"\n- **{other}**: gold rank = {r.get(f'{os}_symmetric_rank_of_gold')}\n")
            lines.append(
                f"\n**{other} top-1 prediction (truncated)**\n\n```\n"
                + str(r.get(f"{os}_top1_predicted_code", ""))
                + "\n```\n"
            )

    _section(
        f"Only {args.label_a} at R@1 (first {len(merged_a)})",
        merged_a,
        primary=args.label_a,
        other=args.label_b,
    )
    _section(
        f"Only {args.label_b} at R@1 (first {len(merged_b)})",
        merged_b,
        primary=args.label_b,
        other=args.label_a,
    )

    with args.output_md.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Symmetric MRR: {args.label_a}={mrr_a:.4f} {args.label_b}={mrr_b:.4f}")
    print(f"R@1-only counts: {args.label_a}={n_a} {args.label_b}={n_b}")


if __name__ == "__main__":
    main()
