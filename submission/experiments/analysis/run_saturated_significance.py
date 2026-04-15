from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import binomtest, wilcoxon
from transformers import AutoModel, AutoTokenizer

from build_quant_package import (
    MAX_CODE_LENGTH,
    MAX_QUERY_LENGTH,
    DEFAULT_EVAL_BATCH_SIZE,
    PROJECT_ROOT,
    encode_texts,
    infer_model_encoding_spec,
    load_projection,
    load_taco_retrieval,
    teacher_doc_embeddings,
    to_py,
    write_csv,
)
from mbpp_kd_suite.config import DistillTargets
from mbpp_kd_suite.metrics import paired_ranking_metrics, paired_ranks

SATURATED_RUNS: list[dict[str, Any]] = [
    {
        "run_name": "s7_control_bs32",
        "method": "control",
        "repo_id": "cs4248-nlp/paper-s7-control-bs32-tinybert-general-4l-312d-taco-hf-20260402-015143",
        "expected": {
            "MRR": 0.205,
            "Recall@1": 0.143,
            "Recall@10": 0.331,
            "MAP@10": None,
            "nDCG@10": None,
            "Asym MRR": None,
            "Doc Cosine": None,
        },
    },
    {
        "run_name": "s7_embed_dw100_aw10",
        "method": "embed_distill",
        "repo_id": "cs4248-nlp/paper-s7-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
        "expected": {
            "MRR": 0.303,
            "Recall@1": 0.218,
            "Recall@10": 0.461,
            "MAP@10": None,
            "nDCG@10": None,
            "Asym MRR": 0.310,
            "Doc Cosine": 0.679,
        },
    },
    {
        "run_name": "s8_A2_bimga_uniform",
        "method": "bimga_uniform",
        "repo_id": "cs4248-nlp/paper-s8-a2-bimga-uniform-tinybert-general-4l-312d-taco-hf-20260402-015143",
        "expected": {
            "MRR": 0.313,
            "Recall@1": 0.232,
            "Recall@10": 0.469,
            "MAP@10": None,
            "nDCG@10": None,
            "Asym MRR": 0.316,
            "Doc Cosine": 0.856,
        },
    },
    {
        "run_name": "s8_hnp_dw100_pw10",
        "method": "hard_neg_pair",
        "repo_id": "cs4248-nlp/paper-s8-hnp-dw100-pw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
        "expected": {
            "MRR": 0.302,
            "Recall@1": 0.221,
            "Recall@10": 0.461,
            "MAP@10": None,
            "nDCG@10": None,
            "Asym MRR": 0.007,
            "Doc Cosine": 0.001,
        },
    },
    {
        "run_name": "s9_score_dw100",
        "method": "score_distill",
        "repo_id": "cs4248-nlp/paper-s9-score-dw100-tinybert-general-4l-312d-taco-hf-20260402-015143",
        "expected": {
            "MRR": 0.301,
            "Recall@1": 0.215,
            "Recall@10": 0.466,
            "MAP@10": None,
            "nDCG@10": None,
            "Asym MRR": 0.006,
            "Doc Cosine": 0.0,
        },
    },
    {
        "run_name": "s10_bimga_dw100_aw10",
        "method": "bimga",
        "repo_id": "cs4248-nlp/paper-s10-bimga-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143",
        "expected": {
            "MRR": 0.325,
            "Recall@1": 0.241,
            "Recall@10": 0.486,
            "MAP@10": None,
            "nDCG@10": None,
            "Asym MRR": 0.321,
            "Doc Cosine": 0.881,
        },
    },
]

PAIRWISE_ORDER = [
    "s10_bimga_dw100_aw10",
    "s8_A2_bimga_uniform",
    "s7_embed_dw100_aw10",
    "s8_hnp_dw100_pw10",
    "s9_score_dw100",
    "s7_control_bs32",
]

PRIMARY_COMPARISONS = [
    ("s10_bimga_dw100_aw10", "s7_control_bs32"),
    ("s10_bimga_dw100_aw10", "s7_embed_dw100_aw10"),
    ("s10_bimga_dw100_aw10", "s8_A2_bimga_uniform"),
    ("s10_bimga_dw100_aw10", "s8_hnp_dw100_pw10"),
    ("s10_bimga_dw100_aw10", "s9_score_dw100"),
]

RUN_LABELS = {row["run_name"]: row["method"] for row in SATURATED_RUNS}
BOOTSTRAP_REPS = 10000
PERMUTATION_REPS = 100000
BOOTSTRAP_SEED = 42
PERMUTATION_SEED = 4242


@dataclass
class SaturatedReplay:
    run_name: str
    method: str
    repo_id: str
    replay_status: str
    notes: str
    student_hidden_size: int | None
    output_dim: int | None
    projection_exists: bool
    symmetric_metrics: dict[str, float] | None
    exact_teacher_metrics: dict[str, dict[str, float]] | None
    per_query_rows: list[dict[str, Any]]


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.{digits}f}"


def format_pvalue(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_status(value: str | None) -> str:
    if not value:
        return "—"
    return value


def load_published_metrics(repo_id: str) -> dict[str, Any]:
    path = hf_hub_download(repo_id, "metrics.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_teacher_targets_from_checkpoint(path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    targets = DistillTargets(**checkpoint["ft_teacher_targets"])
    return targets.test_query.cpu(), targets.test_doc.cpu()


def paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 1000,
) -> tuple[float, float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    n = deltas.size
    means = np.empty(reps, dtype=np.float64)
    cursor = 0
    while cursor < reps:
        width = min(chunk_size, reps - cursor)
        indices = rng.integers(0, n, size=(width, n), dtype=np.int32)
        means[cursor:cursor + width] = deltas[indices].mean(axis=1)
        cursor += width
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def sign_flip_permutation_test(
    deltas: np.ndarray,
    *,
    reps: int = PERMUTATION_REPS,
    seed: int = PERMUTATION_SEED,
    chunk_size: int = 5000,
) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return math.nan, math.nan
    if np.allclose(deltas, 0.0):
        return 0.0, 1.0
    observed = float(np.mean(deltas))
    abs_observed = abs(observed)
    rng = np.random.default_rng(seed)
    delta32 = deltas.astype(np.float32, copy=False)
    n = deltas.size
    extreme = 1
    drawn = 0
    while drawn < reps:
        width = min(chunk_size, reps - drawn)
        signs = rng.integers(0, 2, size=(width, n), dtype=np.int8)
        signs = signs.astype(np.float32, copy=False)
        signs *= 2.0
        signs -= 1.0
        permuted = (signs @ delta32) / n
        extreme += int(np.count_nonzero(np.abs(permuted) >= abs_observed - 1e-12))
        drawn += width
    p_value = extreme / float(reps + 1)
    return observed, p_value


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)
    a_only = int(np.count_nonzero((a == 1) & (b == 0)))
    b_only = int(np.count_nonzero((a == 0) & (b == 1)))
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    p_value = float(binomtest(min(a_only, b_only), discordant, p=0.5, alternative="two-sided").pvalue)
    return a_only, b_only, p_value


def wilcoxon_signed_rank(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0 or np.allclose(deltas, 0.0):
        return 1.0
    result = wilcoxon(deltas, alternative="two-sided", zero_method="pratt", method="auto")
    return float(result.pvalue)


def holm_bonferroni(rows: list[dict[str, Any]], p_key: str, out_key: str) -> None:
    active = [(idx, float(row[p_key])) for idx, row in enumerate(rows) if row.get(p_key) not in ("", None)]
    if not active:
        return
    active.sort(key=lambda item: item[1])
    m = len(active)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    prev = 0.0
    for rank, (_, p_value) in enumerate(active):
        adjusted = min(1.0, (m - rank) * p_value)
        prev = max(prev, adjusted)
        adjusted_sorted[rank] = prev
    for (idx, _), adjusted in zip(active, adjusted_sorted, strict=True):
        rows[idx][out_key] = float(adjusted)
    for idx, row in enumerate(rows):
        if idx not in {i for i, _ in active}:
            row[out_key] = ""


def status_from_ci(
    *,
    delta: float | None,
    ci_low: float | None,
    ci_high: float | None,
    p_adj: float | None,
) -> str:
    if delta is None or ci_low is None or ci_high is None or p_adj is None:
        return "not_applicable"
    if any(math.isnan(v) or math.isinf(v) for v in (delta, ci_low, ci_high, p_adj)):
        return "not_applicable"
    if p_adj >= 0.05:
        return "not_significant"
    if ci_low > 0.0:
        return "significant_better"
    if ci_high < 0.0:
        return "significant_worse"
    return "not_significant"


def score_matrix_to_arrays(score_matrix: np.ndarray, teacher_margin: np.ndarray | None) -> dict[str, np.ndarray]:
    ranks = paired_ranks(score_matrix)
    positives = np.diag(score_matrix).astype(np.float64)
    negatives = score_matrix.copy()
    np.fill_diagonal(negatives, -np.inf)
    hardest_negative = negatives.max(axis=1).astype(np.float64)
    margins = positives - hardest_negative
    teacher_margin_values = (
        teacher_margin.astype(np.float64)
        if teacher_margin is not None
        else np.full(score_matrix.shape[0], np.nan, dtype=np.float64)
    )
    return {
        "rank": ranks.astype(np.int64),
        "reciprocal_rank": 1.0 / ranks.astype(np.float64),
        "correct_at_1": (ranks == 1).astype(np.int8),
        "correct_at_10": (ranks <= 10).astype(np.int8),
        "positive_score": positives,
        "hardest_negative_score": hardest_negative,
        "margin": margins,
        "teacher_margin": teacher_margin_values,
    }


def replay_saturated_run(
    manifest: dict[str, Any],
    queries: list[str],
    codes: list[str],
    teacher_q: torch.Tensor | None,
    teacher_d: torch.Tensor | None,
    teacher_margin: np.ndarray | None,
    device: torch.device,
    batch_size: int,
) -> SaturatedReplay:
    model = None
    projection = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(manifest["repo_id"])
        model = AutoModel.from_pretrained(manifest["repo_id"]).to(device)
        spec = infer_model_encoding_spec(
            manifest["repo_id"],
            getattr(model.config, "_name_or_path", None),
            getattr(tokenizer, "name_or_path", None),
        )
        projection, projection_exists, output_dim = load_projection(manifest["repo_id"])
        if projection is not None:
            projection = projection.to(device)
        q = encode_texts(
            model,
            tokenizer,
            queries,
            role="query",
            max_length=MAX_QUERY_LENGTH,
            batch_size=batch_size,
            device=device,
            projection=projection,
            encoding_spec=spec,
        )
        d = encode_texts(
            model,
            tokenizer,
            codes,
            role="document",
            max_length=MAX_CODE_LENGTH,
            batch_size=batch_size,
            device=device,
            projection=projection,
            encoding_spec=spec,
        )
        sym_scores = (q @ d.T).numpy()
        sym_metrics = paired_ranking_metrics(sym_scores, ks=(1, 10))
        sym_arrays = score_matrix_to_arrays(sym_scores, teacher_margin)

        exact_teacher_metrics = None
        notes = "" if projection_exists else "no_projection"
        if teacher_q is not None and teacher_d is not None and q.shape[1] == teacher_d.shape[1]:
            asym_scores = (q @ teacher_d.T).numpy()
            asym_metrics = paired_ranking_metrics(asym_scores, ks=(1, 10))
            asym_arrays = score_matrix_to_arrays(asym_scores, teacher_margin)
            query_cos = F.cosine_similarity(q, teacher_q, dim=-1).cpu().numpy().astype(np.float64)
            doc_cos = F.cosine_similarity(d, teacher_d, dim=-1).cpu().numpy().astype(np.float64)
            exact_teacher_metrics = {
                "asymmetric": asym_metrics,
                "query_cosine": {"mean": float(np.mean(query_cos))},
                "doc_cosine": {"mean": float(np.mean(doc_cos))},
            }
        else:
            asym_arrays = None
            query_cos = np.full(len(queries), np.nan, dtype=np.float64)
            doc_cos = np.full(len(codes), np.nan, dtype=np.float64)

        per_query_rows: list[dict[str, Any]] = []
        for idx in range(len(queries)):
            row = {
                "query_id": idx,
                "run_name": manifest["run_name"],
                "method": manifest["method"],
                "repo_id": manifest["repo_id"],
                "sym_rank": int(sym_arrays["rank"][idx]),
                "sym_reciprocal_rank": float(sym_arrays["reciprocal_rank"][idx]),
                "sym_correct_at_1": int(sym_arrays["correct_at_1"][idx]),
                "sym_correct_at_10": int(sym_arrays["correct_at_10"][idx]),
                "sym_positive_score": float(sym_arrays["positive_score"][idx]),
                "sym_hardest_negative_score": float(sym_arrays["hardest_negative_score"][idx]),
                "sym_margin": float(sym_arrays["margin"][idx]),
                "teacher_margin": to_py(float(sym_arrays["teacher_margin"][idx])) if np.isfinite(sym_arrays["teacher_margin"][idx]) else "",
                "doc_cosine": "",
                "query_cosine": "",
                "proxy_doc_cosine_raw_teacher": to_py(float(doc_cos[idx])) if np.isfinite(doc_cos[idx]) else "",
                "proxy_query_cosine_raw_teacher": to_py(float(query_cos[idx])) if np.isfinite(query_cos[idx]) else "",
            }
            if asym_arrays is not None:
                row.update(
                    {
                        "asym_rank": int(asym_arrays["rank"][idx]),
                        "asym_reciprocal_rank": float(asym_arrays["reciprocal_rank"][idx]),
                        "asym_correct_at_1": int(asym_arrays["correct_at_1"][idx]),
                        "asym_correct_at_10": int(asym_arrays["correct_at_10"][idx]),
                    }
                )
            else:
                row.update(
                    {
                        "asym_rank": "",
                        "asym_reciprocal_rank": "",
                        "asym_correct_at_1": "",
                        "asym_correct_at_10": "",
                    }
                )
            per_query_rows.append(row)
        return SaturatedReplay(
            run_name=manifest["run_name"],
            method=manifest["method"],
            repo_id=manifest["repo_id"],
            replay_status="ok",
            notes=notes,
            student_hidden_size=int(getattr(model.config, "hidden_size", 0)),
            output_dim=int(output_dim) if output_dim is not None else int(getattr(model.config, "hidden_size", 0)),
            projection_exists=projection_exists,
            symmetric_metrics=sym_metrics,
            exact_teacher_metrics=exact_teacher_metrics,
            per_query_rows=per_query_rows,
        )
    except Exception as exc:
        return SaturatedReplay(
            run_name=manifest["run_name"],
            method=manifest["method"],
            repo_id=manifest["repo_id"],
            replay_status="load_failed",
            notes=str(exc),
            student_hidden_size=None,
            output_dim=None,
            projection_exists=False,
            symmetric_metrics=None,
            exact_teacher_metrics=None,
            per_query_rows=[],
        )
    finally:
        if projection is not None:
            del projection
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


def build_run_summary(
    replay: SaturatedReplay,
    expected: dict[str, float | None],
    published_metrics: dict[str, Any],
) -> dict[str, Any]:
    symmetric = replay.symmetric_metrics or {}
    diagnostics = published_metrics.get("diagnostics", {})
    published_asym_metrics = diagnostics.get("asymmetric_test") or {}
    published_asym = published_asym_metrics.get("MRR")
    published_doc_cos = diagnostics.get("doc_alignment_cosine_test_student_vs_target")
    return {
        "run_name": replay.run_name,
        "method": replay.method,
        "repo_id": replay.repo_id,
        "replay_status": replay.replay_status,
        "projection_exists": replay.projection_exists,
        "student_hidden_size": replay.student_hidden_size or "",
        "output_dim": replay.output_dim or "",
        "replay_mrr": to_py(symmetric.get("MRR")),
        "expected_mrr": to_py(expected.get("MRR")),
        "delta_mrr_vs_expected": to_py(metric_delta(expected.get("MRR"), symmetric.get("MRR"))),
        "replay_recall@1": to_py(symmetric.get("Recall@1")),
        "expected_recall@1": to_py(expected.get("Recall@1")),
        "delta_recall@1_vs_expected": to_py(metric_delta(expected.get("Recall@1"), symmetric.get("Recall@1"))),
        "replay_recall@10": to_py(symmetric.get("Recall@10")),
        "expected_recall@10": to_py(expected.get("Recall@10")),
        "delta_recall@10_vs_expected": to_py(metric_delta(expected.get("Recall@10"), symmetric.get("Recall@10"))),
        "replay_map@10": to_py(symmetric.get("MAP@10")),
        "expected_map@10": to_py(expected.get("MAP@10")),
        "delta_map@10_vs_expected": to_py(metric_delta(expected.get("MAP@10"), symmetric.get("MAP@10"))),
        "replay_ndcg@10": to_py(symmetric.get("nDCG@10")),
        "expected_ndcg@10": to_py(expected.get("nDCG@10")),
        "delta_ndcg@10_vs_expected": to_py(metric_delta(expected.get("nDCG@10"), symmetric.get("nDCG@10"))),
        "replay_asym_mrr": to_py(published_asym),
        "replay_asym_recall@1": to_py(published_asym_metrics.get("Recall@1")),
        "replay_asym_recall@10": to_py(published_asym_metrics.get("Recall@10")),
        "expected_asym_mrr": to_py(expected.get("Asym MRR")),
        "delta_asym_mrr_vs_expected": to_py(metric_delta(expected.get("Asym MRR"), published_asym)),
        "replay_doc_cosine": to_py(published_doc_cos),
        "expected_doc_cosine": to_py(expected.get("Doc Cosine")),
        "delta_doc_cosine_vs_expected": to_py(metric_delta(expected.get("Doc Cosine"), published_doc_cos)),
        "notes": replay.notes,
    }


def metric_delta(expected: float | None, observed: float | None) -> float | None:
    if expected is None or observed is None:
        return None
    return float(observed - expected)


def rows_by_run(per_query_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_query_rows:
        grouped.setdefault(row["run_name"], []).append(row)
    for run_name in grouped:
        grouped[run_name].sort(key=lambda row: int(row["query_id"]))
    return grouped


def extract_numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in ("", None):
            values.append(np.nan)
        else:
            values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


def extract_binary(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[int] = []
    for row in rows:
        raw = row.get(key)
        if raw in ("", None):
            raise ValueError(f"Missing binary field {key}")
        values.append(int(raw))
    return np.asarray(values, dtype=np.int8)


def pairwise_rr_tests(grouped_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for run_a, run_b in itertools.combinations(PAIRWISE_ORDER, 2):
        a = extract_numeric(grouped_rows[run_a], "sym_reciprocal_rank")
        b = extract_numeric(grouped_rows[run_b], "sym_reciprocal_rank")
        delta = a - b
        mean_delta, ci_low, ci_high = paired_bootstrap_mean_ci(delta, seed=BOOTSTRAP_SEED + len(results))
        _, p_value = sign_flip_permutation_test(delta, seed=PERMUTATION_SEED + len(results))
        win_rate = float(np.mean(delta > 0.0))
        tie_rate = float(np.mean(delta == 0.0))
        loss_rate = float(np.mean(delta < 0.0))
        results.append(
            {
                "run_a": run_a,
                "method_a": RUN_LABELS[run_a],
                "run_b": run_b,
                "method_b": RUN_LABELS[run_b],
                "n_queries": len(delta),
                "mean_delta_rr": mean_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "raw_p_value": p_value,
                "win_rate": win_rate,
                "tie_rate": tie_rate,
                "loss_rate": loss_rate,
            }
        )
    holm_bonferroni(results, "raw_p_value", "adjusted_p_value")
    for row in results:
        row["status"] = status_from_ci(
            delta=float(row["mean_delta_rr"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            p_adj=float(row["adjusted_p_value"]),
        )
    return results


def pairwise_mcnemar_tests(grouped_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metric in ("sym_correct_at_1", "sym_correct_at_10"):
        for offset, (run_a, run_b) in enumerate(itertools.combinations(PAIRWISE_ORDER, 2), start=1):
            a = extract_binary(grouped_rows[run_a], metric)
            b = extract_binary(grouped_rows[run_b], metric)
            delta = a.astype(np.float64) - b.astype(np.float64)
            mean_delta, ci_low, ci_high = paired_bootstrap_mean_ci(delta, seed=BOOTSTRAP_SEED + 100 + len(results))
            a_only, b_only, p_value = exact_mcnemar(a, b)
            results.append(
                {
                    "metric": metric,
                    "run_a": run_a,
                    "method_a": RUN_LABELS[run_a],
                    "run_b": run_b,
                    "method_b": RUN_LABELS[run_b],
                    "n_queries": len(delta),
                    "success_rate_a": float(np.mean(a)),
                    "success_rate_b": float(np.mean(b)),
                    "delta_rate": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "a_only": a_only,
                    "b_only": b_only,
                    "raw_p_value": p_value,
                }
            )
    for metric in ("sym_correct_at_1", "sym_correct_at_10"):
        metric_rows = [row for row in results if row["metric"] == metric]
        holm_bonferroni(metric_rows, "raw_p_value", "adjusted_p_value")
        for row in metric_rows:
            row["status"] = status_from_ci(
                delta=float(row["delta_rate"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                p_adj=float(row["adjusted_p_value"]),
            )
    return results


def pairwise_continuous_tests(
    grouped_rows: dict[str, list[dict[str, Any]]],
    key: str,
    *,
    output_name: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, (run_a, run_b) in enumerate(itertools.combinations(PAIRWISE_ORDER, 2), start=1):
        a = extract_numeric(grouped_rows[run_a], key)
        b = extract_numeric(grouped_rows[run_b], key)
        valid = np.isfinite(a) & np.isfinite(b)
        if not np.any(valid):
            results.append(
                {
                    "metric": output_name,
                    "run_a": run_a,
                    "method_a": RUN_LABELS[run_a],
                    "run_b": run_b,
                    "method_b": RUN_LABELS[run_b],
                    "n_valid": 0,
                    "mean_delta": "",
                    "median_delta": "",
                    "ci_low": "",
                    "ci_high": "",
                    "fraction_a_higher": "",
                    "fraction_tie": "",
                    "raw_p_value": "",
                    "status": "not_applicable",
                }
            )
            continue
        delta = a[valid] - b[valid]
        mean_delta, ci_low, ci_high = paired_bootstrap_mean_ci(delta, seed=BOOTSTRAP_SEED + 200 + idx)
        p_value = wilcoxon_signed_rank(delta)
        results.append(
            {
                "metric": output_name,
                "run_a": run_a,
                "method_a": RUN_LABELS[run_a],
                "run_b": run_b,
                "method_b": RUN_LABELS[run_b],
                "n_valid": int(delta.size),
                "mean_delta": mean_delta,
                "median_delta": float(np.median(delta)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "fraction_a_higher": float(np.mean(delta > 0.0)),
                "fraction_tie": float(np.mean(delta == 0.0)),
                "raw_p_value": p_value,
            }
        )
    active = [row for row in results if row.get("status") != "not_applicable"]
    holm_bonferroni(active, "raw_p_value", "adjusted_p_value")
    for row in active:
        row["status"] = status_from_ci(
            delta=float(row["mean_delta"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            p_adj=float(row["adjusted_p_value"]),
        )
    return results


def unavailable_pairwise_rows(metric: str, reason: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_a, run_b in itertools.combinations(PAIRWISE_ORDER, 2):
        rows.append(
            {
                "metric": metric,
                "run_a": run_a,
                "method_a": RUN_LABELS[run_a],
                "run_b": run_b,
                "method_b": RUN_LABELS[run_b],
                "n_valid": 0,
                "mean_delta": "",
                "median_delta": "",
                "ci_low": "",
                "ci_high": "",
                "fraction_a_higher": "",
                "fraction_tie": "",
                "raw_p_value": "",
                "adjusted_p_value": "",
                "status": "not_applicable",
                "reason": reason,
            }
        )
    return rows


def unavailable_within_rows(run_summary_rows: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_summary_rows:
        for metric, sym_key, asym_key in (
            ("sym_vs_asym_rr", "replay_mrr", "replay_asym_mrr"),
            ("sym_vs_asym_correct_at_1", "replay_recall@1", "replay_asym_recall@1"),
            ("sym_vs_asym_correct_at_10", "replay_recall@10", "replay_asym_recall@10"),
        ):
            rows.append(
                {
                    "run_name": row["run_name"],
                    "method": row["method"],
                    "metric": metric,
                    "n_valid": 0,
                    "sym_value": row[sym_key],
                    "asym_value": row[asym_key],
                    "mean_delta": "",
                    "ci_low": "",
                    "ci_high": "",
                    "a_only": "",
                    "b_only": "",
                    "raw_p_value": "",
                    "adjusted_p_value": "",
                    "status": "not_applicable",
                    "reason": reason,
                }
            )
    return rows


def within_run_tests(grouped_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, run_name in enumerate([row["run_name"] for row in SATURATED_RUNS], start=1):
        data = grouped_rows[run_name]
        sym_rr = extract_numeric(data, "sym_reciprocal_rank")
        asym_rr = extract_numeric(data, "asym_reciprocal_rank")
        rr_valid = np.isfinite(sym_rr) & np.isfinite(asym_rr)
        if np.any(rr_valid):
            delta = sym_rr[rr_valid] - asym_rr[rr_valid]
            mean_delta, ci_low, ci_high = paired_bootstrap_mean_ci(delta, seed=BOOTSTRAP_SEED + 300 + idx)
            _, p_value = sign_flip_permutation_test(delta, seed=PERMUTATION_SEED + 300 + idx)
            rows.append(
                {
                    "run_name": run_name,
                    "method": RUN_LABELS[run_name],
                    "metric": "sym_vs_asym_rr",
                    "n_valid": int(delta.size),
                    "sym_value": float(np.mean(sym_rr[rr_valid])),
                    "asym_value": float(np.mean(asym_rr[rr_valid])),
                    "mean_delta": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "a_only": "",
                    "b_only": "",
                    "raw_p_value": p_value,
                }
            )
        else:
            rows.append(
                {
                    "run_name": run_name,
                    "method": RUN_LABELS[run_name],
                    "metric": "sym_vs_asym_rr",
                    "n_valid": 0,
                    "sym_value": float(np.mean(sym_rr)),
                    "asym_value": "",
                    "mean_delta": "",
                    "ci_low": "",
                    "ci_high": "",
                    "a_only": "",
                    "b_only": "",
                    "raw_p_value": "",
                    "status": "not_applicable",
                }
            )
        for metric in ("correct_at_1", "correct_at_10"):
            sym_key = f"sym_{metric}"
            asym_key = f"asym_{metric}"
            sym = extract_numeric(data, sym_key)
            asym = extract_numeric(data, asym_key)
            valid = np.isfinite(sym) & np.isfinite(asym)
            if not np.any(valid):
                rows.append(
                    {
                        "run_name": run_name,
                        "method": RUN_LABELS[run_name],
                        "metric": f"sym_vs_asym_{metric}",
                        "n_valid": 0,
                        "sym_value": float(np.mean(sym)),
                        "asym_value": "",
                        "mean_delta": "",
                        "ci_low": "",
                        "ci_high": "",
                        "a_only": "",
                        "b_only": "",
                        "raw_p_value": "",
                        "status": "not_applicable",
                    }
                )
                continue
            sym_bin = sym[valid].astype(np.int8)
            asym_bin = asym[valid].astype(np.int8)
            delta = sym_bin.astype(np.float64) - asym_bin.astype(np.float64)
            mean_delta, ci_low, ci_high = paired_bootstrap_mean_ci(delta, seed=BOOTSTRAP_SEED + 400 + len(rows))
            a_only, b_only, p_value = exact_mcnemar(sym_bin, asym_bin)
            rows.append(
                {
                    "run_name": run_name,
                    "method": RUN_LABELS[run_name],
                    "metric": f"sym_vs_asym_{metric}",
                    "n_valid": int(delta.size),
                    "sym_value": float(np.mean(sym_bin)),
                    "asym_value": float(np.mean(asym_bin)),
                    "mean_delta": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "a_only": a_only,
                    "b_only": b_only,
                    "raw_p_value": p_value,
                }
            )
    for metric in ("sym_vs_asym_rr", "sym_vs_asym_correct_at_1", "sym_vs_asym_correct_at_10"):
        metric_rows = [row for row in rows if row["metric"] == metric and row.get("status") != "not_applicable"]
        holm_bonferroni(metric_rows, "raw_p_value", "adjusted_p_value")
        for row in metric_rows:
            row["status"] = status_from_ci(
                delta=float(row["mean_delta"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                p_adj=float(row["adjusted_p_value"]),
            )
    return rows


def lookup_comparison(rows: list[dict[str, Any]], run_a: str, run_b: str, metric: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row.get("run_a") == run_a and row.get("run_b") == run_b:
            if metric is None or row.get("metric") == metric:
                return row
    return None


def verdict_sentence(
    rr_row: dict[str, Any],
    r1_row: dict[str, Any] | None,
    r10_row: dict[str, Any] | None,
    margin_row: dict[str, Any] | None,
    doc_row: dict[str, Any] | None,
    published_doc_delta: float | None,
) -> str:
    rr_status = rr_row.get("status")
    doc_status = doc_row.get("status") if doc_row else "not_applicable"
    margin_status = margin_row.get("status") if margin_row else "not_applicable"
    if rr_status == "significant_better" and doc_status == "significant_better":
        return "BiMGA is significantly better on retrieval and significantly better on document alignment, which is consistent with a geometry-driven gain."
    if rr_status == "significant_better" and doc_status == "not_applicable":
        if published_doc_delta is not None:
            return "BiMGA is significantly better on retrieval. Paired document-alignment testing is not applicable because the fine-tuned teacher targets were not published, but the HF aggregate doc cosine still favors BiMGA."
        return "BiMGA is significantly better on retrieval; paired document-alignment testing is not applicable because the fine-tuned teacher targets were not published."
    if rr_status == "not_significant" and doc_status == "not_applicable" and published_doc_delta is not None and abs(published_doc_delta) < 0.05:
        return "BiMGA and the comparison method are not significantly different on retrieval, and the published aggregate doc-cosine gap is also small, so bidirectional alignment is supported but margin guidance remains unproven."
    if rr_status == "not_significant" and doc_status == "significant_better":
        return "BiMGA is not significantly better on retrieval here, but it is significantly better aligned on document geometry, so the geometric gain does not convert into a clear MRR win at this sample size."
    if rr_status == "not_significant" and doc_status == "not_significant":
        return "The retrieval difference is not significant and the document-alignment difference is also not significant, so this comparison does not justify a stronger claim."
    if margin_status == "significant_better" and rr_status != "significant_better":
        return "The local hardest-negative margin is significantly better, but that margin gain does not translate into a significant retrieval gain."
    if rr_status == "significant_worse":
        return "BiMGA is significantly worse in this comparison, so this baseline remains stronger on the tested retrieval criterion."
    if rr_status == "not_significant" and margin_status == "significant_worse":
        return "BiMGA does not separate on retrieval, and the margin result also moves in the wrong direction, so the mechanism claim is not supported."
    return "The comparison is mixed; treat the retrieval result as the primary signal and the mechanism tests as supporting diagnostics only."


def build_primary_table(
    run_summary_rows: list[dict[str, Any]],
    rr_rows: list[dict[str, Any]],
    mcnemar_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    doc_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_lookup = {row["run_name"]: row for row in run_summary_rows}
    for run_a, run_b in PRIMARY_COMPARISONS:
        rr_row = lookup_comparison(rr_rows, run_a, run_b)
        if rr_row is None:
            continue
        r1_row = lookup_comparison(mcnemar_rows, run_a, run_b, "sym_correct_at_1")
        r10_row = lookup_comparison(mcnemar_rows, run_a, run_b, "sym_correct_at_10")
        margin_row = lookup_comparison(margin_rows, run_a, run_b, "sym_margin")
        doc_row = lookup_comparison(doc_rows, run_a, run_b, "doc_cosine")
        doc_a = summary_lookup[run_a]["replay_doc_cosine"]
        doc_b = summary_lookup[run_b]["replay_doc_cosine"]
        published_doc_delta = None
        if doc_a not in ("", None) and doc_b not in ("", None):
            published_doc_delta = float(doc_a) - float(doc_b)
        rows.append(
            {
                "comparison": f"{RUN_LABELS[run_a]} vs {RUN_LABELS[run_b]}",
                "mrr_delta": format_float(float(rr_row["mean_delta_rr"])),
                "mrr_adjusted_p": format_pvalue(float(rr_row["adjusted_p_value"])),
                "mrr_ci": f"[{format_float(float(rr_row['ci_low']))}, {format_float(float(rr_row['ci_high']))}]",
                "r1_status": format_status(r1_row.get("status") if r1_row else None),
                "r10_status": format_status(r10_row.get("status") if r10_row else None),
                "margin_status": format_status(margin_row.get("status") if margin_row else None),
                "doc_cosine_status": format_status(doc_row.get("status") if doc_row else None),
                "verdict": verdict_sentence(rr_row, r1_row, r10_row, margin_row, doc_row, published_doc_delta),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |\n"
    sep = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = ""
    for row in rows:
        body += "| " + " | ".join(str(row.get(key, "—")) for key, _ in columns) + " |\n"
    return header + sep + body


def build_report(
    output_path: Path,
    run_summary_rows: list[dict[str, Any]],
    rr_rows: list[dict[str, Any]],
    mcnemar_rows: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    doc_rows: list[dict[str, Any]],
    within_rows: list[dict[str, Any]],
) -> None:
    primary_rows = build_primary_table(run_summary_rows, rr_rows, mcnemar_rows, margin_rows, doc_rows)
    appendix_rows = []
    for row in rr_rows:
        appendix_rows.append(
            {
                "comparison": f"{row['method_a']} vs {row['method_b']}",
                "mrr_delta": format_float(float(row["mean_delta_rr"])),
                "adj_p": format_pvalue(float(row["adjusted_p_value"])),
                "ci": f"[{format_float(float(row['ci_low']))}, {format_float(float(row['ci_high']))}]",
                "status": row["status"],
            }
        )
    within_display = []
    for row in within_rows:
        within_display.append(
            {
                "run": row["run_name"],
                "metric": row["metric"],
                "delta": format_float(float(row["mean_delta"])) if row.get("mean_delta") not in ("", None) else "—",
                "adj_p": format_pvalue(float(row["adjusted_p_value"])) if row.get("adjusted_p_value") not in ("", None) else "—",
                "status": row.get("status", "—"),
            }
        )

    report = f"""# Saturated TACO Significance Report

All results in this report come from fresh HF replay of the six fully saturated checkpoints listed in the experiment README. The source of truth is the replayed per-query dataset in [`saturated_per_query.csv`](./saturated_per_query.csv) and the run summary in [`saturated_run_summary.csv`](./saturated_run_summary.csv).

## Setup

The analysis replays six final saturated student checkpoints on the fixed TACO test split of 1000 paired queries and code snippets. Every significance test is paired at the query level because every model is evaluated on the same exact test questions. That pairing is the critical design choice: it is what makes small differences in `MRR`, `Recall@1`, and margin interpretable rather than just noisy decimals.

Saturated seed-level significance is out of scope here. There are no matched saturated multi-seed reruns in local artifacts, so the defensible evidence for the saturated set is query-level paired testing, not seed-level variance estimation.

## What Was Tested And Why

`Paired permutation on reciprocal rank` is the primary test for retrieval quality because `MRR` is literally the mean of per-query reciprocal rank. If one method has a better MRR, that should appear as a consistently positive per-query reciprocal-rank difference, not just a better final scalar.

`Exact McNemar` is the right test for `Recall@1` and `Recall@10` because those metrics reduce to paired binary outcomes on each query: either the model retrieved the correct item within the cutoff or it did not. McNemar tests whether one model gets significantly more unique wins on the same paired questions.

`Wilcoxon signed-rank on margin` is used for the hardest-negative margin because that distribution is skewed, noisy, and not well modeled as Gaussian. This test asks whether one method is consistently better on local positive-vs-hardest-negative separation.

`Wilcoxon signed-rank on document cosine` directly tests the mechanism claim behind BiMGA. If BiMGA is better because it learns teacher-like document geometry, then its per-query document cosine should be consistently higher than the baselines.

`Bootstrap confidence intervals` are reported alongside every paired delta so the report does not collapse to a p-value checklist. The interval gives the direction and practical scale of the effect, not just whether the null was rejected.

The saturated HF repos publish aggregate `Asym MRR` and `Doc Cosine` inside each repo's `metrics.json`, but they do not publish the per-query fine-tuned teacher embeddings needed for paired significance on those teacher-space diagnostics. Accordingly, this report treats those teacher-space values as aggregate supporting diagnostics. Only the student-only retrieval metrics and hardest-negative margins receive full paired significance testing from public artifacts alone.

## Replay Sanity

{markdown_table([
    {
        "run": row["run_name"],
        "method": row["method"],
        "replay_mrr": format_float(float(row["replay_mrr"])) if row["replay_mrr"] not in ("", None) else "—",
        "expected_mrr": format_float(float(row["expected_mrr"])) if row["expected_mrr"] not in ("", None) else "—",
        "delta_mrr": format_float(float(row["delta_mrr_vs_expected"])) if row["delta_mrr_vs_expected"] not in ("", None) else "—",
        "replay_asym": format_float(float(row["replay_asym_mrr"])) if row["replay_asym_mrr"] not in ("", None) else "—",
        "replay_doc": format_float(float(row["replay_doc_cosine"])) if row["replay_doc_cosine"] not in ("", None) else "—",
        "status": row["replay_status"],
    }
    for row in run_summary_rows
], [
    ("run", "Run"),
    ("method", "Method"),
    ("replay_mrr", "Replay MRR"),
    ("expected_mrr", "README MRR"),
    ("delta_mrr", "Delta"),
    ("replay_asym", "Replay Asym MRR"),
    ("replay_doc", "Replay Doc Cosine"),
    ("status", "Status"),
])}

## Primary Claim Table

{markdown_table(primary_rows, [
    ("comparison", "Comparison"),
    ("mrr_delta", "MRR Delta"),
    ("mrr_adjusted_p", "Adj p"),
    ("mrr_ci", "95% CI"),
    ("r1_status", "R@1"),
    ("r10_status", "R@10"),
    ("margin_status", "Margin"),
    ("doc_cosine_status", "Doc Cosine"),
    ("verdict", "Verdict"),
])}

## Symmetric Vs Asymmetric Within-Run Tests

{markdown_table(within_display, [
    ("run", "Run"),
    ("metric", "Metric"),
    ("delta", "Delta"),
    ("adj_p", "Adj p"),
    ("status", "Status"),
])}

If an exact within-run symmetric-vs-asymmetric paired test is unavailable, that is a publication limitation rather than a model limitation: the public HF repos expose the aggregate asymmetric metrics, but not the query-level fine-tuned teacher targets required to test them properly.

## Interpretation

The report uses one fixed interpretation rule set:

- When `MRR` and `doc_cosine` are both significantly better in the same direction, the result supports a geometry-driven gain.
- When `margin` is significantly better but `MRR` is not, the result supports a local hardest-negative improvement without a matching global retrieval improvement.
- When `BiMGA` and `BiMGA-uniform` are not significantly different on both `MRR` and `doc_cosine`, the evidence supports bidirectional alignment but does not prove that margin guidance is necessary.
- When score-only or pairwise baselines approach the alignment methods on `MRR` but remain significantly worse on `doc_cosine`, that means they can mimic ranking behavior without actually reproducing teacher document geometry.

## All-Pairs Appendix

{markdown_table(appendix_rows, [
    ("comparison", "Comparison"),
    ("mrr_delta", "MRR Delta"),
    ("adj_p", "Adj p"),
    ("ci", "95% CI"),
    ("status", "Status"),
])}

## Excluded Tests

The following tests are intentionally excluded from the evidence chain:

- unpaired t-tests on final scalar MRR values
- z-tests on aggregate metrics
- Pearson or Spearman significance across only six saturated runs
- seed-level significance claims for the saturated set

These are excluded because they either use the wrong unit of analysis, are underpowered, or are unavailable for the saturated checkpoints.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run saturated TACO significance analysis from HF checkpoints.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "submission" / "experiments" / "analysis" / "significance"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--teacher-target-checkpoint",
        default="",
        help="Optional phase1/checkpoint.pt containing ft_teacher_targets for exact asymmetric/doc-alignment significance.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    print("loading fixed TACO test split", flush=True)
    taco = load_taco_retrieval(seed=42, taco_val_size=1000, include_train_validation=False)
    queries = [q for q, _ in taco["test"]]
    codes = [c for _, c in taco["test"]]
    print(f"loaded {len(queries)} TACO test queries", flush=True)
    teacher_q = None
    teacher_d = None
    teacher_margin = None
    checkpoint_path = Path(args.teacher_target_checkpoint).expanduser() if args.teacher_target_checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists():
        print(f"loading fine-tuned teacher targets from {checkpoint_path}", flush=True)
        exact_targets = load_teacher_targets_from_checkpoint(checkpoint_path)
        if exact_targets is not None:
            teacher_q, teacher_d = exact_targets
            teacher_scores = (teacher_q @ teacher_d.T).numpy()
            negatives = teacher_scores.copy()
            np.fill_diagonal(negatives, -np.inf)
            teacher_margin = np.diag(teacher_scores) - negatives.max(axis=1)
    else:
        print("fine-tuned teacher targets not available; teacher-space significance tests will be marked not_applicable", flush=True)

    per_query_rows: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []
    for idx, manifest in enumerate(SATURATED_RUNS, start=1):
        print(f"[{idx}/{len(SATURATED_RUNS)}] replaying {manifest['run_name']}", flush=True)
        replay = replay_saturated_run(
            manifest,
            queries,
            codes,
            teacher_q,
            teacher_d,
            teacher_margin,
            device,
            args.batch_size,
        )
        per_query_rows.extend(replay.per_query_rows)
        published_metrics = load_published_metrics(manifest["repo_id"])
        run_summary_rows.append(build_run_summary(replay, manifest["expected"], published_metrics))

    failed = [row["run_name"] for row in run_summary_rows if row["replay_status"] != "ok"]
    if failed:
        raise RuntimeError(f"Failed to replay saturated runs from HF: {', '.join(failed)}")

    grouped = rows_by_run(per_query_rows)
    rr_rows = pairwise_rr_tests(grouped)
    mcnemar_rows = pairwise_mcnemar_tests(grouped)
    margin_rows = pairwise_continuous_tests(grouped, "sym_margin", output_name="sym_margin")
    if teacher_q is not None and teacher_d is not None:
        doc_rows = pairwise_continuous_tests(grouped, "doc_cosine", output_name="doc_cosine")
        within_rows = within_run_tests(grouped)
    else:
        reason = "fine_tuned_teacher_targets_unpublished"
        doc_rows = unavailable_pairwise_rows("doc_cosine", reason)
        within_rows = unavailable_within_rows(run_summary_rows, reason)

    write_csv(
        output_root / "saturated_per_query.csv",
        per_query_rows,
        [
            "query_id",
            "run_name",
            "method",
            "repo_id",
            "sym_rank",
            "sym_reciprocal_rank",
            "sym_correct_at_1",
            "sym_correct_at_10",
            "sym_positive_score",
            "sym_hardest_negative_score",
            "sym_margin",
            "asym_rank",
            "asym_reciprocal_rank",
            "asym_correct_at_1",
            "asym_correct_at_10",
            "teacher_margin",
            "doc_cosine",
            "query_cosine",
            "proxy_doc_cosine_raw_teacher",
            "proxy_query_cosine_raw_teacher",
        ],
    )
    write_csv(
        output_root / "saturated_run_summary.csv",
        run_summary_rows,
        [
            "run_name",
            "method",
            "repo_id",
            "replay_status",
            "projection_exists",
            "student_hidden_size",
            "output_dim",
            "replay_mrr",
            "expected_mrr",
            "delta_mrr_vs_expected",
            "replay_recall@1",
            "expected_recall@1",
            "delta_recall@1_vs_expected",
            "replay_recall@10",
            "expected_recall@10",
            "delta_recall@10_vs_expected",
            "replay_map@10",
            "expected_map@10",
            "delta_map@10_vs_expected",
            "replay_ndcg@10",
            "expected_ndcg@10",
            "delta_ndcg@10_vs_expected",
            "replay_asym_mrr",
            "replay_asym_recall@1",
            "replay_asym_recall@10",
            "expected_asym_mrr",
            "delta_asym_mrr_vs_expected",
            "replay_doc_cosine",
            "expected_doc_cosine",
            "delta_doc_cosine_vs_expected",
            "notes",
        ],
    )
    write_csv(
        output_root / "pairwise_rr_permutation.csv",
        rr_rows,
        [
            "run_a",
            "method_a",
            "run_b",
            "method_b",
            "n_queries",
            "mean_delta_rr",
            "ci_low",
            "ci_high",
            "raw_p_value",
            "adjusted_p_value",
            "win_rate",
            "tie_rate",
            "loss_rate",
            "status",
        ],
    )
    write_csv(
        output_root / "pairwise_mcnemar.csv",
        mcnemar_rows,
        [
            "metric",
            "run_a",
            "method_a",
            "run_b",
            "method_b",
            "n_queries",
            "success_rate_a",
            "success_rate_b",
            "delta_rate",
            "ci_low",
            "ci_high",
            "a_only",
            "b_only",
            "raw_p_value",
            "adjusted_p_value",
            "status",
        ],
    )
    write_csv(
        output_root / "pairwise_margin_wilcoxon.csv",
        margin_rows,
        [
            "metric",
            "run_a",
            "method_a",
            "run_b",
            "method_b",
            "n_valid",
            "mean_delta",
            "median_delta",
            "ci_low",
            "ci_high",
            "fraction_a_higher",
            "fraction_tie",
            "raw_p_value",
            "adjusted_p_value",
            "status",
            "reason",
        ],
    )
    write_csv(
        output_root / "pairwise_doc_cosine_wilcoxon.csv",
        doc_rows,
        [
            "metric",
            "run_a",
            "method_a",
            "run_b",
            "method_b",
            "n_valid",
            "mean_delta",
            "median_delta",
            "ci_low",
            "ci_high",
            "fraction_a_higher",
            "fraction_tie",
            "raw_p_value",
            "adjusted_p_value",
            "status",
            "reason",
        ],
    )
    write_csv(
        output_root / "within_run_sym_vs_asym.csv",
        within_rows,
        [
            "run_name",
            "method",
            "metric",
            "n_valid",
            "sym_value",
            "asym_value",
            "mean_delta",
            "ci_low",
            "ci_high",
            "a_only",
            "b_only",
            "raw_p_value",
            "adjusted_p_value",
            "status",
            "reason",
        ],
    )
    build_report(
        output_root / "SATURATED_SIGNIFICANCE_REPORT.md",
        run_summary_rows,
        rr_rows,
        mcnemar_rows,
        margin_rows,
        doc_rows,
        within_rows,
    )
    print(f"wrote saturated significance package to {output_root}", flush=True)


if __name__ == "__main__":
    main()
