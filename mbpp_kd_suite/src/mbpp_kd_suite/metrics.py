from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .constants import METHOD_ORDER, TRAINED_BASELINE_NAME
from .modeling import (
    StudentQueryEncoder,
    encode_student_texts,
    encode_texts_backbone,
    infer_model_encoding_spec,
)


def reciprocal_rank_metrics(score_matrix: np.ndarray) -> dict[str, float]:
    n = score_matrix.shape[0]
    ranks: list[int] = []
    for i in range(n):
        order = np.argsort(-score_matrix[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    reciprocal = [1.0 / rank for rank in ranks]
    return {
        "MRR": float(np.mean(reciprocal)),
        "Recall@1": float(np.mean([rank <= 1 for rank in ranks])),
        "Recall@5": float(np.mean([rank <= 5 for rank in ranks])),
        "Recall@10": float(np.mean([rank <= 10 for rank in ranks])),
        "MedianRank": float(np.median(ranks)),
    }


def score_metrics_from_embeddings(query_embs: torch.Tensor, doc_embs: torch.Tensor) -> dict[str, float]:
    scores = (query_embs @ doc_embs.T).numpy()
    return reciprocal_rank_metrics(scores)


@torch.no_grad()
def evaluate_asymmetric(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    queries: list[str],
    fixed_doc_embs: torch.Tensor,
    max_query_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    query_embs = encode_student_texts(
        student_model=student_model,
        tokenizer=tokenizer,
        texts=queries,
        text_role="query",
        max_length=max_query_length,
        batch_size=eval_batch_size,
        device=device,
        desc="eval_asym_q",
    )
    return score_metrics_from_embeddings(query_embs, fixed_doc_embs.cpu())


@torch.no_grad()
def evaluate_symmetric_student(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    queries: list[str],
    codes: list[str],
    max_query_length: int,
    max_code_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    query_embs = encode_student_texts(
        student_model=student_model,
        tokenizer=tokenizer,
        texts=queries,
        text_role="query",
        max_length=max_query_length,
        batch_size=eval_batch_size,
        device=device,
        desc="eval_sym_q",
    )
    code_embs = encode_student_texts(
        student_model=student_model,
        tokenizer=tokenizer,
        texts=codes,
        text_role="document",
        max_length=max_code_length,
        batch_size=eval_batch_size,
        device=device,
        desc="eval_sym_d",
    )
    return score_metrics_from_embeddings(query_embs, code_embs)


def evaluate_student_mode(
    eval_mode: str,
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    queries: list[str],
    codes: list[str],
    fixed_doc_embs: torch.Tensor,
    max_query_length: int,
    max_code_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if eval_mode == "symmetric":
        return evaluate_symmetric_student(
            student_model=student_model,
            tokenizer=tokenizer,
            queries=queries,
            codes=codes,
            max_query_length=max_query_length,
            max_code_length=max_code_length,
            eval_batch_size=eval_batch_size,
            device=device,
        )
    if fixed_doc_embs.shape[1] != student_model.output_hidden_size:
        raise ValueError(
            "Asymmetric evaluation requires the student output dimension to match the fixed document "
            "embeddings. Use --eval-mode symmetric for teacher-free student baselines."
        )
    return evaluate_asymmetric(
        student_model=student_model,
        tokenizer=tokenizer,
        queries=queries,
        fixed_doc_embs=fixed_doc_embs,
        max_query_length=max_query_length,
        eval_batch_size=eval_batch_size,
        device=device,
    )


@torch.no_grad()
def evaluate_symmetric_backbone(
    model_name: str,
    val_queries: list[str],
    val_codes: list[str],
    test_queries: list[str],
    test_codes: list[str],
    max_query_length: int,
    max_code_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    encoding_spec = infer_model_encoding_spec(
        model_name,
        getattr(model.config, "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None),
    )
    model.eval()

    val_q = encode_texts_backbone(
        model=model,
        tokenizer=tokenizer,
        texts=val_queries,
        text_role="query",
        encoding_spec=encoding_spec,
        max_length=max_query_length,
        batch_size=eval_batch_size,
        device=device,
        desc="direct_val_q",
    )
    val_d = encode_texts_backbone(
        model=model,
        tokenizer=tokenizer,
        texts=val_codes,
        text_role="document",
        encoding_spec=encoding_spec,
        max_length=max_code_length,
        batch_size=eval_batch_size,
        device=device,
        desc="direct_val_d",
    )
    test_q = encode_texts_backbone(
        model=model,
        tokenizer=tokenizer,
        texts=test_queries,
        text_role="query",
        encoding_spec=encoding_spec,
        max_length=max_query_length,
        batch_size=eval_batch_size,
        device=device,
        desc="direct_test_q",
    )
    test_d = encode_texts_backbone(
        model=model,
        tokenizer=tokenizer,
        texts=test_codes,
        text_role="document",
        encoding_spec=encoding_spec,
        max_length=max_code_length,
        batch_size=eval_batch_size,
        device=device,
        desc="direct_test_d",
    )
    return {
        "validation": score_metrics_from_embeddings(val_q, val_d),
        "test": score_metrics_from_embeddings(test_q, test_d),
    }


@torch.no_grad()
def query_alignment_cosine(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    queries: list[str],
    target_query_embs: torch.Tensor,
    max_query_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> float:
    student_q = encode_student_texts(
        student_model=student_model,
        tokenizer=tokenizer,
        texts=queries,
        text_role="query",
        max_length=max_query_length,
        batch_size=eval_batch_size,
        device=device,
        desc="align_q",
    )
    return float(F.cosine_similarity(student_q, target_query_embs.cpu(), dim=-1).mean().item())


@torch.no_grad()
def doc_alignment_cosine_student_vs_target(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    codes: list[str],
    target_doc_embs: torch.Tensor,
    max_code_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> float:
    student_d = encode_student_texts(
        student_model=student_model,
        tokenizer=tokenizer,
        texts=codes,
        text_role="document",
        max_length=max_code_length,
        batch_size=eval_batch_size,
        device=device,
        desc="align_d",
    )
    return float(F.cosine_similarity(student_d, target_doc_embs.cpu(), dim=-1).mean().item())


def summarize_analysis(result: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    direct_small = result.get("direct_small_student")
    supervised_student = result.get(TRAINED_BASELINE_NAME)

    for method_name in METHOD_ORDER:
        if method_name not in result:
            continue

        method_result = result[method_name]
        entry: dict[str, float] = {
            "validation_minus_test_mrr": method_result["validation"]["MRR"] - method_result["test"]["MRR"],
        }
        if direct_small is not None:
            entry["test_mrr_gap_vs_direct_small"] = (
                method_result["test"]["MRR"] - direct_small["test"]["MRR"]
            )
        if supervised_student is not None and method_name != TRAINED_BASELINE_NAME:
            entry["test_mrr_gap_vs_supervised_student"] = (
                method_result["test"]["MRR"] - supervised_student["test"]["MRR"]
            )
        diagnostics = method_result.get("diagnostics", {})
        symmetric_test = diagnostics.get("symmetric_test")
        asymmetric_test = diagnostics.get("asymmetric_test")
        if symmetric_test is not None and asymmetric_test is not None:
            entry["symmetric_minus_asymmetric_test_mrr"] = (
                symmetric_test["MRR"] - asymmetric_test["MRR"]
            )
        q_align = diagnostics.get("query_alignment_cosine")
        if q_align is not None:
            entry["query_alignment_train_minus_test"] = q_align["train"] - q_align["test"]
        analysis[method_name] = entry

    return analysis
