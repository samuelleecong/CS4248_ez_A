from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import DistillTargets, RetrievalSplit, RetrievalSplits, TrainConfig
from .constants import METHOD_ORDER, TRAINED_BASELINE_NAME
from .data import make_indexed_pair_dataloader, make_pair_dataloader, make_query_dataloader
from .metrics import (
    doc_alignment_cosine_student_vs_target,
    evaluate_asymmetric,
    evaluate_student_mode,
    evaluate_symmetric_student,
    query_alignment_cosine,
)
from .modeling import StudentQueryEncoder, encode_student_backbone_texts, to_device
from .runtime import maybe_empty_device_cache


@dataclass
class LossComponents:
    one_hot: torch.Tensor
    distill_kl: torch.Tensor
    align: torch.Tensor
    pairwise: torch.Tensor
    relation: torch.Tensor
    dark_kl: torch.Tensor
    dark_confidence: float = 0.0

    @classmethod
    def from_one_hot(cls, one_hot: torch.Tensor, device: torch.device) -> LossComponents:
        zero = torch.zeros((), device=device)
        return cls(
            one_hot=one_hot,
            distill_kl=zero,
            align=zero,
            pairwise=zero,
            relation=zero,
            dark_kl=zero,
        )

    def total(self, cfg: TrainConfig) -> torch.Tensor:
        return (
            self.one_hot
            + cfg.distill_weight * self.distill_kl
            + cfg.align_weight * self.align
            + cfg.pair_weight * self.pairwise
            + cfg.relation_weight * self.relation
            + cfg.distill_weight * self.dark_kl
        )

    def scalar_items(self) -> dict[str, float]:
        return {
            "one_hot": float(self.one_hot.item()),
            "distill_kl": float(self.distill_kl.item()),
            "align": float(self.align.item()),
            "pairwise": float(self.pairwise.item()),
            "relation": float(self.relation.item()),
            "dark_kl": float(self.dark_kl.item()),
            "dark_confidence": self.dark_confidence,
        }


@torch.no_grad()
def initialize_projection_from_targets(
    cfg: TrainConfig,
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    train_queries: list[str],
    train_codes: list[str],
    targets: DistillTargets,
    device: torch.device,
) -> dict[str, float] | None:
    if cfg.projection_init == "none" or isinstance(student_model.proj, nn.Identity):
        return None

    source_embs = [
        encode_student_backbone_texts(
            student_model=student_model,
            tokenizer=tokenizer,
            texts=train_queries,
            text_role="query",
            max_length=cfg.max_query_length,
            batch_size=cfg.eval_batch_size,
            device=device,
            desc="proj_init_q",
        )
    ]
    target_embs = [targets.train_query.cpu()]

    if cfg.projection_init == "least_squares_both":
        source_embs.append(
            encode_student_backbone_texts(
                student_model=student_model,
                tokenizer=tokenizer,
                texts=train_codes,
                text_role="document",
                max_length=cfg.max_code_length,
                batch_size=cfg.eval_batch_size,
                device=device,
                desc="proj_init_d",
            )
        )
        target_embs.append(targets.train_doc.cpu())

    design = torch.cat(source_embs, dim=0).float()
    target = torch.cat(target_embs, dim=0).float()
    solution = torch.linalg.lstsq(design, target).solution
    student_model.proj.weight.data.copy_(solution.T.to(student_model.proj.weight.device))

    projected = F.normalize(design @ solution, p=2, dim=-1)
    cosine = F.cosine_similarity(projected, target, dim=-1).mean().item()
    mse = torch.mean((projected - target) ** 2).item()
    return {
        "projection_init_cosine": float(cosine),
        "projection_init_mse": float(mse),
        "projection_init_examples": float(design.shape[0]),
    }


def one_hot_loss(student_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(student_scores.size(0), device=student_scores.device)
    return F.cross_entropy(student_scores / temperature, labels)


def distill_kl(student_scores: torch.Tensor, teacher_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_scores / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def align_loss(student_query_emb: torch.Tensor, target_query_emb: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1).mean()


def contrastive_kd_loss(
    student_query_emb: torch.Tensor,
    teacher_query_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    relation_scores = student_query_emb @ teacher_query_emb.T
    return one_hot_loss(relation_scores, temperature)


def margin_mse_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
    batch_size = student_scores.size(0)
    if batch_size < 2:
        return torch.zeros((), device=student_scores.device)
    pos_s = student_scores.diag().unsqueeze(1)
    pos_t = teacher_scores.diag().unsqueeze(1)
    student_margin = pos_s - student_scores
    teacher_margin = pos_t - teacher_scores
    mask = 1.0 - torch.eye(batch_size, device=student_scores.device)
    return (mask * (student_margin - teacher_margin) ** 2).sum() / mask.sum()


def all_pairs_distill_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
    """KL on 2-way softmax over *(positive, every in-batch other doc)* — all ``j != i`` (PairDistill / Sam pairdistil).

    For each query i and column j, logits are ``[sim(q_i, d_i^+), sim(q_i, d_j)]``; teacher and student each
    yield a binary distribution; loss is ``KL(P_teacher || P_student)`` averaged over all off-diagonal pairs.
    """
    b = student_scores.size(0)
    if b < 2:
        return torch.zeros((), device=student_scores.device)
    pos_t = teacher_scores.diag().unsqueeze(1).expand(b, b)
    pos_s = student_scores.diag().unsqueeze(1).expand(b, b)
    pair_teacher = torch.stack([pos_t, teacher_scores], dim=2)
    pair_student = torch.stack([pos_s, student_scores], dim=2)
    teacher_probs = F.softmax(pair_teacher, dim=2)
    student_log_probs = F.log_softmax(pair_student, dim=2)
    kl_per_pair = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=2)
    mask = 1.0 - torch.eye(b, device=student_scores.device, dtype=kl_per_pair.dtype)
    return (kl_per_pair * mask).sum() / mask.sum()


def bimga_loss(
    student_query_emb: torch.Tensor,
    student_doc_emb: torch.Tensor,
    target_query_emb: torch.Tensor,
    target_doc_emb: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Bidirectional Margin-Guided Alignment (BiMGA).

    Aligns both query and document embeddings to teacher space, weighted by the
    teacher's ranking confidence (margin between positive and hardest negative).
    """
    batch_size = teacher_scores.size(0)
    if batch_size < 2:
        q_align = torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1).mean()
        d_align = torch.linalg.vector_norm(student_doc_emb - target_doc_emb, dim=-1).mean()
        return q_align + d_align

    # Compute teacher margin: positive score - hardest in-batch negative score
    pos_scores = teacher_scores.diag()  # (B,)
    mask = torch.eye(batch_size, device=teacher_scores.device, dtype=torch.bool)
    neg_scores = teacher_scores.masked_fill(mask, -1e9)
    hardest_neg = neg_scores.max(dim=-1).values  # (B,)
    margin = pos_scores - hardest_neg  # (B,)

    # Sigmoid weighting: confident teacher -> strong alignment
    weights = torch.sigmoid(margin / temperature)  # (B,)

    # Weighted bidirectional alignment
    q_dist = torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1)  # (B,)
    d_dist = torch.linalg.vector_norm(student_doc_emb - target_doc_emb, dim=-1)  # (B,)
    weighted_align = (weights * (q_dist + d_dist)).mean()
    return weighted_align


def bimga_uniform_loss(
    student_query_emb: torch.Tensor,
    student_doc_emb: torch.Tensor,
    target_query_emb: torch.Tensor,
    target_doc_emb: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional alignment WITHOUT margin weighting (ablation A2).

    Aligns both query and document embeddings to teacher space with uniform
    weight (w_i = 1.0 for all samples). Tests whether bidirectional alignment
    helps independently of the margin-guided weighting.
    """
    q_dist = torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1)
    d_dist = torch.linalg.vector_norm(student_doc_emb - target_doc_emb, dim=-1)
    return (q_dist + d_dist).mean()


def bimga_query_only_loss(
    student_query_emb: torch.Tensor,
    target_query_emb: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Margin-weighted query-only alignment (ablation A3).

    Applies margin-guided weighting to query alignment only, without document
    alignment. Tests whether margin weighting helps independently of
    bidirectional alignment.
    """
    batch_size = teacher_scores.size(0)
    if batch_size < 2:
        return torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1).mean()

    pos_scores = teacher_scores.diag()
    mask = torch.eye(batch_size, device=teacher_scores.device, dtype=torch.bool)
    neg_scores = teacher_scores.masked_fill(mask, -1e9)
    hardest_neg = neg_scores.max(dim=-1).values
    margin = pos_scores - hardest_neg

    weights = torch.sigmoid(margin / temperature)
    q_dist = torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1)
    return (weights * q_dist).mean()


def bimga_progressive_loss(
    student_query_emb: torch.Tensor,
    student_doc_emb: torch.Tensor,
    target_query_emb: torch.Tensor,
    target_doc_emb: torch.Tensor,
    teacher_scores: torch.Tensor,
    base_temperature: float,
    epoch: int,
    total_epochs: int,
    tau_max: float = 2.0,
    tau_min: float = 0.1,
) -> torch.Tensor:
    """BiMGA with annealing temperature: uniform early -> selective late."""
    batch_size = teacher_scores.size(0)
    if batch_size < 2:
        q_align = torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1).mean()
        d_align = torch.linalg.vector_norm(student_doc_emb - target_doc_emb, dim=-1).mean()
        return q_align + d_align

    progress = epoch / max(total_epochs, 1)
    effective_tau = tau_max * (1 - progress) + tau_min * progress

    pos_scores = teacher_scores.diag()
    mask = torch.eye(batch_size, device=teacher_scores.device, dtype=torch.bool)
    neg_scores = teacher_scores.masked_fill(mask, -1e9)
    hardest_neg = neg_scores.max(dim=-1).values
    margin = pos_scores - hardest_neg
    weights = torch.sigmoid(margin / effective_tau)

    q_dist = torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1)
    d_dist = torch.linalg.vector_norm(student_doc_emb - target_doc_emb, dim=-1)
    return (weights * (q_dist + d_dist)).mean()


def precompute_hard_negatives(teacher_targets: DistillTargets, k: int = 32) -> torch.Tensor:
    """Pre-compute top-k hard negative doc indices for each training query."""
    q = teacher_targets.train_query
    d = teacher_targets.train_doc
    n = q.size(0)
    all_indices = []
    chunk_size = 1024
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = q[start:end] @ d.T  # (chunk, N)
        for i in range(end - start):
            sim[i, start + i] = -1e9  # mask positive
        topk_indices = sim.topk(k, dim=-1).indices
        all_indices.append(topk_indices)
    return torch.cat(all_indices, dim=0)  # (N, K)


def bimga_bridge_loss(
    student_query_emb: torch.Tensor,
    student_doc_emb: torch.Tensor,
    target_query_emb: torch.Tensor,
    target_doc_emb: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
    shared_ratio: float = 0.5,
    ortho_weight: float = 0.01,
) -> torch.Tensor:
    """BiMGA with semantic bridge: align shared subspace, regularize orthogonality."""
    batch_size = teacher_scores.size(0)
    d = student_query_emb.size(-1)
    k = int(d * shared_ratio)

    sq_shared, sq_specific = student_query_emb[:, :k], student_query_emb[:, k:]
    sd_shared, sd_specific = student_doc_emb[:, :k], student_doc_emb[:, k:]
    tq_shared = target_query_emb[:, :k]
    td_shared = target_doc_emb[:, :k]

    if batch_size < 2:
        q_align = torch.linalg.vector_norm(sq_shared - tq_shared, dim=-1).mean()
        d_align = torch.linalg.vector_norm(sd_shared - td_shared, dim=-1).mean()
        return q_align + d_align

    # BiMGA-style margin weighting on shared subspace
    pos_scores = teacher_scores.diag()
    mask = torch.eye(batch_size, device=teacher_scores.device, dtype=torch.bool)
    neg_scores = teacher_scores.masked_fill(mask, -1e9)
    hardest_neg = neg_scores.max(dim=-1).values
    margin = pos_scores - hardest_neg
    weights = torch.sigmoid(margin / temperature)

    q_dist = torch.linalg.vector_norm(sq_shared - tq_shared, dim=-1)
    d_dist = torch.linalg.vector_norm(sd_shared - td_shared, dim=-1)
    align = (weights * (q_dist + d_dist)).mean()

    # Cross-correlation orthogonality (Barlow Twins style)
    if d > k and k > 0:
        cross_q = (sq_shared.T @ sq_specific) / batch_size
        cross_d = (sd_shared.T @ sd_specific) / batch_size
        ortho = cross_q.pow(2).mean() + cross_d.pow(2).mean()
    else:
        ortho = torch.zeros((), device=student_query_emb.device)

    return align + ortho_weight * ortho


def pointwise_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_scores, teacher_scores)


def pairwise_preference_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
    hard_negatives: int,
) -> torch.Tensor:
    """BCE on margins for teacher's top-``hard_negatives`` **negative** columns per row (not all in-batch pairs)."""
    batch_size = student_scores.size(0)
    if batch_size < 2:
        return torch.zeros((), device=student_scores.device)

    negative_k = min(hard_negatives, batch_size - 1)
    if negative_k <= 0:
        return torch.zeros((), device=student_scores.device)

    mask = torch.eye(batch_size, dtype=torch.bool, device=student_scores.device)
    teacher_negatives = teacher_scores.masked_fill(mask, -1e9)
    negative_indices = teacher_negatives.topk(negative_k, dim=-1).indices
    row_indices = torch.arange(batch_size, device=student_scores.device).unsqueeze(-1)

    student_positive = student_scores.diag().unsqueeze(-1)
    teacher_positive = teacher_scores.diag().unsqueeze(-1)
    student_negative = student_scores[row_indices, negative_indices]
    teacher_negative = teacher_scores[row_indices, negative_indices]

    teacher_preference = torch.sigmoid((teacher_positive - teacher_negative) / temperature)
    student_logits = (student_positive - student_negative) / temperature
    return F.binary_cross_entropy_with_logits(student_logits, teacher_preference)


def adam_dark_example_loss(
    student_query_emb: torch.Tensor,
    teacher_query_emb: torch.Tensor,
    teacher_doc_embs: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
    hard_negatives: int,
    dark_mix_ratio: float,
) -> tuple[torch.Tensor, float]:
    batch_size = teacher_scores.size(0)
    if batch_size < 2:
        return torch.zeros((), device=teacher_scores.device), 0.0

    negative_k = min(hard_negatives, batch_size - 1)
    if negative_k <= 0:
        return torch.zeros((), device=teacher_scores.device), 0.0

    mask = torch.eye(batch_size, dtype=torch.bool, device=teacher_scores.device)
    teacher_negatives = teacher_scores.masked_fill(mask, -1e9)
    negative_indices = teacher_negatives.topk(negative_k, dim=-1).indices
    row_indices = torch.arange(batch_size, device=teacher_scores.device).unsqueeze(-1)

    positive_docs = teacher_doc_embs[row_indices.squeeze(-1)]
    hard_negative_docs = teacher_doc_embs[negative_indices]
    dark_docs = F.normalize(
        dark_mix_ratio * positive_docs.unsqueeze(1) + (1.0 - dark_mix_ratio) * hard_negative_docs,
        p=2,
        dim=-1,
    )

    candidate_docs = torch.cat([positive_docs.unsqueeze(1), dark_docs], dim=1)
    student_scores_subset = torch.einsum("bd,bkd->bk", student_query_emb, candidate_docs)
    teacher_scores_subset = torch.einsum("bd,bkd->bk", teacher_query_emb, candidate_docs)

    student_log_probs = F.log_softmax(student_scores_subset / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_scores_subset / temperature, dim=-1)
    per_row_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    teacher_margin = teacher_scores_subset[:, 0] - teacher_scores_subset[:, 1:].mean(dim=-1)
    confidence = torch.sigmoid(teacher_margin / temperature)
    return (per_row_kl * confidence).mean(), float(confidence.mean().item())


def fit_hpd_targets(cfg: TrainConfig, full_teacher_targets: DistillTargets) -> DistillTargets:
    fit_matrix = torch.cat(
        [full_teacher_targets.train_query, full_teacher_targets.train_doc], dim=0
    ).detach().cpu().numpy()
    pca = PCA(n_components=cfg.hpd_dim, random_state=cfg.seed)
    pca.fit(fit_matrix)

    mean = torch.from_numpy(pca.mean_.astype(np.float32))
    components = torch.from_numpy(pca.components_.astype(np.float32))

    def project(embs: torch.Tensor) -> torch.Tensor:
        centered = embs - mean
        projected = centered @ components.T
        return F.normalize(projected, p=2, dim=-1)

    projected_splits: dict[str, torch.Tensor] = {}
    for split_name in ("train", "validation", "test"):
        prefix = "val" if split_name == "validation" else split_name
        query_embs, doc_embs = full_teacher_targets.split(split_name)
        projected_splits[f"{prefix}_query"] = project(query_embs)
        projected_splits[f"{prefix}_doc"] = project(doc_embs)

    return DistillTargets(name=f"hpd_pca_{cfg.hpd_dim}", **projected_splits)


def make_method_targets(cfg: TrainConfig, full_teacher_targets: DistillTargets) -> dict[str, DistillTargets]:
    method_targets = {method_name: full_teacher_targets for method_name in METHOD_ORDER}
    method_targets["hpd"] = fit_hpd_targets(cfg=cfg, full_teacher_targets=full_teacher_targets)
    return method_targets


def _slice_batch_tensor(embs: torch.Tensor, batch_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return embs[batch_indices].to(device)


def _build_train_loader(
    name: str,
    split: RetrievalSplit,
    tokenizer: AutoTokenizer,
    student_model: StudentQueryEncoder,
    cfg: TrainConfig,
) -> Any:
    if name == TRAINED_BASELINE_NAME:
        return make_pair_dataloader(
            queries=split.queries,
            codes=split.codes,
            tokenizer=tokenizer,
            encoding_spec=student_model.encoding_spec,
            batch_size=cfg.batch_size,
            max_query_length=cfg.max_query_length,
            max_code_length=cfg.max_code_length,
            shuffle=True,
        )
    # KD methods: need indices (for teacher target lookup) + codes (to encode with student)
    return make_indexed_pair_dataloader(
        queries=split.queries,
        codes=split.codes,
        tokenizer=tokenizer,
        encoding_spec=student_model.encoding_spec,
        batch_size=cfg.batch_size,
        max_query_length=cfg.max_query_length,
        max_code_length=cfg.max_code_length,
        shuffle=True,
    )


def _compute_supervised_batch_losses(
    student_model: StudentQueryEncoder,
    batch: Any,
    cfg: TrainConfig,
    device: torch.device,
) -> LossComponents:
    tokenized_queries, tokenized_codes = batch
    tokenized_queries = to_device(tokenized_queries, device)
    tokenized_codes = to_device(tokenized_codes, device)

    student_q = student_model.encode(tokenized_queries)
    student_d = student_model.encode(tokenized_codes)
    one_hot = one_hot_loss(student_q @ student_d.T, cfg.temperature)
    return LossComponents.from_one_hot(one_hot=one_hot, device=device)


def _compute_kd_batch_losses(
    name: str,
    student_model: StudentQueryEncoder,
    batch: Any,
    targets: DistillTargets,
    full_teacher_targets: DistillTargets,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
    hard_neg_indices: torch.Tensor | None = None,
) -> LossComponents:
    batch_indices, tokenized_queries, tokenized_codes = batch
    tokenized_queries = to_device(tokenized_queries, device)
    tokenized_codes = to_device(tokenized_codes, device)

    student_q = student_model.encode(tokenized_queries)
    student_d = student_model.encode(tokenized_codes)
    train_target_q, train_target_d = targets.split("train")
    teacher_train_q, teacher_train_d = full_teacher_targets.split("train")

    target_q = _slice_batch_tensor(train_target_q, batch_indices, device)
    target_d = _slice_batch_tensor(train_target_d, batch_indices, device)
    teacher_q = _slice_batch_tensor(teacher_train_q, batch_indices, device)
    teacher_d = _slice_batch_tensor(teacher_train_d, batch_indices, device)

    # student_scores uses student-encoded docs so train/eval are consistent
    student_scores = student_q @ student_d.T
    teacher_scores = target_q @ target_d.T
    full_teacher_scores = teacher_q @ teacher_d.T

    name = name.removeprefix("phase2_")
    losses = LossComponents.from_one_hot(
        one_hot=one_hot_loss(student_scores, cfg.temperature),
        device=device,
    )
    dt = cfg.distill_temperature
    if name == "score_distill":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
    elif name == "embed_distill":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = align_loss(student_q, target_q)
    elif name == "qed_align":
        losses.align = align_loss(student_q, target_q)
    elif name == "distilcse_lite":
        losses.relation = contrastive_kd_loss(student_q, target_q, dt)
    elif name == "hard_negative_pair_distill":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.pairwise = pairwise_preference_loss(
            student_scores=student_scores,
            teacher_scores=teacher_scores,
            temperature=dt,
            hard_negatives=cfg.pair_hard_negatives,
        )
    elif name == "adam_lite":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.dark_kl, losses.dark_confidence = adam_dark_example_loss(
            student_query_emb=student_q,
            teacher_query_emb=teacher_q,
            teacher_doc_embs=teacher_d,
            teacher_scores=full_teacher_scores,
            temperature=dt,
            hard_negatives=cfg.dark_negatives,
            dark_mix_ratio=cfg.dark_mix_ratio,
        )
    elif name == "margin_mse":
        losses.pairwise = margin_mse_loss(student_scores, teacher_scores)
    elif name == "all_pairs_distill":
        losses.distill_kl = all_pairs_distill_loss(student_scores, teacher_scores)
    elif name == "pointwise":
        losses.distill_kl = pointwise_loss(student_scores, teacher_scores)
    elif name == "hpd":
        losses.align = align_loss(student_q, target_q)
    elif name == "bimga":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_loss(
            student_query_emb=student_q,
            student_doc_emb=student_d,
            target_query_emb=target_q,
            target_doc_emb=target_d,
            teacher_scores=teacher_scores,
            temperature=dt,
        )
    elif name == "bimga_uniform":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_uniform_loss(
            student_query_emb=student_q,
            student_doc_emb=student_d,
            target_query_emb=target_q,
            target_doc_emb=target_d,
        )
    elif name == "bimga_query_only":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_query_only_loss(
            student_query_emb=student_q,
            target_query_emb=target_q,
            teacher_scores=teacher_scores,
            temperature=dt,
        )
    elif name == "bimga_progressive":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_progressive_loss(
            student_query_emb=student_q,
            student_doc_emb=student_d,
            target_query_emb=target_q,
            target_doc_emb=target_d,
            teacher_scores=teacher_scores,
            base_temperature=dt,
            epoch=epoch,
            total_epochs=total_epochs,
        )
    elif name == "bimga_hardneg":
        # Expand score matrix with teacher-mined hard negatives
        if hard_neg_indices is not None:
            hn_idx = hard_neg_indices[batch_indices]  # (B, K)
            hn_docs = teacher_train_d[hn_idx.flatten()].to(device).reshape(
                batch_indices.size(0), hn_idx.size(1), -1
            )
            student_hn_scores = torch.einsum("bd,bkd->bk", student_q, hn_docs)
            teacher_hn_scores = torch.einsum("bd,bkd->bk", target_q, hn_docs)
            aug_student = torch.cat([student_scores, student_hn_scores], dim=1)
            aug_teacher = torch.cat([teacher_scores, teacher_hn_scores], dim=1)
            losses.distill_kl = distill_kl(aug_student, aug_teacher, dt)
        else:
            losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_loss(
            student_query_emb=student_q,
            student_doc_emb=student_d,
            target_query_emb=target_q,
            target_doc_emb=target_d,
            teacher_scores=teacher_scores,
            temperature=dt,
        )
    elif name == "bimga_attn_pool":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_loss(
            student_query_emb=student_q,
            student_doc_emb=student_d,
            target_query_emb=target_q,
            target_doc_emb=target_d,
            teacher_scores=teacher_scores,
            temperature=dt,
        )
    elif name == "bimga_bridge":
        losses.distill_kl = distill_kl(student_scores, teacher_scores, dt)
        losses.align = bimga_bridge_loss(
            student_query_emb=student_q,
            student_doc_emb=student_d,
            target_query_emb=target_q,
            target_doc_emb=target_d,
            teacher_scores=teacher_scores,
            temperature=dt,
        )
    else:
        raise ValueError(f"Unknown method: {name}")
    return losses


def _evaluate_split(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    split: RetrievalSplit,
    fixed_doc_embs: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    return evaluate_student_mode(
        eval_mode=cfg.eval_mode,
        student_model=student_model,
        tokenizer=tokenizer,
        queries=split.queries,
        codes=split.codes,
        fixed_doc_embs=fixed_doc_embs,
        max_query_length=cfg.max_query_length,
        max_code_length=cfg.max_code_length,
        eval_batch_size=cfg.eval_batch_size,
        device=device,
    )


def _evaluate_all_splits(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    data: RetrievalSplits,
    targets: DistillTargets,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split_name, split in data.items():
        _, fixed_doc_embs = targets.split(split_name)
        metrics[split_name] = _evaluate_split(
            student_model=student_model,
            tokenizer=tokenizer,
            split=split,
            fixed_doc_embs=fixed_doc_embs,
            cfg=cfg,
            device=device,
        )
    return metrics


def _build_diagnostics(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    data: RetrievalSplits,
    targets: DistillTargets,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    symmetric_validation = evaluate_symmetric_student(
        student_model=student_model,
        tokenizer=tokenizer,
        queries=data.validation.queries,
        codes=data.validation.codes,
        max_query_length=cfg.max_query_length,
        max_code_length=cfg.max_code_length,
        eval_batch_size=cfg.eval_batch_size,
        device=device,
    )
    symmetric_test = evaluate_symmetric_student(
        student_model=student_model,
        tokenizer=tokenizer,
        queries=data.test.queries,
        codes=data.test.codes,
        max_query_length=cfg.max_query_length,
        max_code_length=cfg.max_code_length,
        eval_batch_size=cfg.eval_batch_size,
        device=device,
    )

    teacher_space_compatible = student_model.output_hidden_size == targets.hidden_size
    diagnostics: dict[str, Any] = {
        "symmetric_validation": symmetric_validation,
        "symmetric_test": symmetric_test,
        "teacher_space_compatible": teacher_space_compatible,
    }
    if not teacher_space_compatible:
        return diagnostics

    diagnostics["asymmetric_validation"] = evaluate_asymmetric(
        student_model=student_model,
        tokenizer=tokenizer,
        queries=data.validation.queries,
        fixed_doc_embs=targets.val_doc,
        max_query_length=cfg.max_query_length,
        eval_batch_size=cfg.eval_batch_size,
        device=device,
    )
    diagnostics["asymmetric_test"] = evaluate_asymmetric(
        student_model=student_model,
        tokenizer=tokenizer,
        queries=data.test.queries,
        fixed_doc_embs=targets.test_doc,
        max_query_length=cfg.max_query_length,
        eval_batch_size=cfg.eval_batch_size,
        device=device,
    )

    query_alignment: dict[str, float] = {}
    for split_name, split in data.items():
        target_query_embs, _ = targets.split(split_name)
        query_alignment[split_name] = query_alignment_cosine(
            student_model=student_model,
            tokenizer=tokenizer,
            queries=split.queries,
            target_query_embs=target_query_embs,
            max_query_length=cfg.max_query_length,
            eval_batch_size=cfg.eval_batch_size,
            device=device,
        )

    diagnostics["query_alignment_cosine"] = query_alignment
    diagnostics["doc_alignment_cosine_test_student_vs_target"] = doc_alignment_cosine_student_vs_target(
        student_model=student_model,
        tokenizer=tokenizer,
        codes=data.test.codes,
        target_doc_embs=targets.test_doc,
        max_code_length=cfg.max_code_length,
        eval_batch_size=cfg.eval_batch_size,
        device=device,
    )
    diagnostics["symmetric_test_minus_asymmetric_test_mrr"] = (
        symmetric_test["MRR"] - diagnostics["asymmetric_test"]["MRR"]
    )
    return diagnostics


def train_student(
    name: str,
    cfg: TrainConfig,
    run_dir: Path,
    device: torch.device,
    data: RetrievalSplits,
    targets: DistillTargets,
    full_teacher_targets: DistillTargets,
    model_name: str | None = None,
    supervised: bool | None = None,
    initial_backbone_state_dict: dict[str, torch.Tensor] | None = None,
    tb_writer: SummaryWriter | None = None,
    hard_neg_indices: torch.Tensor | None = None,
) -> tuple[dict[str, Any], StudentQueryEncoder, AutoTokenizer]:
    effective_model = model_name or cfg.student_model
    is_supervised = supervised if supervised is not None else (name == TRAINED_BASELINE_NAME)
    use_attn_pool = "attn_pool" in name
    student_tokenizer = AutoTokenizer.from_pretrained(effective_model)
    student_model = StudentQueryEncoder(
        model_name=effective_model,
        target_hidden_size=None if is_supervised else targets.hidden_size,
        use_attention_pool=use_attn_pool,
    ).to(device)
    if initial_backbone_state_dict is not None:
        student_model.backbone.load_state_dict(initial_backbone_state_dict)
    init_stats = initialize_projection_from_targets(
        cfg=cfg,
        student_model=student_model,
        tokenizer=student_tokenizer,
        train_queries=data.train.queries,
        train_codes=data.train.codes,
        targets=targets,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loader_name = TRAINED_BASELINE_NAME if is_supervised else name
    train_loader = _build_train_loader(
        name=loader_name,
        split=data.train,
        tokenizer=student_tokenizer,
        student_model=student_model,
        cfg=cfg,
    )

    best_val_mrr = -math.inf
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    no_improve = 0
    stopped_epoch = cfg.epochs

    for epoch in range(1, cfg.epochs + 1):
        student_model.train()
        loss_sums = {
            "loss": 0.0,
            "one_hot": 0.0,
            "distill_kl": 0.0,
            "align": 0.0,
            "pairwise": 0.0,
            "relation": 0.0,
            "dark_kl": 0.0,
            "dark_confidence": 0.0,
        }
        steps = 0

        pbar = tqdm(train_loader, desc=f"{name} epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in pbar:
            if is_supervised:
                losses = _compute_supervised_batch_losses(
                    student_model=student_model,
                    batch=batch,
                    cfg=cfg,
                    device=device,
                )
            else:
                losses = _compute_kd_batch_losses(
                    name=name,
                    student_model=student_model,
                    batch=batch,
                    targets=targets,
                    full_teacher_targets=full_teacher_targets,
                    cfg=cfg,
                    device=device,
                    epoch=epoch,
                    total_epochs=cfg.epochs,
                    hard_neg_indices=hard_neg_indices,
                )

            total_loss = losses.total(cfg)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            steps += 1
            loss_sums["loss"] += float(total_loss.item())
            for key, value in losses.scalar_items().items():
                loss_sums[key] += value
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", hot=f"{losses.one_hot.item():.4f}")

            if tb_writer is not None:
                global_step = (epoch - 1) * len(train_loader) + steps
                tb_writer.add_scalar(f"{name}/step/total_loss", total_loss.item(), global_step)
                tb_writer.add_scalar(f"{name}/step/one_hot", losses.one_hot.item(), global_step)
                if losses.distill_kl.item() != 0.0:
                    tb_writer.add_scalar(f"{name}/step/distill_kl", losses.distill_kl.item(), global_step)
                if losses.align.item() != 0.0:
                    tb_writer.add_scalar(f"{name}/step/align", losses.align.item(), global_step)
                if losses.pairwise.item() != 0.0:
                    tb_writer.add_scalar(f"{name}/step/pairwise", losses.pairwise.item(), global_step)
                if losses.relation.item() != 0.0:
                    tb_writer.add_scalar(f"{name}/step/relation", losses.relation.item(), global_step)
                if losses.dark_kl.item() != 0.0:
                    tb_writer.add_scalar(f"{name}/step/dark_kl", losses.dark_kl.item(), global_step)

        train_stats = {"epoch": float(epoch)}
        train_stats.update({key: value / max(steps, 1) for key, value in loss_sums.items()})
        maybe_empty_device_cache(device)

        val_metrics = _evaluate_split(
            student_model=student_model,
            tokenizer=student_tokenizer,
            split=data.validation,
            fixed_doc_embs=targets.val_doc,
            cfg=cfg,
            device=device,
        )
        train_stats.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(train_stats)

        if tb_writer is not None:
            for key, value in train_stats.items():
                if key == "epoch":
                    continue
                tb_writer.add_scalar(f"{name}/epoch/{key}", value, epoch)
            tb_writer.flush()

        if val_metrics["MRR"] > best_val_mrr:
            best_val_mrr = val_metrics["MRR"]
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in student_model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if cfg.early_stopping_patience > 0 and no_improve >= cfg.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch} (no val MRR improvement for {cfg.early_stopping_patience} epochs)")
                stopped_epoch = epoch
                break

    assert best_state_dict is not None
    student_model.load_state_dict(best_state_dict)
    student_model.eval()

    final_metrics = _evaluate_all_splits(
        student_model=student_model,
        tokenizer=student_tokenizer,
        data=data,
        targets=targets,
        cfg=cfg,
        device=device,
    )
    metrics: dict[str, Any] = {
        "model_name": effective_model,
        "target_space": "student_native" if is_supervised else targets.name,
        "evaluation_mode": cfg.eval_mode,
        "stopped_epoch": stopped_epoch,
        "train": final_metrics["train"],
        "validation": final_metrics["validation"],
        "test": final_metrics["test"],
    }
    if init_stats is not None:
        metrics["initialization"] = init_stats
    if cfg.run_diagnostics:
        metrics["diagnostics"] = _build_diagnostics(
            student_model=student_model,
            tokenizer=student_tokenizer,
            data=data,
            targets=targets,
            cfg=cfg,
            device=device,
        )

    exp_dir = run_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if cfg.save_models:
        model_dir = exp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        student_model.backbone.save_pretrained(model_dir / "backbone")
        student_tokenizer.save_pretrained(model_dir / "tokenizer")
        if not isinstance(student_model.proj, nn.Identity):
            torch.save(student_model.proj.state_dict(), model_dir / "projection.pt")

    return metrics, student_model, student_tokenizer
