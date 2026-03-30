from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .runtime import maybe_empty_device_cache


@dataclass(frozen=True)
class ModelEncodingSpec:
    pooling: str = "mean"
    query_prefix: str = ""
    doc_prefix: str = ""


def infer_model_encoding_spec(*name_hints: str | None) -> ModelEncodingSpec:
    joined = " ".join(hint for hint in name_hints if hint).lower()
    if "bge-" in joined and "reranker" not in joined:
        return ModelEncodingSpec(
            pooling="cls",
            query_prefix="Represent this sentence for searching relevant passages: ",
        )
    if "e5-" in joined:
        return ModelEncodingSpec(
            pooling="mean",
            query_prefix="query: ",
            doc_prefix="passage: ",
        )
    return ModelEncodingSpec()


def format_texts_for_role(
    texts: list[str],
    text_role: str,
    encoding_spec: ModelEncodingSpec,
) -> list[str]:
    if text_role == "query" and encoding_spec.query_prefix:
        return [f"{encoding_spec.query_prefix}{text}" for text in texts]
    if text_role == "document" and encoding_spec.doc_prefix:
        return [f"{encoding_spec.doc_prefix}{text}" for text in texts]
    return texts


class AttentionPooling(nn.Module):
    """Learned attention-weighted pooling over token embeddings."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = self.attention(last_hidden_state).squeeze(-1)  # (B, L)
        weights = weights.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(weights, dim=-1)  # (B, L)
        return (last_hidden_state * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)


class StudentQueryEncoder(nn.Module):
    def __init__(self, model_name: str, target_hidden_size: int | None = None, use_attention_pool: bool = False) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.encoding_spec = infer_model_encoding_spec(
            model_name,
            getattr(self.backbone.config, "_name_or_path", None),
        )
        student_hidden = int(self.backbone.config.hidden_size)
        self.output_hidden_size = student_hidden if target_hidden_size is None else int(target_hidden_size)
        self.proj = (
            nn.Identity()
            if student_hidden == self.output_hidden_size
            else nn.Linear(student_hidden, self.output_hidden_size, bias=False)
        )
        self.use_attention_pool = use_attention_pool
        if use_attention_pool:
            self.attn_pool = AttentionPooling(student_hidden)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_attention_pool:
            return self.attn_pool(last_hidden_state, attention_mask)
        return pool_hidden_state(last_hidden_state, attention_mask, self.encoding_spec.pooling)

    def encode(self, tokenized_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**tokenized_batch)
        pooled = self._pool(outputs.last_hidden_state, tokenized_batch["attention_mask"])
        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def pooled_backbone(self, tokenized_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**tokenized_batch)
        return self._pool(outputs.last_hidden_state, tokenized_batch["attention_mask"])


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def pool_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0]
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


@torch.no_grad()
def encode_texts_backbone(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    text_role: str,
    encoding_spec: ModelEncodingSpec,
    max_length: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    model.eval()
    all_embs: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = format_texts_for_role(
            texts[start : start + batch_size],
            text_role=text_role,
            encoding_spec=encoding_spec,
        )
        tokens = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = to_device(tokens, device)
        outputs = model(**tokens)
        pooled = pool_hidden_state(
            outputs.last_hidden_state,
            tokens["attention_mask"],
            encoding_spec.pooling,
        )
        all_embs.append(F.normalize(pooled, p=2, dim=-1).cpu())
        maybe_empty_device_cache(device)
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_student_texts(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    texts: list[str],
    text_role: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    student_model.eval()
    all_embs: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = format_texts_for_role(
            texts[start : start + batch_size],
            text_role=text_role,
            encoding_spec=student_model.encoding_spec,
        )
        tokens = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = to_device(tokens, device)
        all_embs.append(student_model.encode(tokens).detach().cpu())
        maybe_empty_device_cache(device)
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_student_backbone_texts(
    student_model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    texts: list[str],
    text_role: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    student_model.eval()
    all_embs: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = format_texts_for_role(
            texts[start : start + batch_size],
            text_role=text_role,
            encoding_spec=student_model.encoding_spec,
        )
        tokens = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = to_device(tokens, device)
        all_embs.append(student_model.pooled_backbone(tokens).detach().cpu())
        maybe_empty_device_cache(device)
    return torch.cat(all_embs, dim=0)
