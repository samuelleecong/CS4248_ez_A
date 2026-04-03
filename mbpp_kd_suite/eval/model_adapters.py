from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModel, AutoTokenizer

from mbpp_kd_suite.modeling import (
    StudentQueryEncoder,
    encode_student_texts,
    encode_texts_backbone,
    format_texts_for_role,
    infer_model_encoding_spec,
    pool_hidden_state,
)


class EncoderAdapter(ABC):
    def __init__(
        self,
        max_query_length: int,
        max_code_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.max_query_length = max_query_length
        self.max_code_length = max_code_length
        self.batch_size = batch_size
        self.device = device

    @property
    @abstractmethod
    def encoding_spec(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_codes(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class HFEncoderAdapter(EncoderAdapter):
    def __init__(
        self,
        model_name_or_path: str,
        max_query_length: int,
        max_code_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            max_query_length=max_query_length,
            max_code_length=max_code_length,
            batch_size=batch_size,
            device=device,
        )
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        self.projection = _maybe_load_projection_layer(model_name_or_path, self.model.config.hidden_size).to(device)
        self.projection.eval()
        self._encoding_spec = infer_model_encoding_spec(
            model_name_or_path,
            getattr(self.model.config, "_name_or_path", None),
            getattr(self.tokenizer, "name_or_path", None),
        )

    @property
    def encoding_spec(self) -> Any:
        return self._encoding_spec

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        return _encode_hf_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            projection=self.projection,
            texts=texts,
            text_role="query",
            encoding_spec=self._encoding_spec,
            max_length=self.max_query_length,
            batch_size=self.batch_size,
            device=self.device,
            desc="eval_queries",
        )

    def encode_codes(self, texts: list[str]) -> torch.Tensor:
        return _encode_hf_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            projection=self.projection,
            texts=texts,
            text_role="document",
            encoding_spec=self._encoding_spec,
            max_length=self.max_code_length,
            batch_size=self.batch_size,
            device=self.device,
            desc="eval_codes",
        )

    def metadata(self) -> dict[str, Any]:
        has_projection = not isinstance(self.projection, torch.nn.Identity)
        output_hidden_size = int(self.model.config.hidden_size)
        if has_projection and isinstance(self.projection, torch.nn.Linear):
            output_hidden_size = int(self.projection.out_features)
        return {
            "adapter_type": "hf",
            "model_name_or_path": self.model_name_or_path,
            "pooling": self._encoding_spec.pooling,
            "query_prefix": self._encoding_spec.query_prefix,
            "doc_prefix": self._encoding_spec.doc_prefix,
            "hidden_size": output_hidden_size,
            "backbone_hidden_size": int(self.model.config.hidden_size),
            "has_projection": has_projection,
        }


class SuiteStudentEncoderAdapter(EncoderAdapter):
    def __init__(
        self,
        checkpoint_root: str,
        max_query_length: int,
        max_code_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            max_query_length=max_query_length,
            max_code_length=max_code_length,
            batch_size=batch_size,
            device=device,
        )
        self.original_path = Path(checkpoint_root).expanduser().resolve()
        self.model_dir = _resolve_suite_model_dir(self.original_path)
        projection_state = None
        projection_path = self.model_dir / "projection.pt"
        target_hidden_size = None
        if projection_path.exists():
            projection_state = torch.load(projection_path, map_location="cpu")
            weight = projection_state.get("weight")
            if weight is None:
                raise ValueError(f"projection.pt at {projection_path} does not contain a 'weight' tensor")
            target_hidden_size = int(weight.shape[0])

        self.student_model = StudentQueryEncoder(str(self.model_dir / "backbone"), target_hidden_size=target_hidden_size).to(device)
        if projection_state is not None:
            self.student_model.proj.load_state_dict(projection_state)
        self.student_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "tokenizer")
        self._has_projection = projection_state is not None

    @property
    def encoding_spec(self) -> Any:
        return self.student_model.encoding_spec

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        return encode_student_texts(
            student_model=self.student_model,
            tokenizer=self.tokenizer,
            texts=texts,
            text_role="query",
            max_length=self.max_query_length,
            batch_size=self.batch_size,
            device=self.device,
            desc="eval_queries",
        )

    def encode_codes(self, texts: list[str]) -> torch.Tensor:
        return encode_student_texts(
            student_model=self.student_model,
            tokenizer=self.tokenizer,
            texts=texts,
            text_role="document",
            max_length=self.max_code_length,
            batch_size=self.batch_size,
            device=self.device,
            desc="eval_codes",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter_type": "suite_student_dir",
            "model_name_or_path": str(self.original_path),
            "resolved_model_dir": str(self.model_dir),
            "pooling": self.student_model.encoding_spec.pooling,
            "query_prefix": self.student_model.encoding_spec.query_prefix,
            "doc_prefix": self.student_model.encoding_spec.doc_prefix,
            "hidden_size": int(self.student_model.output_hidden_size),
            "has_projection": self._has_projection,
        }


def load_model_adapter(
    model_source: str,
    model_name_or_path: str,
    checkpoint_format: str | None,
    max_query_length: int,
    max_code_length: int,
    batch_size: int,
    device: torch.device,
) -> EncoderAdapter:
    normalized_source = model_source.strip().lower()
    normalized_format = (checkpoint_format or "auto").strip().lower()

    if normalized_source == "hf":
        return HFEncoderAdapter(
            model_name_or_path=model_name_or_path,
            max_query_length=max_query_length,
            max_code_length=max_code_length,
            batch_size=batch_size,
            device=device,
        )

    if normalized_source != "local":
        raise ValueError(f"Unsupported model source: {model_source}")

    path = Path(model_name_or_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {path}")

    if normalized_format in {"auto", "suite_student_dir"} and _looks_like_suite_model_dir(path):
        return SuiteStudentEncoderAdapter(
            checkpoint_root=str(path),
            max_query_length=max_query_length,
            max_code_length=max_code_length,
            batch_size=batch_size,
            device=device,
        )

    if normalized_format in {"auto", "hf_dir"} and _looks_like_hf_dir(path):
        return HFEncoderAdapter(
            model_name_or_path=str(path),
            max_query_length=max_query_length,
            max_code_length=max_code_length,
            batch_size=batch_size,
            device=device,
        )

    raise ValueError(
        f"Unable to resolve local checkpoint format for {path}. Expected an HF directory or a suite student checkpoint."
    )


def _resolve_suite_model_dir(path: Path) -> Path:
    if (path / "backbone").exists() and (path / "tokenizer").exists():
        return path
    nested = path / "model"
    if (nested / "backbone").exists() and (nested / "tokenizer").exists():
        return nested
    raise ValueError(f"Path does not look like a suite student checkpoint directory: {path}")


def _looks_like_suite_model_dir(path: Path) -> bool:
    try:
        _resolve_suite_model_dir(path)
        return True
    except ValueError:
        return False


def _looks_like_hf_dir(path: Path) -> bool:
    return (path / "config.json").exists()


def _encode_hf_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    projection: torch.nn.Module,
    texts: list[str],
    text_role: str,
    encoding_spec: Any,
    max_length: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    if isinstance(projection, torch.nn.Identity):
        return encode_texts_backbone(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            text_role=text_role,
            encoding_spec=encoding_spec,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            desc=desc,
        )

    model.eval()
    projection.eval()
    all_embs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
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
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = model(**tokens)
        pooled = pool_hidden_state(
            outputs.last_hidden_state,
            tokens["attention_mask"],
            encoding_spec.pooling,
        )
        projected = projection(pooled)
        all_embs.append(F.normalize(projected, p=2, dim=-1).cpu())
    return torch.cat(all_embs, dim=0)


def _maybe_load_projection_layer(model_name_or_path: str, backbone_hidden_size: int) -> torch.nn.Module:
    projection_path = _resolve_projection_path(model_name_or_path)
    if projection_path is None:
        return torch.nn.Identity()

    projection_state = torch.load(projection_path, map_location="cpu", weights_only=True)
    weight = projection_state.get("weight")
    if weight is None:
        raise ValueError(f"projection.pt at {projection_path} does not contain a 'weight' tensor")

    projection = torch.nn.Linear(backbone_hidden_size, int(weight.shape[0]), bias=False)
    projection.load_state_dict(projection_state)
    return projection


def _resolve_projection_path(model_name_or_path: str) -> Path | None:
    local_path = Path(model_name_or_path).expanduser()
    if local_path.exists():
        projection_path = local_path / "projection.pt"
        return projection_path if projection_path.exists() else None

    try:
        projection_file = hf_hub_download(repo_id=model_name_or_path, filename="projection.pt")
    except EntryNotFoundError:
        return None
    return Path(projection_file)
