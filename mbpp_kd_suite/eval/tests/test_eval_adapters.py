from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eval.data_adapters import get_dataset_adapter
from eval.model_adapters import HFEncoderAdapter

_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class EvalAdaptersTest(unittest.TestCase):
    def test_mbpp_local_file_produces_nonempty_splits(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            path = Path(tmpdir) / "mbpp.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for idx in range(20):
                    handle.write(
                        json.dumps(
                            {
                                "task_id": idx,
                                "text": f"solve task {idx}",
                                "code": f"def fn_{idx}():\n    return {idx}\n",
                            }
                        )
                        + "\n"
                    )
            corpus = get_dataset_adapter("mbpp", path=str(path)).load()
            self.assertGreater(len(corpus.train), 0)
            self.assertGreater(len(corpus.validation), 0)
            self.assertGreater(len(corpus.test), 0)

    def test_codesearchnet_local_directory_loads_python_splits(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            root = Path(tmpdir)
            for split_name, folder_name in (("train", "train"), ("validation", "valid"), ("test", "test")):
                split_dir = root / folder_name
                split_dir.mkdir(parents=True, exist_ok=True)
                file_path = split_dir / "part-0.jsonl.gz"
                rows = [
                    {
                        "repo": "owner/repo",
                        "path": f"src/{split_name}.py",
                        "func_name": f"fn_{split_name}",
                        "docstring": f"do {split_name}",
                        "code": f"def fn_{split_name}():\n    return '{split_name}'\n",
                    }
                ]
                with gzip.open(file_path, "wt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            corpus = get_dataset_adapter("codesearchnet", path=str(root)).load()
            self.assertEqual(len(corpus.train), 1)
            self.assertEqual(len(corpus.validation), 1)
            self.assertEqual(len(corpus.test), 1)

    def test_codesearchnet_remote_style_schema_is_supported(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            root = Path(tmpdir)
            for split_name, folder_name in (("train", "train"), ("validation", "valid"), ("test", "test")):
                split_dir = root / folder_name
                split_dir.mkdir(parents=True, exist_ok=True)
                file_path = split_dir / "part-0.jsonl"
                rows = [
                    {
                        "repository_name": "owner/repo",
                        "func_path_in_repository": f"src/{split_name}.py",
                        "func_name": f"fn_{split_name}",
                        "func_documentation_string": f"do {split_name}",
                        "whole_func_string": f"def fn_{split_name}():\n    return '{split_name}'\n",
                        "func_code_url": f"https://example.com/{split_name}",
                    }
                ]
                with file_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            corpus = get_dataset_adapter("codesearchnet", path=str(root)).load()
            self.assertEqual(corpus.test[0].query, "do test")
            self.assertIn("return 'test'", corpus.test[0].code)

    def test_unsupported_dataset_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_dataset_adapter("unknown-dataset")

    def test_hf_adapter_applies_projection_pt_when_present(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            projection_path = Path(tmpdir) / "projection.pt"
            torch.save({"weight": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])}, projection_path)

            with (
                patch("eval.model_adapters.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()),
                patch("eval.model_adapters.AutoModel.from_pretrained", return_value=_FakeBackboneModel()),
                patch("eval.model_adapters.hf_hub_download", return_value=str(projection_path)),
            ):
                adapter = HFEncoderAdapter(
                    model_name_or_path="cs4248-nlp/mock-projected-student",
                    max_query_length=16,
                    max_code_length=16,
                    batch_size=4,
                    device=torch.device("cpu"),
                )

            encoded = adapter.encode_queries(["alpha beta gamma"])
            self.assertEqual(tuple(encoded.shape), (1, 2))
            self.assertTrue(torch.allclose(torch.linalg.norm(encoded, dim=-1), torch.ones(1), atol=1e-5))
            self.assertEqual(adapter.metadata()["hidden_size"], 2)
            self.assertTrue(adapter.metadata()["has_projection"])


class _FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def __call__(
        self,
        texts: list[str],
        max_length: int,
        truncation: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del max_length, truncation, padding, return_tensors
        rows = []
        for text in texts:
            words = text.split()
            rows.append([float(len(words)), float(len(text)), float(sum(ch in "aeiou" for ch in text.lower()))])
        input_ids = torch.tensor(rows, dtype=torch.float32)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeBackboneModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3, _name_or_path="fake-backbone")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(1))


if __name__ == "__main__":
    unittest.main()
