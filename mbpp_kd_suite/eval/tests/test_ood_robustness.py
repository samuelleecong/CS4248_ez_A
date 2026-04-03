from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.ood_data import load_mbpp_ood_corpus, load_taco_retrieval_corpus
from eval.ood_robustness import WorkflowConfig, run_workflow
from eval.perturbations import perturb_queries

_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class PerturbationTest(unittest.TestCase):
    def test_clean_tier_preserves_queries(self) -> None:
        queries = ["Write a function to count prime numbers in a list."]
        self.assertEqual(perturb_queries(queries, tier="clean", seed=7), queries)

    def test_same_seed_produces_identical_perturbations(self) -> None:
        queries = ["Return the largest odd number in the array"]
        first = perturb_queries(queries, tier="mixed_heavy", seed=11)
        second = perturb_queries(queries, tier="mixed_heavy", seed=11)
        self.assertEqual(first, second)

    def test_heavier_noise_changes_more_characters(self) -> None:
        queries = ["Create a Python function that sums even values from a nested list."]
        light = perturb_queries(queries, tier="typo_light", seed=5)[0]
        heavy = perturb_queries(queries, tier="typo_heavy", seed=5)[0]
        source = queries[0]
        light_delta = _changed_word_count(source, light)
        heavy_delta = _changed_word_count(source, heavy)
        self.assertGreaterEqual(heavy_delta, light_delta)


class OODDataTest(unittest.TestCase):
    def test_mbpp_ood_split_is_deterministic_and_persistable(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            mbpp_path = Path(tmpdir) / "mbpp.jsonl"
            with mbpp_path.open("w", encoding="utf-8") as handle:
                for idx in range(30):
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

            corpus_a, ids_a = load_mbpp_ood_corpus(str(mbpp_path), split_seed=17)
            corpus_b, ids_b = load_mbpp_ood_corpus(str(mbpp_path), split_seed=17)

            self.assertEqual(ids_a, ids_b)
            self.assertEqual([record.id for record in corpus_a.test], [record.id for record in corpus_b.test])
            self.assertGreater(len(ids_a["test"]), 0)

    def test_local_taco_split_dir_produces_nonempty_pairs(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            root = Path(tmpdir)
            payload = {
                "question": "Return the sum of all digits in n",
                "starter_code": "",
                "solutions": json.dumps(["def solve(n):\n    return sum(map(int, str(n)))\n"]),
            }
            for split_name, filename in (
                ("train", "train.jsonl"),
                ("validation", "validation.jsonl"),
                ("test", "test.jsonl"),
            ):
                path = root / filename
                with path.open("w", encoding="utf-8") as handle:
                    for idx in range(3):
                        row = dict(payload)
                        row["question"] = f"{payload['question']} {split_name} {idx}"
                        handle.write(json.dumps(row) + "\n")

            corpus = load_taco_retrieval_corpus(
                "BEE-spoke-data/TACO-hf",
                dataset_path=str(root),
                split_seed=3,
                split="test",
            )
            self.assertGreater(len(corpus.train), 0)
            self.assertGreater(len(corpus.validation), 0)
            self.assertGreater(len(corpus.test), 0)


class WorkflowSmokeTest(unittest.TestCase):
    def test_workflow_writes_expected_artifacts_with_mock_adapter(self) -> None:
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmpdir:
            root = Path(tmpdir)
            mbpp_path = root / "mbpp.jsonl"
            taco_dir = root / "taco"
            taco_dir.mkdir(parents=True, exist_ok=True)

            with mbpp_path.open("w", encoding="utf-8") as handle:
                for idx in range(25):
                    handle.write(
                        json.dumps(
                            {
                                "task_id": idx,
                                "text": f"mbpp prompt {idx}",
                                "code": f"def mbpp_{idx}():\n    return {idx}\n",
                            }
                        )
                        + "\n"
                    )

            for split_name, filename in (
                ("train", "train.jsonl"),
                ("validation", "validation.jsonl"),
                ("test", "test.jsonl"),
            ):
                path = taco_dir / filename
                with path.open("w", encoding="utf-8") as handle:
                    for idx in range(6):
                        handle.write(
                            json.dumps(
                                {
                                    "question": f"{split_name} question {idx}",
                                    "starter_code": "",
                                    "solutions": json.dumps([f"def fn_{split_name}_{idx}():\n    return {idx}\n"]),
                                }
                            )
                            + "\n"
                        )

            cfg = WorkflowConfig(
                models=("org/mock-model",),
                task="all",
                mbpp_dataset_path=str(mbpp_path),
                taco_dataset_name="BEE-spoke-data/TACO-hf",
                taco_dataset_path=str(taco_dir),
                split="test",
                split_seed=9,
                perturbation_tier="mixed_light",
                ks=(1, 5, 10),
                max_query_length=64,
                max_code_length=64,
                batch_size=8,
                device="cpu",
                output_dir=str(root / "runs"),
            )

            with patch("eval.ood_robustness.HFEncoderAdapter", new=_FakeHFEncoderAdapter):
                run_dir = run_workflow(cfg)

            metrics_path = run_dir / "metrics.csv"
            per_query_path = run_dir / "per_query_results.csv"
            summary_path = run_dir / "summary.json"
            selected_ids_path = run_dir / "selected_ids.json"

            self.assertTrue(metrics_path.exists())
            self.assertTrue(per_query_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(selected_ids_path.exists())

            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["task"] == "mbpp_ood" for row in rows))
            self.assertTrue(any(row["task"] == "taco_robustness" for row in rows))
            self.assertTrue(any(row["perturbation_tier"] == "clean" for row in rows))
            self.assertTrue(any(row["perturbation_tier"] == "mixed_light" for row in rows))

            taco_clean = next(row for row in rows if row["task"] == "taco_robustness" and row["perturbation_tier"] == "clean")
            taco_noisy = next(row for row in rows if row["task"] == "taco_robustness" and row["perturbation_tier"] == "mixed_light")
            self.assertEqual(float(taco_clean["delta_mrr_vs_clean"]), 0.0)
            self.assertLessEqual(float(taco_noisy["delta_mrr_vs_clean"]), 0.0)


class _FakeHFEncoderAdapter:
    def __init__(
        self,
        model_name_or_path: str,
        max_query_length: int,
        max_code_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        del model_name_or_path, max_query_length, max_code_length, batch_size, device

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([_vectorize(text) for text in texts], dtype=torch.float32)

    def encode_codes(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([_vectorize(text) for text in texts], dtype=torch.float32)


def _vectorize(text: str) -> np.ndarray:
    lowered = text.lower()
    ascii_sum = sum(ord(char) for char in lowered) % 997
    vowel_count = sum(char in "aeiou" for char in lowered)
    digit_count = sum(char.isdigit() for char in lowered)
    length = len(lowered)
    vec = np.array([ascii_sum, vowel_count, digit_count, length], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _changed_word_count(left: str, right: str) -> int:
    left_words = left.split()
    right_words = right.split()
    max_len = max(len(left_words), len(right_words))
    left_words += [""] * (max_len - len(left_words))
    right_words += [""] * (max_len - len(right_words))
    return sum(word_left != word_right for word_left, word_right in zip(left_words, right_words))


if __name__ == "__main__":
    unittest.main()
