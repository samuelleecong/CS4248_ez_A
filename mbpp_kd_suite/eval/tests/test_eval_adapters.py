from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from eval.data_adapters import get_dataset_adapter


class EvalAdaptersTest(unittest.TestCase):
    def test_mbpp_local_file_produces_nonempty_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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


if __name__ == "__main__":
    unittest.main()
