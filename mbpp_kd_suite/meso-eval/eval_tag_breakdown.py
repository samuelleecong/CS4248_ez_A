"""Evaluate saved models from a sweep run directory, broken down by TACO tags and difficulty.

Loads each saved student model, encodes the full TACO test set, and computes MRR
at the aggregate level and per tag / per difficulty level. Also evaluates the teacher
model as an upper-bound baseline.

Usage:
    uv run python eval_tag_breakdown.py \
        --run-dir /path/to/paper_experiments_s1/20260410_234932 \
        --output eval_tag_results.json

    # Include teacher upper bound:
    uv run python eval_tag_breakdown.py \
        --run-dir /path/to/paper_experiments_s1/20260410_234932 \
        --teacher-model sentence-transformers/all-MiniLM-L6-v2 \
        --output eval_tag_results.json

    # Dry run — just print discovered models:
    uv run python eval_tag_breakdown.py \
        --run-dir /path/to/paper_experiments_s1/20260410_234932 \
        --dry-run
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


# ── Pooling ───────────────────────────────────────────────────────────────────

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: torch.device,
    max_length: int = 160,
) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ── MRR computation ───────────────────────────────────────────────────────────

def compute_mrr_per_example(scores: torch.Tensor) -> list[float]:
    """Given a (N, N) similarity matrix where diagonal = positive pair, return per-example MRR."""
    n = scores.size(0)
    reciprocal_ranks = []
    for i in range(n):
        row = scores[i]
        pos_score = row[i].item()
        # Rank = number of docs scoring >= pos_score (including itself)
        rank = (row >= pos_score).sum().item()
        reciprocal_ranks.append(1.0 / rank)
    return reciprocal_ranks


def aggregate_mrr(reciprocal_ranks: list[float]) -> dict:
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    sorted_rr = sorted(reciprocal_ranks, reverse=True)
    r_at_1 = sum(1 for rr in reciprocal_ranks if rr >= 1.0) / len(reciprocal_ranks)
    r_at_5 = sum(1 for rr in reciprocal_ranks if rr >= 1 / 5) / len(reciprocal_ranks)
    r_at_10 = sum(1 for rr in reciprocal_ranks if rr >= 1 / 10) / len(reciprocal_ranks)
    return {"MRR": mrr, "Recall@1": r_at_1, "Recall@5": r_at_5, "Recall@10": r_at_10, "n": len(reciprocal_ranks)}


def breakdown_by_field(
    reciprocal_ranks: list[float],
    field_values: list[list[str]],
    min_examples: int = 10,
) -> dict[str, dict]:
    """Group RR values by categorical field (tags or difficulty), return per-group MRR."""
    group_rrs: dict[str, list[float]] = defaultdict(list)
    for rr, values in zip(reciprocal_ranks, field_values):
        for v in values:
            group_rrs[v].append(rr)
    return {
        group: aggregate_mrr(rrs)
        for group, rrs in sorted(group_rrs.items())
        if len(rrs) >= min_examples
    }


# ── TACO data loading ─────────────────────────────────────────────────────────

def parse_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str) and value.strip():
        # skill_types is stored as a Python list string e.g. "['Sorting', 'Greedy algorithms']"
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(value)
                if isinstance(parsed, list):
                    return [v.strip() for v in parsed if isinstance(v, str) and v.strip()]
            except Exception:
                pass
        return [value.strip()]
    return []


def load_taco_test(dataset_name: str = "BEE-spoke-data/TACO-hf") -> tuple[
    list[str], list[str], list[list[str]], list[str]
]:
    """Load TACO test split, returning (queries, codes, skills_per_example, difficulties)."""
    print(f"Loading TACO test split from {dataset_name} ...")
    raw = load_dataset(dataset_name, split="test")

    queries, codes, skills_list, difficulties = [], [], [], []
    for row in raw:
        question = (row.get("question") or "").strip()
        if not question:
            continue

        raw_sol = row.get("solutions", [])
        if isinstance(raw_sol, str):
            try:
                raw_sol = json.loads(raw_sol)
            except Exception:
                raw_sol = []
        solutions = [s.strip() for s in raw_sol if isinstance(s, str) and s.strip()]
        if not solutions:
            continue

        starter = (row.get("starter_code") or "").strip()
        query = f"{question}\n\nStarter code:\n{starter}" if starter else question

        skills = parse_str_list(row.get("skill_types", []))
        if not skills:
            skills = ["(untagged)"]

        difficulty = (row.get("difficulty") or "UNKNOWN").strip()

        queries.append(query)
        codes.append(solutions[0])
        skills_list.append(skills)
        difficulties.append(difficulty)

    print(f"  Loaded {len(queries)} test examples")
    return queries, codes, skills_list, difficulties


# ── Model discovery ───────────────────────────────────────────────────────────

def find_saved_models(run_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return (run_name, backbone_dir, tokenizer_dir) for each saved model."""
    found = []
    for backbone_dir in sorted(run_dir.rglob("backbone")):
        if not (backbone_dir / "model.safetensors").exists():
            continue
        tokenizer_dir = backbone_dir.parent / "tokenizer"
        if not tokenizer_dir.exists():
            continue
        rel = backbone_dir.relative_to(run_dir)
        run_name = rel.parts[0]  # e.g. s1_bimga_dw100_aw10
        found.append((run_name, backbone_dir, tokenizer_dir))
    return found


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_teacher_margins(scores: torch.Tensor) -> list[float]:
    """Positive score minus hardest in-batch negative score, per example."""
    n = scores.size(0)
    pos_scores = scores.diag()
    mask = torch.eye(n, dtype=torch.bool, device=scores.device)
    neg_scores = scores.masked_fill(mask, -1e9)
    hardest_neg = neg_scores.max(dim=1).values
    return (pos_scores - hardest_neg).tolist()


def average_by_field(
    values: list[float],
    field_values: list[list[str]],
    min_examples: int = 1,
) -> dict[str, dict]:
    """Group scalar values by categorical field, return mean and n per group."""
    group_vals: dict[str, list[float]] = defaultdict(list)
    for v, groups in zip(values, field_values):
        for g in groups:
            group_vals[g].append(v)
    return {
        group: {"mean": sum(vs) / len(vs), "n": len(vs)}
        for group, vs in sorted(group_vals.items())
        if len(vs) >= min_examples
    }


def _compute_results(rrs: list[float], skills_list: list[list[str]], difficulties: list[str]) -> dict:
    return {
        "overall": aggregate_mrr(rrs),
        "by_skill": breakdown_by_field(rrs, skills_list, min_examples=1),
        "by_difficulty": breakdown_by_field(rrs, [[d] for d in difficulties], min_examples=1),
    }


def eval_model(
    backbone_dir: Path,
    tokenizer_dir: Path,
    queries: list[str],
    codes: list[str],
    skills_list: list[list[str]],
    difficulties: list[str],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    print(f"    Loading model from {backbone_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = AutoModel.from_pretrained(str(backbone_dir)).to(device).eval()

    print(f"    Encoding {len(queries)} queries ...")
    q_embs = encode_texts(queries, model, tokenizer, batch_size, device)
    print(f"    Encoding {len(codes)} documents ...")
    d_embs = encode_texts(codes, model, tokenizer, batch_size, device)

    scores = q_embs @ d_embs.T  # (N, N)
    rrs = compute_mrr_per_example(scores)
    return _compute_results(rrs, skills_list, difficulties)


def eval_hf_model(
    model_name: str,
    queries: list[str],
    codes: list[str],
    skills_list: list[list[str]],
    difficulties: list[str],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    print(f"    Loading {model_name} from HuggingFace ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    q_embs = encode_texts(queries, model, tokenizer, batch_size, device)
    d_embs = encode_texts(codes, model, tokenizer, batch_size, device)

    scores = q_embs @ d_embs.T
    rrs = compute_mrr_per_example(scores)
    margins = compute_teacher_margins(scores)

    result = _compute_results(rrs, skills_list, difficulties)
    result["margin_by_skill"] = average_by_field(margins, skills_list)
    result["margin_by_difficulty"] = average_by_field(margins, [[d] for d in difficulties])
    result["margin_overall"] = {"mean": sum(margins) / len(margins), "n": len(margins)}
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-skill / per-difficulty MRR breakdown")
    parser.add_argument("--run-dir", required=True, help="Sweep run directory (contains s1_* subdirs)")
    parser.add_argument("--output", default="eval_tag_results.json", help="Output JSON path")
    parser.add_argument("--teacher-model", default="", help="HuggingFace teacher model name for upper-bound eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset", default="BEE-spoke-data/TACO-hf")
    parser.add_argument("--dry-run", action="store_true", help="Print discovered models and exit")
    parser.add_argument("--runs", default="", help="Comma-separated run names to evaluate (default: all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}")
        return

    models = find_saved_models(run_dir)
    if not models:
        print(f"No saved models found in {run_dir}")
        return

    if args.dry_run:
        print(f"Found {len(models)} models:")
        for name, backbone, _ in models:
            print(f"  {name}  ->  {backbone}")
        return

    filter_runs = {r.strip() for r in args.runs.split(",") if r.strip()}
    if filter_runs:
        models = [(n, b, t) for n, b, t in models if n in filter_runs]
        print(f"Filtered to {len(models)} models: {[n for n, _, _ in models]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    queries, codes, skills_list, difficulties = load_taco_test(args.dataset)

    all_results: dict[str, dict] = {}

    # Evaluate teacher upper bound
    if args.teacher_model:
        print(f"\n[teacher] {args.teacher_model}")
        all_results["__teacher__"] = eval_hf_model(
            args.teacher_model, queries, codes, skills_list, difficulties, device, args.batch_size
        )
        t_mrr = all_results["__teacher__"]["overall"]["MRR"]
        print(f"    => MRR={t_mrr:.4f}")

    # Evaluate each saved model
    for run_name, backbone_dir, tokenizer_dir in models:
        print(f"\n[{run_name}]")
        try:
            result = eval_model(backbone_dir, tokenizer_dir, queries, codes, skills_list, difficulties, device, args.batch_size)
            all_results[run_name] = result
            print(f"    => MRR={result['overall']['MRR']:.4f}")
        except Exception as e:
            print(f"    => FAILED: {e}")
            all_results[run_name] = {"error": str(e)}

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Quick summary
    print(f"\n{'Run':<35} | {'MRR':>6} | {'R@1':>6} | {'R@10':>6}")
    print("-" * 60)
    for name, result in all_results.items():
        if "error" in result or "overall" not in result:
            continue
        o = result["overall"]
        print(f"{name:<35} | {o['MRR']:>6.4f} | {o['Recall@1']:>6.4f} | {o['Recall@10']:>6.4f}")


if __name__ == "__main__":
    main()
