"""Model merging and cluster-based routing experiments.

Explores two ways to combine trained KD students:
1. Model merging: average weights of different students
2. Cluster-based routing: route queries to the best student per cluster

Uses 30-epoch models from Set 1.

Usage:
    cd mbpp_kd_suite
    .venv/Scripts/python.exe run_merge_and_route.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

from mbpp_kd_suite.config import DistillTargets, TrainConfig
from mbpp_kd_suite.data import dataset_dict_to_splits, load_retrieval_dataset
from mbpp_kd_suite.metrics import (
    evaluate_symmetric_student,
    paired_ranking_metrics,
    score_metrics_from_embeddings,
)
from mbpp_kd_suite.modeling import StudentQueryEncoder, encode_student_texts
from mbpp_kd_suite.runtime import pick_device, set_seed


# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = "artifacts/two_phase_tinybert4l_taco_dt4/20260328_040837/phase1/checkpoint.pt"
RESULTS_DIR = Path("artifacts/paper_experiments/20260402_015143")
STUDENT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
TEACHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = "BEE-spoke-data/TACO-hf"
MAX_QUERY_LENGTH = 160
MAX_CODE_LENGTH = 256
EVAL_BATCH_SIZE = 64

# Models to use (30-epoch, best configs)
MODELS = {
    "bimga": "s1_bimga_dw50_aw10",
    "embed_distill": "s1_embed_dw100_aw10",
    "hard_neg_pair": "s1_hnp_dw100_pw10",
    "score_distill": "s1_score_dw100",
    "bimga_uniform": "s3_A2_bimga_uniform",
}


def load_student_model(run_name: str, device: torch.device) -> tuple[StudentQueryEncoder, AutoTokenizer]:
    """Load a trained student from its saved artifacts."""
    # Find the model directory (run_name/method_name/model/)
    run_dir = RESULTS_DIR / run_name
    model_dirs = list(run_dir.glob("*/model/backbone"))
    if not model_dirs:
        raise FileNotFoundError(f"No model found in {run_dir}")
    backbone_dir = model_dirs[0]
    model_root = backbone_dir.parent
    tokenizer_dir = model_root / "tokenizer"
    proj_path = model_root / "projection.pt"

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    # Need target_hidden_size to create projection layer
    target_hidden_size = 384  # MiniLM-L6-v2
    model = StudentQueryEncoder(
        model_name=STUDENT_MODEL,
        target_hidden_size=target_hidden_size,
    )
    # Load backbone
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(str(backbone_dir))
    model.backbone = backbone
    # Load projection
    if proj_path.exists():
        model.proj.load_state_dict(torch.load(proj_path, map_location="cpu", weights_only=True))

    model = model.to(device).eval()
    return model, tokenizer


def evaluate_model(
    model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    queries: list[str],
    codes: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model in symmetric mode."""
    return evaluate_symmetric_student(
        student_model=model,
        tokenizer=tokenizer,
        queries=queries,
        codes=codes,
        max_query_length=MAX_QUERY_LENGTH,
        max_code_length=MAX_CODE_LENGTH,
        eval_batch_size=EVAL_BATCH_SIZE,
        device=device,
    )


def get_embeddings(
    model: StudentQueryEncoder,
    tokenizer: AutoTokenizer,
    queries: list[str],
    codes: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get query and code embeddings from a model."""
    query_embs = encode_student_texts(
        student_model=model, tokenizer=tokenizer,
        texts=queries, text_role="query",
        max_length=MAX_QUERY_LENGTH, batch_size=EVAL_BATCH_SIZE,
        device=device, desc="encode_q",
    )
    code_embs = encode_student_texts(
        student_model=model, tokenizer=tokenizer,
        texts=codes, text_role="document",
        max_length=MAX_CODE_LENGTH, batch_size=EVAL_BATCH_SIZE,
        device=device, desc="encode_d",
    )
    return query_embs, code_embs


# ═══════════════════════════════════════════════════════════════════════════
# MODEL MERGING
# ═══════════════════════════════════════════════════════════════════════════

def merge_models(
    model_a: StudentQueryEncoder,
    model_b: StudentQueryEncoder,
    alpha: float,
    device: torch.device,
) -> StudentQueryEncoder:
    """Create a merged model: (1-alpha)*A + alpha*B."""
    merged = StudentQueryEncoder(
        model_name=STUDENT_MODEL,
        target_hidden_size=384,
    ).to(device)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged_state = {}
    for key in state_a:
        merged_state[key] = (1 - alpha) * state_a[key] + alpha * state_b[key]
    merged.load_state_dict(merged_state)
    merged.eval()
    return merged


def run_merging_experiments(
    models: dict[str, tuple[StudentQueryEncoder, AutoTokenizer]],
    test_queries: list[str],
    test_codes: list[str],
    device: torch.device,
) -> list[dict]:
    """Try merging different model pairs at different ratios."""
    results = []

    # Pairs to merge
    pairs = [
        ("bimga", "embed_distill"),
        ("bimga", "hard_neg_pair"),
        ("bimga", "score_distill"),
        ("bimga_uniform", "embed_distill"),
        ("embed_distill", "hard_neg_pair"),
    ]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    tokenizer = list(models.values())[0][1]  # All share same tokenizer

    for name_a, name_b in pairs:
        model_a = models[name_a][0]
        model_b = models[name_b][0]

        for alpha in alphas:
            label = f"{name_a}({1-alpha:.2f})+{name_b}({alpha:.2f})"
            print(f"  Merging: {label}")

            merged = merge_models(model_a, model_b, alpha, device)
            metrics = evaluate_model(merged, tokenizer, test_queries, test_codes, device)

            results.append({
                "model_a": name_a,
                "model_b": name_b,
                "alpha": alpha,
                "label": label,
                "MRR": metrics["MRR"],
                "Recall@1": metrics["Recall@1"],
                "Recall@10": metrics["Recall@10"],
            })
            print(f"    MRR={metrics['MRR']:.4f}")

            del merged
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTER-BASED ROUTING
# ═══════════════════════════════════════════════════════════════════════════

def run_routing_experiments(
    models: dict[str, tuple[StudentQueryEncoder, AutoTokenizer]],
    teacher_targets: DistillTargets,
    test_queries: list[str],
    test_codes: list[str],
    val_queries: list[str],
    val_codes: list[str],
    device: torch.device,
) -> dict:
    """Cluster queries and route to best student per cluster."""
    results = {}
    tokenizer = list(models.values())[0][1]

    # Get teacher query embeddings for clustering
    _, teacher_test_d = teacher_targets.split("test")
    _, teacher_val_d = teacher_targets.split("validation")
    teacher_test_q, _ = teacher_targets.split("test")
    teacher_val_q, _ = teacher_targets.split("validation")

    # Get per-query scores for each model on validation set
    print("\n  Encoding validation set with each model...")
    val_model_scores: dict[str, np.ndarray] = {}  # model -> (N,N) score matrix
    val_model_q_embs: dict[str, torch.Tensor] = {}
    val_model_d_embs: dict[str, torch.Tensor] = {}
    for name, (model, tok) in models.items():
        q_embs, d_embs = get_embeddings(model, tok, val_queries, val_codes, device)
        val_model_q_embs[name] = q_embs
        val_model_d_embs[name] = d_embs
        val_model_scores[name] = (q_embs @ d_embs.T).numpy()

    print("  Encoding test set with each model...")
    test_model_scores: dict[str, np.ndarray] = {}
    test_model_q_embs: dict[str, torch.Tensor] = {}
    test_model_d_embs: dict[str, torch.Tensor] = {}
    for name, (model, tok) in models.items():
        q_embs, d_embs = get_embeddings(model, tok, test_queries, test_codes, device)
        test_model_q_embs[name] = q_embs
        test_model_d_embs[name] = d_embs
        test_model_scores[name] = (q_embs @ d_embs.T).numpy()

    # Per-query reciprocal rank for each model
    def per_query_rr(scores: np.ndarray) -> np.ndarray:
        """Reciprocal rank for each query."""
        n = scores.shape[0]
        rr = np.zeros(n)
        for i in range(n):
            order = np.argsort(-scores[i])
            rank = int(np.where(order == i)[0][0]) + 1
            rr[i] = 1.0 / rank
        return rr

    val_rr = {name: per_query_rr(scores) for name, scores in val_model_scores.items()}
    test_rr = {name: per_query_rr(scores) for name, scores in test_model_scores.items()}

    # Try different k values
    for k in [4, 8, 12, 16]:
        print(f"\n  Routing with k={k} clusters...")

        # Cluster validation queries using teacher embeddings
        val_q_np = teacher_val_q.numpy()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        val_clusters = kmeans.fit_predict(val_q_np)

        # For each cluster, find best model by validation MRR
        routing_dict = {}
        for c in range(k):
            mask = val_clusters == c
            cluster_size = mask.sum()
            best_model = None
            best_mrr = -1
            for name, rr in val_rr.items():
                cluster_mrr = rr[mask].mean()
                if cluster_mrr > best_mrr:
                    best_mrr = cluster_mrr
                    best_model = name
            routing_dict[c] = best_model

        # Assign test queries to clusters
        test_q_np = teacher_test_q.numpy()
        test_clusters = kmeans.predict(test_q_np)

        # Hard routing: use best model's scores for each query
        n_test = len(test_queries)
        hard_routed_rr = np.zeros(n_test)
        for i in range(n_test):
            c = test_clusters[i]
            chosen_model = routing_dict[c]
            hard_routed_rr[i] = test_rr[chosen_model][i]

        hard_mrr = hard_routed_rr.mean()

        # Soft routing: weighted average of all models' scores per cluster
        # Weight = validation cluster MRR for that model
        soft_routed_rr = np.zeros(n_test)
        for i in range(n_test):
            c = test_clusters[i]
            mask = val_clusters == c
            weights = {}
            total_w = 0
            for name, rr in val_rr.items():
                w = rr[mask].mean()
                weights[name] = w
                total_w += w
            # Weighted combination of scores
            combined_scores = np.zeros(len(test_codes))
            for name, w in weights.items():
                combined_scores += (w / total_w) * test_model_scores[name][i]
            rank = int(np.where(np.argsort(-combined_scores) == i)[0][0]) + 1
            soft_routed_rr[i] = 1.0 / rank

        soft_mrr = soft_routed_rr.mean()

        # Oracle routing: pick the best model per query (upper bound)
        oracle_rr = np.max([rr for rr in test_rr.values()], axis=0)
        oracle_mrr = oracle_rr.mean()

        # Best single model
        single_best_name = max(test_rr, key=lambda n: test_rr[n].mean())
        single_best_mrr = test_rr[single_best_name].mean()

        cluster_info = {}
        for c in range(k):
            mask = val_clusters == c
            cluster_info[c] = {
                "size": int(mask.sum()),
                "best_model": routing_dict[c],
                "val_mrr": float(val_rr[routing_dict[c]][mask].mean()),
            }

        results[f"k={k}"] = {
            "k": k,
            "hard_routing_mrr": float(hard_mrr),
            "soft_routing_mrr": float(soft_mrr),
            "oracle_mrr": float(oracle_mrr),
            "single_best_model": single_best_name,
            "single_best_mrr": float(single_best_mrr),
            "routing_dict": {str(c): v for c, v in routing_dict.items()},
            "clusters": cluster_info,
        }

        print(f"    Single best ({single_best_name}): {single_best_mrr:.4f}")
        print(f"    Hard routing:  {hard_mrr:.4f} ({hard_mrr - single_best_mrr:+.4f})")
        print(f"    Soft routing:  {soft_mrr:.4f} ({soft_mrr - single_best_mrr:+.4f})")
        print(f"    Oracle:        {oracle_mrr:.4f} ({oracle_mrr - single_best_mrr:+.4f})")
        print(f"    Routing: {routing_dict}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    set_seed(42)
    device = pick_device()
    print(f"Device: {device}")

    # Load dataset
    dataset = load_retrieval_dataset(dataset_name=DATASET_NAME, taco_val_size=1000, seed=42)
    data = dataset_dict_to_splits(dataset)
    print(f"Splits: train={len(data.train.queries)}, val={len(data.validation.queries)}, test={len(data.test.queries)}")

    # Load teacher targets (for clustering)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    teacher_targets = DistillTargets(**ckpt["ft_teacher_targets"])

    # Load all models
    print("\nLoading models...")
    models: dict[str, tuple[StudentQueryEncoder, AutoTokenizer]] = {}
    for name, run_name in MODELS.items():
        print(f"  {name} <- {run_name}")
        model, tokenizer = load_student_model(run_name, device)
        models[name] = (model, tokenizer)

    # Baseline: evaluate each model individually
    print("\n" + "=" * 60)
    print("  INDIVIDUAL MODEL BASELINES")
    print("=" * 60)
    baselines = {}
    for name, (model, tok) in models.items():
        metrics = evaluate_model(model, tok, data.test.queries, data.test.codes, device)
        baselines[name] = metrics
        print(f"  {name:>20}: MRR={metrics['MRR']:.4f}  R@1={metrics['Recall@1']:.3f}  R@10={metrics['Recall@10']:.3f}")

    # Model merging
    print("\n" + "=" * 60)
    print("  MODEL MERGING")
    print("=" * 60)
    merge_results = run_merging_experiments(
        models, data.test.queries, data.test.codes, device
    )

    # Cluster-based routing
    print("\n" + "=" * 60)
    print("  CLUSTER-BASED ROUTING")
    print("=" * 60)
    route_results = run_routing_experiments(
        models, teacher_targets,
        data.test.queries, data.test.codes,
        data.validation.queries, data.validation.codes,
        device,
    )

    # Save all results
    output = {
        "baselines": baselines,
        "merging": merge_results,
        "routing": route_results,
    }
    out_path = RESULTS_DIR / "merge_and_route_results.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    best_single = max(baselines, key=lambda n: baselines[n]["MRR"])
    print(f"\n  Best single model: {best_single} (MRR={baselines[best_single]['MRR']:.4f})")

    best_merge = max(merge_results, key=lambda r: r["MRR"])
    print(f"  Best merge: {best_merge['label']} (MRR={best_merge['MRR']:.4f}, delta={best_merge['MRR'] - baselines[best_single]['MRR']:+.4f})")

    for k_str, r in route_results.items():
        print(f"  Routing {k_str}: hard={r['hard_routing_mrr']:.4f} ({r['hard_routing_mrr'] - baselines[best_single]['MRR']:+.4f})  "
              f"soft={r['soft_routing_mrr']:.4f} ({r['soft_routing_mrr'] - baselines[best_single]['MRR']:+.4f})  "
              f"oracle={r['oracle_mrr']:.4f}")


if __name__ == "__main__":
    main()
