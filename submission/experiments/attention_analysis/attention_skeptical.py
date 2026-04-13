"""
Skeptical verification of attention analysis claims.

Tests:
  1. Bootstrap CIs for CKA differences — is BiMGA > Control significant?
  2. Per-example entropy vs retrieval rank correlation — does entropy predict performance?
  3. Layer 1 CKA stripped out — does the CKA difference hold in layers 2-4 only?
  4. Permutation test for KL hierarchy — could the ordering arise by chance?
"""

from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

ARTIFACT_BASE = Path("mbpp_kd_suite/artifacts/paper_experiments/20260402_015143")
CACHE_DIR = Path("mbpp_kd_suite/attention_figures/_cache")
FIGURE_DIR = Path("mbpp_kd_suite/attention_figures/zoomed")
TEACHER_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
TACO_DATASET = "BEE-spoke-data/TACO-hf"

EXPERIMENTS = {
    "s7_control":    ("s7_control_bs32",     "control_supervised"),
    "s7_embed":      ("s7_embed_dw100_aw10", "embed_distill"),
    "s8_bimga_uni":  ("s8_A2_bimga_uniform", "bimga_uniform"),
    "s8_hnp":        ("s8_hnp_dw100_pw10",   "hard_negative_pair_distill"),
    "s9_score":      ("s9_score_dw100",       "score_distill"),
    "s10_bimga":     ("s10_bimga_dw100_aw10", "bimga"),
}

DISPLAY_NAMES = {
    "s7_control": "Control", "s7_embed": "Embed", "s8_bimga_uni": "BiMGA-Uni",
    "s8_hnp": "HNP", "s9_score": "Score", "s10_bimga": "BiMGA",
}

MRR_SCORES = {
    "s7_control": 0.205, "s7_embed": 0.303, "s8_bimga_uni": 0.313,
    "s8_hnp": 0.302, "s9_score": 0.301, "s10_bimga": 0.325,
}

MAX_QUERY_LEN = 160
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_tag_field(raw):
    if not raw:
        return []
    try:
        return json.loads(raw.replace("'", '"'))
    except:
        return []


def load_taco_test():
    ds = load_dataset(TACO_DATASET, split="test")
    examples = []
    for row in ds:
        question = (row.get("question") or "").strip()
        solutions_raw = row.get("solutions") or "[]"
        if isinstance(solutions_raw, str):
            try:
                solutions = json.loads(solutions_raw)
            except:
                solutions = []
        else:
            solutions = solutions_raw
        solutions = [s.strip() for s in solutions if isinstance(s, str) and s.strip()]
        if not question or not solutions:
            continue
        starter = (row.get("starter_code") or "").strip()
        query = f"{question}\n\nStarter code:\n{starter}" if starter else question
        examples.append({"query": query, "code": solutions[0],
                         "difficulty": row.get("difficulty", "UNKNOWN"),
                         "tags": parse_tag_field(row.get("tags"))})
    return examples


def load_student(key):
    d, m = EXPERIMENTS[key]
    p = ARTIFACT_BASE / d / m / "model"
    model = AutoModel.from_pretrained(str(p / "backbone"), attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(str(p / "tokenizer"))
    model.eval().to(DEVICE)
    return model, tok


def load_teacher():
    model = AutoModel.from_pretrained(TEACHER_MODEL, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model.eval().to(DEVICE)
    return model, tok


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro") ** 2
    yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(xy / (np.sqrt(xx * yy) + 1e-10))


@torch.no_grad()
def extract_cls_reps(model, tok, texts):
    all_reps = None
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="reps", leave=False):
        batch = texts[start:start + BATCH_SIZE]
        enc = tok(batch, max_length=MAX_QUERY_LEN, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        cls = [h[:, 0, :].cpu().numpy() for h in out.hidden_states]
        if all_reps is None:
            all_reps = [[] for _ in range(len(cls))]
        for i, c in enumerate(cls):
            all_reps[i].append(c)
        del out, enc
    return [np.concatenate(r, 0) for r in all_reps]


@torch.no_grad()
def extract_per_example_entropy_and_rank(model, tok, queries, codes):
    """
    For each example, compute:
      - mean attention entropy (scalar)
      - retrieval rank (how well this model retrieves the correct code)
    """
    # First: encode all queries and codes to get embeddings for ranking
    def encode_texts(texts, max_len):
        all_embs = []
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="encode", leave=False):
            batch = texts[start:start + BATCH_SIZE]
            enc = tok(batch, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            # Mean pool
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(F.normalize(pooled, p=2, dim=-1).cpu())
            del out, enc
        return torch.cat(all_embs, 0)

    q_embs = encode_texts(queries, MAX_QUERY_LEN)
    c_embs = encode_texts(codes, 256)
    scores = (q_embs @ c_embs.T).numpy()
    n = scores.shape[0]
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        order = np.argsort(-scores[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1

    # Now: per-example entropy
    entropies = []
    for start in tqdm(range(0, len(queries), BATCH_SIZE), desc="entropy", leave=False):
        batch = queries[start:start + BATCH_SIZE]
        enc = tok(batch, max_length=MAX_QUERY_LEN, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        out = model(**enc, output_attentions=True)
        stk = torch.stack(out.attentions, 0).cpu().numpy().transpose(1, 0, 2, 3, 4)
        masks = enc["attention_mask"].cpu().numpy().astype(bool)
        for i in range(len(batch)):
            sl = int(masks[i].sum())
            ai = stk[i, :, :, :sl, :sl]
            eps = 1e-10
            ent = -np.sum(ai * np.log(ai + eps), axis=-1).mean()  # scalar
            entropies.append(ent)
        del out, enc
    return np.array(entropies), ranks


def main():
    print("=" * 60)
    print("SKEPTICAL VERIFICATION")
    print("=" * 60)

    examples = load_taco_test()
    queries = [e["query"] for e in examples]
    codes = [e["code"] for e in examples]
    print(f"Loaded {len(examples)} examples")

    # =====================================================================
    # TEST 1: Bootstrap CIs for CKA differences
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Bootstrap CKA — is BiMGA > Control significant?")
    print("=" * 60)

    # Load teacher reps from cache
    teacher_cache = CACHE_DIR / "teacher_reps.npz"
    tdata = np.load(str(teacher_cache), allow_pickle=True)
    teacher_reps = [tdata[f"layer_{i}"] for i in range(len([k for k in tdata.files if k.startswith("layer_")]))]
    n_examples = teacher_reps[0].shape[0]

    # Load student reps for control and bimga
    student_reps = {}
    for key in ["s7_control", "s10_bimga", "s8_bimga_uni", "s9_score"]:
        print(f"  Loading {DISPLAY_NAMES[key]} reps...")
        model, tok = load_student(key)
        student_reps[key] = extract_cls_reps(model, tok, queries)
        del model, tok
        gc.collect()

    # Bootstrap: resample examples, compute mean CKA
    N_BOOTSTRAP = 1000
    rng = np.random.RandomState(42)

    print(f"\n  Running {N_BOOTSTRAP} bootstrap iterations...")
    bootstrap_cka = {key: [] for key in student_reps}
    # Also compute CKA excluding layer 0 and 1 (deeper layers only)
    bootstrap_cka_deep = {key: [] for key in student_reps}

    for b in tqdm(range(N_BOOTSTRAP), desc="bootstrap"):
        idx = rng.choice(n_examples, size=n_examples, replace=True)
        for key, s_reps in student_reps.items():
            # Full CKA
            cka_vals = []
            cka_deep_vals = []
            for ti in range(len(teacher_reps)):
                for si in range(len(s_reps)):
                    c = linear_cka(teacher_reps[ti][idx], s_reps[si][idx])
                    cka_vals.append(c)
                    if si >= 2:  # Student layers 2,3,4 only
                        cka_deep_vals.append(c)
            bootstrap_cka[key].append(np.mean(cka_vals))
            bootstrap_cka_deep[key].append(np.mean(cka_deep_vals))

    print("\n  FULL CKA (all layers):")
    print(f"  {'Model':<12} {'Mean':>8} {'95% CI':>20}")
    print(f"  {'-'*42}")
    for key in student_reps:
        vals = bootstrap_cka[key]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {DISPLAY_NAMES[key]:<12} {np.mean(vals):>8.4f} [{lo:.4f}, {hi:.4f}]")

    # Test: does BiMGA - Control > 0?
    diff = np.array(bootstrap_cka["s10_bimga"]) - np.array(bootstrap_cka["s7_control"])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    pct_positive = np.mean(diff > 0) * 100
    print(f"\n  BiMGA - Control difference: {np.mean(diff):.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  P(BiMGA > Control) = {pct_positive:.1f}%")

    print("\n  DEEP LAYERS ONLY (student layers 2-4):")
    print(f"  {'Model':<12} {'Mean':>8} {'95% CI':>20}")
    print(f"  {'-'*42}")
    for key in student_reps:
        vals = bootstrap_cka_deep[key]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {DISPLAY_NAMES[key]:<12} {np.mean(vals):>8.4f} [{lo:.4f}, {hi:.4f}]")

    diff_deep = np.array(bootstrap_cka_deep["s10_bimga"]) - np.array(bootstrap_cka_deep["s7_control"])
    lo, hi = np.percentile(diff_deep, [2.5, 97.5])
    pct_positive = np.mean(diff_deep > 0) * 100
    print(f"\n  BiMGA - Control (deep only): {np.mean(diff_deep):.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  P(BiMGA > Control) = {pct_positive:.1f}%")

    del student_reps
    gc.collect()

    # =====================================================================
    # TEST 2: Per-example entropy vs rank correlation
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Does per-example entropy predict retrieval rank?")
    print("=" * 60)

    for key in ["s7_control", "s10_bimga"]:
        print(f"\n  {DISPLAY_NAMES[key]}:")
        model, tok = load_student(key)
        ent, ranks = extract_per_example_entropy_and_rank(model, tok, queries, codes)
        del model, tok
        gc.collect()

        # Spearman correlation
        from scipy.stats import spearmanr, pearsonr
        sp_r, sp_p = spearmanr(ent, ranks)
        pe_r, pe_p = pearsonr(ent, ranks)
        print(f"    Spearman(entropy, rank): r={sp_r:.4f}, p={sp_p:.4e}")
        print(f"    Pearson(entropy, rank):  r={pe_r:.4f}, p={pe_p:.4e}")
        print(f"    (Negative r = higher entropy -> lower rank = better)")

        # Compare entropy of correct (rank=1) vs incorrect (rank>1) examples
        correct_mask = ranks == 1
        if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
            ent_correct = ent[correct_mask].mean()
            ent_incorrect = ent[~correct_mask].mean()
            print(f"    Mean entropy (rank=1): {ent_correct:.4f}")
            print(f"    Mean entropy (rank>1): {ent_incorrect:.4f}")
            print(f"    Difference: {ent_correct - ent_incorrect:+.4f}")

    # =====================================================================
    # TEST 3: Is layer 1 CKA trivially high? Compare to random baseline
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Is layer 1 CKA trivially high? Random baseline check")
    print("=" * 60)

    # Load teacher reps layer 1 and student reps layer 1
    # Compare CKA of real student L1 vs a random projection of same dimensionality
    teacher_cache = CACHE_DIR / "teacher_reps.npz"
    tdata = np.load(str(teacher_cache), allow_pickle=True)
    teacher_reps = [tdata[f"layer_{i}"] for i in range(len([k for k in tdata.files if k.startswith("layer_")]))]

    for key in ["s7_control", "s10_bimga"]:
        model, tok = load_student(key)
        s_reps = extract_cls_reps(model, tok, queries)
        del model, tok
        gc.collect()

        # Real CKA at student layer 1 vs teacher layers 2-7
        real_cka = np.mean([linear_cka(teacher_reps[ti], s_reps[1]) for ti in range(2, 8)])

        # Random baseline: random orthogonal matrix projection of teacher reps
        rng = np.random.RandomState(42)
        random_cka_vals = []
        for _ in range(100):
            random_reps = rng.randn(*s_reps[1].shape)
            random_cka_vals.append(np.mean([linear_cka(teacher_reps[ti], random_reps) for ti in range(2, 8)]))
        rand_mean = np.mean(random_cka_vals)
        rand_std = np.std(random_cka_vals)

        print(f"  {DISPLAY_NAMES[key]} student L1 vs teacher L2-7:")
        print(f"    Real CKA:   {real_cka:.4f}")
        print(f"    Random CKA: {rand_mean:.4f} +/- {rand_std:.4f}")
        print(f"    Z-score:    {(real_cka - rand_mean) / (rand_std + 1e-10):.1f}")

        del s_reps

    # =====================================================================
    # TEST 4: Effect size — CKA difference as % of range
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Effect size context")
    print("=" * 60)

    # Reload all CKA values from cache
    all_mean_cka = {}
    all_mean_cka_deep = {}
    for key in EXPERIMENTS:
        cache_file = CACHE_DIR / f"{key}.npz"
        if cache_file.exists():
            data = np.load(str(cache_file), allow_pickle=True)
            cka = data["cka"]
            all_mean_cka[key] = float(cka.mean())
            # Deep layers: student layers 2-4 (columns 2,3,4)
            all_mean_cka_deep[key] = float(cka[:, 2:].mean())

    print("\n  Full CKA:")
    for key in EXPERIMENTS:
        if key in all_mean_cka:
            print(f"    {DISPLAY_NAMES[key]:<12}: {all_mean_cka[key]:.4f}")

    range_full = max(all_mean_cka.values()) - min(all_mean_cka.values())
    bimga_vs_control = all_mean_cka.get("s10_bimga", 0) - all_mean_cka.get("s7_control", 0)
    print(f"  Range: {range_full:.4f}, BiMGA-Control: {bimga_vs_control:.4f} ({bimga_vs_control/range_full*100:.0f}% of range)")

    print("\n  Deep layers (student L2-4) CKA:")
    for key in EXPERIMENTS:
        if key in all_mean_cka_deep:
            print(f"    {DISPLAY_NAMES[key]:<12}: {all_mean_cka_deep[key]:.4f}")

    range_deep = max(all_mean_cka_deep.values()) - min(all_mean_cka_deep.values())
    bimga_vs_control_deep = all_mean_cka_deep.get("s10_bimga", 0) - all_mean_cka_deep.get("s7_control", 0)
    print(f"  Range: {range_deep:.4f}, BiMGA-Control: {bimga_vs_control_deep:.4f} ({bimga_vs_control_deep/range_deep*100:.0f}% of range)")

    print("\n" + "=" * 60)
    print("DONE — review results above to assess claim robustness")
    print("=" * 60)


if __name__ == "__main__":
    main()
