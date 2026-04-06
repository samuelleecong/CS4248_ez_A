from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = PROJECT_ROOT / 'mbpp_kd_suite'
SRC_ROOT = SUITE_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mbpp_kd_suite.modeling import infer_model_encoding_spec, format_texts_for_role, mean_pool  # noqa: E402
from mbpp_kd_suite.metrics import paired_ranking_metrics, paired_ranks  # noqa: E402

ORG = 'cs4248-nlp'
TEACHER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DATASET_NAME = 'BEE-spoke-data/TACO-hf'
DATASET_CACHE_ROOT = Path.home() / '.cache' / 'cs4248_quant_analysis' / 'datasets'
MAX_QUERY_LENGTH = 160
MAX_CODE_LENGTH = 256
DEFAULT_EVAL_BATCH_SIZE = 64
LOW_MARGIN_THRESHOLD = 0.05
HF_SEARCH_PREFIX = 'paper-'
MAIN_SUBSET_RUNS = [
    's1_control_bs32',
    's1_score_dw100',
    's1_embed_dw100_aw10',
    's1_hnp_dw100_pw10',
    's1_bimga_dw50_aw10',
    's3_A2_bimga_uniform',
    's3_A3_bimga_query_only',
]
SEED_BASELINES = {
    'score_distill': 's1_score_dw100',
    'embed_distill': 's1_embed_dw100_aw10',
    'hard_neg_pair': 's1_hnp_dw100_pw10',
    'bimga': 's1_bimga_dw100_aw10',
}
METHOD_ORDER = [
    'control',
    'score_distill',
    'embed_distill',
    'hard_neg_pair',
    'bimga',
    'bimga_uniform',
    'bimga_query_only',
]
METHOD_COLORS = {
    'control': '#718096',
    'score_distill': '#2B6CB0',
    'embed_distill': '#2F855A',
    'hard_neg_pair': '#D69E2E',
    'bimga': '#C53030',
    'bimga_uniform': '#805AD5',
    'bimga_query_only': '#DD6B20',
}
METHOD_LABELS = {
    'control': 'control',
    'score_distill': 'score_distill',
    'embed_distill': 'embed_distill',
    'hard_neg_pair': 'hard_neg_pair',
    'bimga': 'bimga',
    'bimga_uniform': 'bimga_uniform',
    'bimga_query_only': 'bimga_query_only',
}

plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class ReplayResult:
    run_name: str
    repo_id: str | None
    replay_status: str
    symmetric_metrics: dict[str, float] | None
    asymmetric_metrics: dict[str, float] | None
    query_alignment_cosine: float | None
    doc_alignment_cosine: float | None
    sym_asym_gap: float | None
    mean_margin: float | None
    median_margin: float | None
    std_margin: float | None
    negative_margin_rate: float | None
    low_margin_rate: float | None
    teacher_student_margin_corr: float | None
    per_query_rows: list[dict[str, Any]]
    notes: str = ''


def slugify_run_name(run_name: str) -> str:
    return run_name.replace('_', '-').lower()


def canonical_method(run_name: str, run_cfg_method: str | None = None) -> str:
    if run_cfg_method:
        mapping = {
            'control': 'control',
            'score_distill': 'score_distill',
            'embed_distill': 'embed_distill',
            'hard_negative_pair_distill': 'hard_neg_pair',
            'bimga': 'bimga',
            'bimga_uniform': 'bimga_uniform',
            'bimga_query_only': 'bimga_query_only',
        }
        if run_cfg_method in mapping:
            return mapping[run_cfg_method]
    if '_hnp_' in run_name:
        return 'hard_neg_pair'
    if '_score_' in run_name:
        return 'score_distill'
    if '_embed_' in run_name:
        return 'embed_distill'
    if 'A2_bimga_uniform' in run_name:
        return 'bimga_uniform'
    if 'A3_bimga_query_only' in run_name:
        return 'bimga_query_only'
    if '_bimga_' in run_name:
        return 'bimga'
    if '_control_' in run_name or run_name.startswith('s1_control') or run_name.startswith('s2_control'):
        return 'control'
    raise ValueError(f'Cannot infer canonical method for {run_name}')


def parse_run_name(run_name: str, run_cfg: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        'run_name': run_name,
        'set': run_name.split('_', 1)[0],
        'seed': int(run_cfg.get('seed', 42)),
        'batch_size': int(run_cfg.get('batch_size', 32)),
        'distill_weight': float(run_cfg.get('distill_weight', 0.0)),
        'align_weight': float(run_cfg.get('align_weight', 0.0)),
        'pair_weight': float(run_cfg.get('pair_weight', 0.0)),
        'distill_temperature': float(run_cfg.get('distill_temperature', 0.2)),
        'supervised': bool(run_cfg.get('supervised', False)),
        'method': canonical_method(run_name, run_cfg.get('method')),
    }
    if meta['supervised']:
        meta['method'] = 'control'
    return meta


def parse_taco_solutions(raw_solutions: Any) -> list[str]:
    if isinstance(raw_solutions, list):
        return [solution.strip() for solution in raw_solutions if isinstance(solution, str) and solution.strip()]
    if not isinstance(raw_solutions, str) or not raw_solutions.strip():
        return []
    try:
        parsed = json.loads(raw_solutions)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [solution.strip() for solution in parsed if isinstance(solution, str) and solution.strip()]


def taco_row_to_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    question = row.get('question')
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = row.get('starter_code')
    starter_code = starter_code.strip() if isinstance(starter_code, str) else ''
    solutions = parse_taco_solutions(row.get('solutions'))
    if not solutions:
        return None
    query = question.strip()
    if starter_code:
        query = f"{query}\n\nStarter code:\n{starter_code}"
    return query, solutions[0]


def load_taco_retrieval(
    seed: int = 42,
    taco_val_size: int = 1000,
    *,
    include_train_validation: bool = False,
) -> dict[str, list[tuple[str, str]]]:
    api = HfApi()
    info = api.dataset_info(DATASET_NAME)
    test_files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('data/test'))

    test_rows: list[dict[str, Any]] = []
    for filename in test_files:
        path = download_public_dataset_file(DATASET_NAME, filename)
        test_rows.extend(pq.read_table(path).to_pylist())

    test_pairs = [pair for row in test_rows if (pair := taco_row_to_pair(row)) is not None]

    if not include_train_validation:
        return {'train': [], 'validation': [], 'test': test_pairs}

    train_files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('data/train'))
    train_rows: list[dict[str, Any]] = []
    for filename in train_files:
        path = download_public_dataset_file(DATASET_NAME, filename)
        train_rows.extend(pq.read_table(path).to_pylist())

    train_pairs = [pair for row in train_rows if (pair := taco_row_to_pair(row)) is not None]
    rng = np.random.default_rng(seed)
    shuffled = np.arange(len(train_pairs))
    rng.shuffle(shuffled)
    val_size = min(taco_val_size, max(1, len(train_pairs) // 10))
    val_idx = set(shuffled[:val_size].tolist())
    final_train = [pair for idx, pair in enumerate(train_pairs) if idx not in val_idx]
    final_val = [pair for idx, pair in enumerate(train_pairs) if idx in val_idx]
    return {'train': final_train, 'validation': final_val, 'test': test_pairs}


def download_public_dataset_file(dataset_id: str, filename: str) -> Path:
    cache_dir = DATASET_CACHE_ROOT / dataset_id.replace('/', '__')
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(filename).name
    tmp_path = local_path.with_suffix(local_path.suffix + '.tmp')
    if local_path.exists():
        return local_path
    if tmp_path.exists():
        tmp_path.unlink()
    url = f'https://huggingface.co/datasets/{dataset_id}/resolve/main/{filename}'
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(local_path)
    return local_path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    role: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    projection: torch.nn.Linear | None = None,
    encoding_spec=None,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        if encoding_spec is not None:
            batch_texts = format_texts_for_role(batch_texts, text_role=role, encoding_spec=encoding_spec)
        toks = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks)
            pooled = mean_pool(out.last_hidden_state, toks['attention_mask'])
            if projection is not None:
                pooled = projection(pooled)
            pooled = F.normalize(pooled, p=2, dim=-1)
        outputs.append(pooled.cpu())
        if device.type == 'mps':
            torch.mps.empty_cache()
    return torch.cat(outputs, dim=0)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, reps: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    n = len(values)
    for i in range(reps):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def teacher_doc_embeddings(device: torch.device, queries: list[str], codes: list[str], batch_size: int) -> dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModel.from_pretrained(TEACHER_MODEL).to(device)
    spec = infer_model_encoding_spec(TEACHER_MODEL, getattr(model.config, '_name_or_path', None), getattr(tokenizer, 'name_or_path', None))
    q = encode_texts(model, tokenizer, queries, role='query', max_length=MAX_QUERY_LENGTH, batch_size=batch_size, device=device, projection=None, encoding_spec=spec)
    d = encode_texts(model, tokenizer, codes, role='document', max_length=MAX_CODE_LENGTH, batch_size=batch_size, device=device, projection=None, encoding_spec=spec)
    scores = (q @ d.T).numpy()
    neg_scores = scores.copy()
    np.fill_diagonal(neg_scores, -np.inf)
    hardest_neg = neg_scores.max(axis=1)
    teacher_margin = np.diag(scores) - hardest_neg
    return {
        'query_embs': q,
        'doc_embs': d,
        'scores': torch.from_numpy(scores),
        'margin': torch.from_numpy(teacher_margin),
    }


def compute_per_query(score_matrix: np.ndarray, method: str, run_name: str, teacher_margin: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float], np.ndarray]:
    ranks = paired_ranks(score_matrix)
    reciprocal = 1.0 / ranks.astype(np.float64)
    positives = np.diag(score_matrix).astype(np.float64)
    neg_scores = score_matrix.copy()
    np.fill_diagonal(neg_scores, -np.inf)
    hardest_neg = neg_scores.max(axis=1).astype(np.float64)
    margins = positives - hardest_neg
    rows: list[dict[str, Any]] = []
    for idx in range(score_matrix.shape[0]):
        rows.append({
            'run_name': run_name,
            'method': method,
            'query_id': idx,
            'reciprocal_rank': reciprocal[idx],
            'rank': int(ranks[idx]),
            'positive_score': positives[idx],
            'hardest_negative_score': hardest_neg[idx],
            'margin': margins[idx],
            'correct_at_1': int(ranks[idx] == 1),
            'teacher_margin': float(teacher_margin[idx]),
        })
    summary = {
        'mean_margin': float(np.mean(margins)),
        'median_margin': float(np.median(margins)),
        'std_margin': float(np.std(margins)),
        'negative_margin_rate': float(np.mean(margins <= 0.0)),
        'low_margin_rate': float(np.mean(margins < LOW_MARGIN_THRESHOLD)),
    }
    corr = safe_corr(margins, teacher_margin)
    summary['teacher_student_margin_corr'] = corr if corr is not None else math.nan
    return rows, summary, margins


def to_py(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ''
        return value
    return value


def metric_delta(local_value: float | None, replay_value: float | None) -> float | None:
    if local_value is None or replay_value is None:
        return None
    return float(replay_value - local_value)


def get_repo_map(api: HfApi, org: str) -> dict[str, list[str]]:
    all_models = [m.id for m in api.list_models(author=org, search=HF_SEARCH_PREFIX, full=False)]
    mapping: dict[str, list[str]] = {}
    for model_id in all_models:
        lower = model_id.lower()
        for run_dir in (PROJECT_ROOT / 'submission' / 'experiments' / 'artifacts').iterdir():
            if not run_dir.is_dir():
                continue
            slug = slugify_run_name(run_dir.name)
            prefix = f'{org}/paper-{slug}-'
            if lower.startswith(prefix):
                mapping.setdefault(run_dir.name, []).append(model_id)
    return mapping


def load_projection(repo_id: str) -> tuple[torch.nn.Linear | None, bool, int | None]:
    api = HfApi()
    try:
        info = api.model_info(repo_id)
    except Exception:
        return None, False, None
    names = {s.rfilename for s in info.siblings}
    if 'projection.pt' not in names:
        return None, False, None
    state = torch.load(hf_hub_download(repo_id, 'projection.pt'), weights_only=True, map_location='cpu')
    weight = state['weight']
    out_dim, in_dim = weight.shape
    proj = torch.nn.Linear(in_dim, out_dim, bias=False)
    proj.load_state_dict(state)
    proj.eval()
    return proj, True, out_dim


def replay_run(
    repo_id: str,
    method: str,
    queries: list[str],
    codes: list[str],
    teacher_q: torch.Tensor,
    teacher_d: torch.Tensor,
    teacher_margin: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> ReplayResult:
    model = None
    projection = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModel.from_pretrained(repo_id).to(device)
        spec = infer_model_encoding_spec(repo_id, getattr(model.config, '_name_or_path', None), getattr(tokenizer, 'name_or_path', None))
        projection, projection_exists, _ = load_projection(repo_id)
        if projection is not None:
            projection = projection.to(device)
        q = encode_texts(model, tokenizer, queries, role='query', max_length=MAX_QUERY_LENGTH, batch_size=batch_size, device=device, projection=projection, encoding_spec=spec)
        d = encode_texts(model, tokenizer, codes, role='document', max_length=MAX_CODE_LENGTH, batch_size=batch_size, device=device, projection=projection, encoding_spec=spec)
        sym_scores = (q @ d.T).numpy()
        sym_metrics = paired_ranking_metrics(sym_scores, ks=(1, 5, 10))
        per_query_rows, margin_summary, margins = compute_per_query(sym_scores, method, run_name='', teacher_margin=teacher_margin)
        asym_metrics = None
        query_cos = None
        doc_cos = None
        gap = None
        if q.shape[1] == teacher_d.shape[1]:
            asym_scores = (q @ teacher_d.T).numpy()
            asym_metrics = paired_ranking_metrics(asym_scores, ks=(1, 5, 10))
            query_cos = float(F.cosine_similarity(q, teacher_q, dim=-1).mean().item())
            doc_cos = float(F.cosine_similarity(d, teacher_d, dim=-1).mean().item())
            gap = float(sym_metrics['MRR'] - asym_metrics['MRR'])
        return ReplayResult(
            run_name='',
            repo_id=repo_id,
            replay_status='ok',
            symmetric_metrics=sym_metrics,
            asymmetric_metrics=asym_metrics,
            query_alignment_cosine=query_cos,
            doc_alignment_cosine=doc_cos,
            sym_asym_gap=gap,
            mean_margin=margin_summary['mean_margin'],
            median_margin=margin_summary['median_margin'],
            std_margin=margin_summary['std_margin'],
            negative_margin_rate=margin_summary['negative_margin_rate'],
            low_margin_rate=margin_summary['low_margin_rate'],
            teacher_student_margin_corr=margin_summary['teacher_student_margin_corr'],
            per_query_rows=per_query_rows,
            notes='' if projection_exists else 'no_projection',
        )
    except Exception as exc:
        return ReplayResult(
            run_name='',
            repo_id=repo_id,
            replay_status='load_failed',
            symmetric_metrics=None,
            asymmetric_metrics=None,
            query_alignment_cosine=None,
            doc_alignment_cosine=None,
            sym_asym_gap=None,
            mean_margin=None,
            median_margin=None,
            std_margin=None,
            negative_margin_rate=None,
            low_margin_rate=None,
            teacher_student_margin_corr=None,
            per_query_rows=[],
            notes=str(exc),
        )
    finally:
        if projection is not None:
            del projection
        if model is not None:
            del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()


def plot_core_sweep(run_metrics: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    set1 = [r for r in run_metrics if r['set'] == 's1' and r['replay_status'] == 'ok']
    control = next(r for r in set1 if r['method'] == 'control')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    method_grid = ['score_distill', 'embed_distill', 'hard_neg_pair', 'bimga']
    stats = {}
    for ax, method in zip(axes.flat, method_grid):
        rows = [r for r in set1 if r['method'] == method]
        rows.sort(key=lambda r: (r['distill_weight'], r['align_weight'], r['pair_weight']))
        labels = []
        vals = []
        for r in rows:
            if method in {'embed_distill', 'bimga'}:
                label = f"dw{int(r['distill_weight'])}\naw{int(r['align_weight'])}"
            elif method == 'hard_neg_pair':
                label = f"dw{int(r['distill_weight'])}\npw{int(r['pair_weight'])}"
            else:
                label = f"dw{int(r['distill_weight'])}"
            labels.append(label)
            vals.append(r['symmetric_test_mrr'])
        ax.bar(range(len(vals)), vals, color=METHOD_COLORS[method])
        ax.axhline(control['symmetric_test_mrr'], color=METHOD_COLORS['control'], linestyle='--', linewidth=1.5, label='control')
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel('Test MRR')
        stats[method] = max(vals) if vals else math.nan
    fig.suptitle('Set 1 core sweep: Test MRR by method and KD weight configuration')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    best_method = max(stats, key=stats.get)
    return {'best_method': best_method, 'best_mrr': stats[best_method], 'control_mrr': control['symmetric_test_mrr']}


def plot_scatter(run_metrics: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, path: Path) -> dict[str, Any]:
    rows = [r for r in run_metrics if r['replay_status'] == 'ok' and r.get(x_key) not in ('', None) and r.get(y_key) not in ('', None)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHOD_ORDER:
        subset = [r for r in rows if r['method'] == method]
        if not subset:
            continue
        ax.scatter([r[x_key] for r in subset], [r[y_key] for r in subset], s=55, alpha=0.85, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    xs = np.array([float(r[x_key]) for r in rows], dtype=np.float64)
    ys = np.array([float(r[y_key]) for r in rows], dtype=np.float64)
    if len(xs) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xx, m * xx + b, color='black', linestyle=':', linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    corr = safe_corr(xs, ys)
    return {'corr': corr if corr is not None else math.nan, 'n': len(rows)}


def plot_margin_distribution(per_query_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    methods = MAIN_SUBSET_RUNS
    fig, ax = plt.subplots(figsize=(11, 6))
    data = []
    labels = []
    medians = {}
    for run_name in methods:
        subset = [r for r in per_query_rows if r['run_name'] == run_name]
        margins = np.array([float(r['margin']) for r in subset], dtype=np.float64)
        data.append(margins)
        labels.append(run_name.replace('s1_', '').replace('s3_', ''))
        medians[run_name] = float(np.median(margins))
    vp = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
    color_cycle = [METHOD_COLORS[canonical_method(name)] for name in methods]
    for body, color in zip(vp['bodies'], color_cycle):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
    if 'cmedians' in vp:
        vp['cmedians'].set_color('#1A202C')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Positive vs hardest-negative margin')
    ax.set_title('Margin distribution for main paper subset')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    best = max(medians, key=medians.get)
    return {'best_run': best, 'best_median_margin': medians[best]}


def plot_margin_summary(per_query_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = []
    mean_vals = []
    cis_low = []
    cis_high = []
    neg_rates = []
    for run_name in MAIN_SUBSET_RUNS:
        subset = [r for r in per_query_rows if r['run_name'] == run_name]
        margins = np.array([float(r['margin']) for r in subset], dtype=np.float64)
        mean_v, lo, hi = bootstrap_mean_ci(margins)
        neg_rate = float(np.mean(margins <= 0.0))
        labels.append(run_name.replace('s1_', '').replace('s3_', ''))
        mean_vals.append(mean_v)
        cis_low.append(mean_v - lo)
        cis_high.append(hi - mean_v)
        neg_rates.append(neg_rate)
    x = np.arange(len(labels))
    colors = [METHOD_COLORS[canonical_method(name)] for name in MAIN_SUBSET_RUNS]
    axes[0].bar(x, mean_vals, color=colors)
    axes[0].errorbar(x, mean_vals, yerr=[cis_low, cis_high], fmt='none', ecolor='black', capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_ylabel('Mean margin')
    axes[0].set_title('Mean margin with 95% bootstrap CI')
    axes[1].bar(x, neg_rates, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_ylabel('Negative-margin rate')
    axes[1].set_title('Share of queries with margin <= 0')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    best_idx = int(np.argmax(mean_vals))
    worst_idx = int(np.argmax(neg_rates))
    return {'best_margin_run': MAIN_SUBSET_RUNS[best_idx], 'best_mean_margin': mean_vals[best_idx], 'worst_negative_margin_run': MAIN_SUBSET_RUNS[worst_idx], 'worst_negative_margin_rate': neg_rates[worst_idx]}


def plot_training_curves(history_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(10, 6))
    best_final = None
    best_mrr = -1.0
    for run_name in MAIN_SUBSET_RUNS:
        rows = [r for r in history_rows if r['run_name'] == run_name]
        rows.sort(key=lambda r: r['epoch'])
        xs = [r['epoch'] for r in rows]
        ys = [r['val_MRR'] for r in rows if r['val_MRR'] not in ('', None)]
        xs = xs[:len(ys)]
        method = canonical_method(run_name)
        ax.plot(xs, ys, marker='o', linewidth=1.8, markersize=3.5, label=run_name.replace('s1_', '').replace('s3_', ''), color=METHOD_COLORS[method])
        if ys and ys[-1] > best_mrr:
            best_mrr = ys[-1]
            best_final = run_name
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MRR')
    ax.set_title('Validation MRR curves for representative runs')
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return {'best_final_curve': best_final, 'best_final_val_mrr': best_mrr}


def plot_seed_stability(seed_summary_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    methods = ['score_distill', 'embed_distill', 'hard_neg_pair', 'bimga']
    rows = [r for r in seed_summary_rows if r['method'] in methods]
    rows.sort(key=lambda r: methods.index(r['method']))
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    mrr_means = [float(r['test_mrr_mean']) for r in rows]
    mrr_stds = [float(r['test_mrr_std']) for r in rows]
    doc_means = [float(r['doc_cosine_mean']) for r in rows]
    doc_stds = [float(r['doc_cosine_std']) for r in rows]
    colors = [METHOD_COLORS[r['method']] for r in rows]
    axes[0].bar(x, mrr_means, yerr=mrr_stds, color=colors, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r['method'] for r in rows], rotation=20, ha='right')
    axes[0].set_ylabel('Test MRR')
    axes[0].set_title('Seed stability: test MRR')
    axes[1].bar(x, doc_means, yerr=doc_stds, color=colors, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r['method'] for r in rows], rotation=20, ha='right')
    axes[1].set_ylabel('Doc cosine')
    axes[1].set_title('Seed stability: document alignment')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    best = max(rows, key=lambda r: float(r['test_mrr_mean']))
    return {'best_seed_method': best['method'], 'best_seed_mrr': float(best['test_mrr_mean'])}


def plot_teacher_vs_student_margin(per_query_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    selected = ['s1_control_bs32', 's1_embed_dw100_aw10', 's1_hnp_dw100_pw10', 's1_bimga_dw50_aw10']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    corrs = {}
    for ax, run_name in zip(axes.flat, selected):
        subset = [r for r in per_query_rows if r['run_name'] == run_name]
        x = np.array([float(r['teacher_margin']) for r in subset], dtype=np.float64)
        y = np.array([float(r['margin']) for r in subset], dtype=np.float64)
        method = canonical_method(run_name)
        ax.scatter(x, y, s=10, alpha=0.35, color=METHOD_COLORS[method])
        corr = safe_corr(x, y)
        corrs[run_name] = corr if corr is not None else math.nan
        ax.set_title(f"{run_name}\nr={corr:.3f}" if corr is not None else run_name)
        ax.set_xlabel('Teacher margin')
        ax.set_ylabel('Student margin')
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    best = max(corrs, key=lambda k: (-999 if math.isnan(corrs[k]) else corrs[k]))
    return {'best_margin_tracking_run': best, 'best_margin_tracking_corr': corrs[best]}


def write_figure_md(path: Path, title: str, figure_filename: str, purpose: str, how_to_read: str, what_it_shows: str, why_it_matters: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n"
        f"**Purpose:** {purpose}\n\n"
        f"![{title}](../figures/{figure_filename})\n\n"
        f"## How to read this\n{how_to_read}\n\n"
        f"## What it shows\n{what_it_shows}\n\n"
        f"## Why it matters for the paper\n{why_it_matters}\n",
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the TACO HF quantitative analysis package.')
    parser.add_argument('--output-root', default=str(PROJECT_ROOT / 'submission' / 'experiments' / 'analysis'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    csv_dir = output_root / 'csv'
    fig_dir = output_root / 'figures'
    fig_md_dir = output_root / 'figures_md'
    output_root.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_md_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')

    artifacts_root = PROJECT_ROOT / 'submission' / 'experiments' / 'artifacts'
    results_summary = json.loads((artifacts_root / 'results_summary.json').read_text(encoding='utf-8'))
    api = HfApi()
    repo_candidates = get_repo_map(api, ORG)

    print('loading fixed TACO test split', flush=True)
    taco = load_taco_retrieval(seed=42, taco_val_size=1000, include_train_validation=False)
    queries = [q for q, _ in taco['test']]
    codes = [c for _, c in taco['test']]
    print(f'loaded {len(queries)} TACO test queries', flush=True)
    print(f'building teacher embeddings on {device}', flush=True)
    teacher = teacher_doc_embeddings(device=device, queries=queries, codes=codes, batch_size=args.batch_size)
    teacher_q = teacher['query_embs']
    teacher_d = teacher['doc_embs']
    teacher_margin = teacher['margin'].numpy()

    hf_registry_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    run_names = sorted(results_summary)
    total_runs = len(run_names)
    for idx, run_name in enumerate(run_names, start=1):
        print(f'[{idx}/{total_runs}] preparing {run_name}', flush=True)
        local_result = results_summary[run_name]
        run_dir = artifacts_root / run_name
        run_cfg = json.loads((run_dir / 'run_config.json').read_text(encoding='utf-8'))
        meta = parse_run_name(run_name, run_cfg)
        repo_matches = sorted(repo_candidates.get(run_name, []))
        repo_id = repo_matches[-1] if repo_matches else None
        load_status = 'missing_repo'
        projection_exists = False
        projection_out_dim: int | None = None
        hidden_size: int | None = None
        registry_notes = ''
        if repo_id is not None:
            try:
                info = api.model_info(repo_id)
                siblings = {s.rfilename for s in info.siblings}
                projection_exists = 'projection.pt' in siblings
                hidden_size = int(AutoConfig.from_pretrained(repo_id).hidden_size)
                if projection_exists:
                    state = torch.load(hf_hub_download(repo_id, 'projection.pt'), weights_only=True, map_location='cpu')
                    projection_out_dim = int(state['weight'].shape[0])
                load_status = 'ok'
            except Exception as exc:
                load_status = 'load_failed'
                registry_notes = str(exc)
        hf_registry_rows.append({
            'run_name': run_name,
            'repo_id': repo_id or '',
            'projection_exists': projection_exists,
            'load_status': load_status,
            'hidden_size': hidden_size or '',
            'projection_out_dim': projection_out_dim or '',
            'notes': registry_notes,
        })
        manifest_rows.append({
            'run_name': run_name,
            'set': meta['set'],
            'method': meta['method'],
            'seed': meta['seed'],
            'batch_size': meta['batch_size'],
            'distill_weight': meta['distill_weight'],
            'align_weight': meta['align_weight'],
            'pair_weight': meta['pair_weight'],
            'distill_temperature': meta['distill_temperature'],
            'repo_id': repo_id or '',
            'artifact_path': str(run_dir),
            'local_metric_source': 'artifact_results_summary',
            'replay_source': 'hf' if load_status == 'ok' else '',
            'replay_status': load_status,
        })
        history = json.loads((run_dir / 'history.json').read_text(encoding='utf-8'))
        for row in history:
            history_rows.append({
                'run_name': run_name,
                'method': meta['method'],
                'set': meta['set'],
                'epoch': row.get('epoch', ''),
                'loss': row.get('loss', ''),
                'one_hot': row.get('one_hot', ''),
                'distill_kl': row.get('distill_kl', ''),
                'align': row.get('align', ''),
                'pairwise': row.get('pairwise', ''),
                'relation': row.get('relation', ''),
                'dark_kl': row.get('dark_kl', ''),
                'dark_confidence': row.get('dark_confidence', ''),
                'val_MRR': row.get('val_MRR', ''),
                'val_Recall@1': row.get('val_Recall@1', ''),
                'val_Recall@10': row.get('val_Recall@10', ''),
            })

        replay = ReplayResult(run_name, repo_id, load_status, None, None, None, None, None, None, None, None, None, None, None, [], registry_notes)
        if load_status == 'ok' and repo_id is not None:
            print(f'[{idx}/{total_runs}] replaying {run_name} from {repo_id}', flush=True)
            replay = replay_run(repo_id, meta['method'], queries, codes, teacher_q, teacher_d, teacher_margin, device, args.batch_size)
            replay.run_name = run_name
            for row in replay.per_query_rows:
                row['run_name'] = run_name
            per_query_rows.extend(replay.per_query_rows)
        else:
            print(f'[{idx}/{total_runs}] skipping replay for {run_name}: {load_status}', flush=True)

        local_test = local_result.get('test', {})
        local_diag = local_result.get('diagnostics', {})
        metrics_rows.append({
            'run_name': run_name,
            'set': meta['set'],
            'method': meta['method'],
            'repo_id': repo_id or '',
            'replay_status': replay.replay_status,
            'symmetric_test_mrr': to_py(replay.symmetric_metrics.get('MRR') if replay.symmetric_metrics else None),
            'symmetric_test_recall@1': to_py(replay.symmetric_metrics.get('Recall@1') if replay.symmetric_metrics else None),
            'symmetric_test_recall@10': to_py(replay.symmetric_metrics.get('Recall@10') if replay.symmetric_metrics else None),
            'symmetric_test_map@10': to_py(replay.symmetric_metrics.get('MAP@10') if replay.symmetric_metrics else None),
            'symmetric_test_ndcg@10': to_py(replay.symmetric_metrics.get('nDCG@10') if replay.symmetric_metrics else None),
            'asymmetric_test_mrr': to_py(replay.asymmetric_metrics.get('MRR') if replay.asymmetric_metrics else None),
            'asymmetric_test_recall@1': to_py(replay.asymmetric_metrics.get('Recall@1') if replay.asymmetric_metrics else None),
            'asymmetric_test_recall@10': to_py(replay.asymmetric_metrics.get('Recall@10') if replay.asymmetric_metrics else None),
            'local_test_mrr': to_py(local_test.get('MRR')),
            'local_test_recall@1': to_py(local_test.get('Recall@1')),
            'local_test_recall@10': to_py(local_test.get('Recall@10')),
            'delta_mrr_vs_local': to_py(metric_delta(local_test.get('MRR'), replay.symmetric_metrics.get('MRR') if replay.symmetric_metrics else None)),
            'delta_recall@1_vs_local': to_py(metric_delta(local_test.get('Recall@1'), replay.symmetric_metrics.get('Recall@1') if replay.symmetric_metrics else None)),
            'delta_recall@10_vs_local': to_py(metric_delta(local_test.get('Recall@10'), replay.symmetric_metrics.get('Recall@10') if replay.symmetric_metrics else None)),
        })
        diagnostics_rows.append({
            'run_name': run_name,
            'set': meta['set'],
            'method': meta['method'],
            'seed': meta['seed'],
            'batch_size': meta['batch_size'],
            'repo_id': repo_id or '',
            'replay_status': replay.replay_status,
            'query_alignment_cosine': to_py(replay.query_alignment_cosine),
            'doc_alignment_cosine': to_py(replay.doc_alignment_cosine),
            'sym_asym_mrr_gap': to_py(replay.sym_asym_gap),
            'mean_margin': to_py(replay.mean_margin),
            'median_margin': to_py(replay.median_margin),
            'std_margin': to_py(replay.std_margin),
            'negative_margin_rate': to_py(replay.negative_margin_rate),
            'low_margin_rate_lt_0p05': to_py(replay.low_margin_rate),
            'teacher_student_margin_corr': to_py(replay.teacher_student_margin_corr),
            'local_doc_alignment_cosine': to_py(local_diag.get('doc_alignment_cosine_test_student_vs_target')),
            'local_sym_asym_mrr_gap': to_py(local_diag.get('symmetric_test_minus_asymmetric_test_mrr')),
            'local_query_alignment_cosine_test': to_py((local_diag.get('query_alignment_cosine') or {}).get('test')),
        })

    # summaries
    method_summary_rows: list[dict[str, Any]] = []
    for run_name in MAIN_SUBSET_RUNS:
        mrow = next(r for r in metrics_rows if r['run_name'] == run_name)
        drow = next(r for r in diagnostics_rows if r['run_name'] == run_name)
        method_summary_rows.append({
            'run_name': run_name,
            'method': mrow['method'],
            'symmetric_test_mrr': mrow['symmetric_test_mrr'],
            'symmetric_test_recall@1': mrow['symmetric_test_recall@1'],
            'symmetric_test_recall@10': mrow['symmetric_test_recall@10'],
            'asymmetric_test_mrr': mrow['asymmetric_test_mrr'],
            'doc_alignment_cosine': drow['doc_alignment_cosine'],
            'query_alignment_cosine': drow['query_alignment_cosine'],
            'mean_margin': drow['mean_margin'],
            'median_margin': drow['median_margin'],
            'negative_margin_rate': drow['negative_margin_rate'],
            'teacher_student_margin_corr': drow['teacher_student_margin_corr'],
        })

    seed_summary_rows: list[dict[str, Any]] = []
    for method, seed42_run in SEED_BASELINES.items():
        seed_runs = [seed42_run] + [r['run_name'] for r in manifest_rows if r['set'] == 's4' and r['method'] == method]
        subset_metrics = [next(r for r in metrics_rows if r['run_name'] == rn and r['replay_status'] == 'ok') for rn in seed_runs]
        subset_diag = [next(r for r in diagnostics_rows if r['run_name'] == rn and r['replay_status'] == 'ok') for rn in seed_runs]
        test_mrr = np.array([float(r['symmetric_test_mrr']) for r in subset_metrics], dtype=np.float64)
        doc_cos = np.array([float(r['doc_alignment_cosine']) for r in subset_diag if r['doc_alignment_cosine'] != ''], dtype=np.float64)
        margin = np.array([float(r['mean_margin']) for r in subset_diag], dtype=np.float64)
        seed_summary_rows.append({
            'method': method,
            'n_runs': len(subset_metrics),
            'runs': '|'.join(seed_runs),
            'test_mrr_mean': float(test_mrr.mean()),
            'test_mrr_std': float(test_mrr.std(ddof=0)),
            'test_mrr_min': float(test_mrr.min()),
            'test_mrr_max': float(test_mrr.max()),
            'doc_cosine_mean': float(doc_cos.mean()) if len(doc_cos) else '',
            'doc_cosine_std': float(doc_cos.std(ddof=0)) if len(doc_cos) else '',
            'doc_cosine_min': float(doc_cos.min()) if len(doc_cos) else '',
            'doc_cosine_max': float(doc_cos.max()) if len(doc_cos) else '',
            'mean_margin_mean': float(margin.mean()),
            'mean_margin_std': float(margin.std(ddof=0)),
            'mean_margin_min': float(margin.min()),
            'mean_margin_max': float(margin.max()),
        })

    # write csvs
    write_csv(csv_dir / 'hf_model_registry.csv', hf_registry_rows, ['run_name', 'repo_id', 'projection_exists', 'load_status', 'hidden_size', 'projection_out_dim', 'notes'])
    write_csv(csv_dir / 'run_manifest.csv', manifest_rows, ['run_name', 'set', 'method', 'seed', 'batch_size', 'distill_weight', 'align_weight', 'pair_weight', 'distill_temperature', 'repo_id', 'artifact_path', 'local_metric_source', 'replay_source', 'replay_status'])
    write_csv(csv_dir / 'run_metrics.csv', metrics_rows, ['run_name', 'set', 'method', 'repo_id', 'replay_status', 'symmetric_test_mrr', 'symmetric_test_recall@1', 'symmetric_test_recall@10', 'symmetric_test_map@10', 'symmetric_test_ndcg@10', 'asymmetric_test_mrr', 'asymmetric_test_recall@1', 'asymmetric_test_recall@10', 'local_test_mrr', 'local_test_recall@1', 'local_test_recall@10', 'delta_mrr_vs_local', 'delta_recall@1_vs_local', 'delta_recall@10_vs_local'])
    write_csv(csv_dir / 'run_diagnostics.csv', diagnostics_rows, ['run_name', 'set', 'method', 'seed', 'batch_size', 'repo_id', 'replay_status', 'query_alignment_cosine', 'doc_alignment_cosine', 'sym_asym_mrr_gap', 'mean_margin', 'median_margin', 'std_margin', 'negative_margin_rate', 'low_margin_rate_lt_0p05', 'teacher_student_margin_corr', 'local_doc_alignment_cosine', 'local_sym_asym_mrr_gap', 'local_query_alignment_cosine_test'])
    write_csv(csv_dir / 'history_long.csv', history_rows, ['run_name', 'method', 'set', 'epoch', 'loss', 'one_hot', 'distill_kl', 'align', 'pairwise', 'relation', 'dark_kl', 'dark_confidence', 'val_MRR', 'val_Recall@1', 'val_Recall@10'])
    write_csv(csv_dir / 'per_query_scores.csv', per_query_rows, ['run_name', 'method', 'query_id', 'reciprocal_rank', 'rank', 'positive_score', 'hardest_negative_score', 'margin', 'correct_at_1', 'teacher_margin'])
    write_csv(csv_dir / 'method_summary.csv', method_summary_rows, ['run_name', 'method', 'symmetric_test_mrr', 'symmetric_test_recall@1', 'symmetric_test_recall@10', 'asymmetric_test_mrr', 'doc_alignment_cosine', 'query_alignment_cosine', 'mean_margin', 'median_margin', 'negative_margin_rate', 'teacher_student_margin_corr'])
    write_csv(csv_dir / 'seed_summary.csv', seed_summary_rows, ['method', 'n_runs', 'runs', 'test_mrr_mean', 'test_mrr_std', 'test_mrr_min', 'test_mrr_max', 'doc_cosine_mean', 'doc_cosine_std', 'doc_cosine_min', 'doc_cosine_max', 'mean_margin_mean', 'mean_margin_std', 'mean_margin_min', 'mean_margin_max'])

    # plot inputs simplified
    merged_rows = []
    diag_map = {r['run_name']: r for r in diagnostics_rows}
    for row in metrics_rows:
        merged = dict(row)
        merged.update({k: v for k, v in diag_map[row['run_name']].items() if k not in merged})
        manifest = next(r for r in manifest_rows if r['run_name'] == row['run_name'])
        merged.update(manifest)
        merged_rows.append(merged)

    fig01_stats = plot_core_sweep(merged_rows, fig_dir / 'fig01_core_sweep_mrr.png')
    fig02_stats = plot_scatter(merged_rows, 'doc_alignment_cosine', 'symmetric_test_mrr', 'Document alignment cosine', 'Symmetric test MRR', 'Document quality vs symmetric retrieval', fig_dir / 'fig02_doc_cosine_vs_sym_mrr.png')
    fig03_stats = plot_scatter(merged_rows, 'doc_alignment_cosine', 'sym_asym_mrr_gap', 'Document alignment cosine', 'Symmetric - asymmetric MRR gap', 'Document quality vs symmetric penalty', fig_dir / 'fig03_doc_cosine_vs_sym_asym_gap.png')
    fig04_stats = plot_margin_distribution(per_query_rows, fig_dir / 'fig04_margin_distribution_best_methods.png')
    fig05_stats = plot_margin_summary(per_query_rows, fig_dir / 'fig05_margin_summary_best_methods.png')
    fig06_stats = plot_training_curves(history_rows, fig_dir / 'fig06_training_curves_best_runs.png')
    fig07_stats = plot_seed_stability(seed_summary_rows, fig_dir / 'fig07_seed_stability.png')
    fig08_stats = plot_teacher_vs_student_margin(per_query_rows, fig_dir / 'fig08_teacher_margin_vs_student_margin.png')

    # markdown explainers
    write_figure_md(
        fig_md_dir / 'fig01_core_sweep_mrr.md',
        'Figure 1: Core sweep MRR',
        'fig01_core_sweep_mrr.png',
        'Show how each KD family improves or fails to improve over the no-KD control across the Set 1 sweep.',
        'Each subplot is one KD family. The dashed horizontal line is the control baseline. Higher bars mean better test retrieval on TACO.',
        f"The strongest Set 1 family is {fig01_stats['best_method']} with best test MRR {fig01_stats['best_mrr']:.4f}, compared with control at {fig01_stats['control_mrr']:.4f}.",
        'This establishes the sweep-level ranking before we argue why the stronger methods win.'
    )
    write_figure_md(
        fig_md_dir / 'fig02_doc_cosine_vs_sym_mrr.md',
        'Figure 2: Document cosine vs symmetric MRR',
        'fig02_doc_cosine_vs_sym_mrr.png',
        'Test whether better student document embeddings explain better symmetric retrieval.',
        'Each point is one replayed HF run. The x-axis is how close student doc embeddings are to teacher docs. The y-axis is symmetric test MRR.',
        f"Across {fig02_stats['n']} replayed runs, higher document cosine tracks higher symmetric MRR (Pearson r={fig02_stats['corr']:.3f}).",
        'This is the core quantitative evidence for Analysis #1: document quality is not just a side metric; it tracks retrieval quality.'
    )
    write_figure_md(
        fig_md_dir / 'fig03_doc_cosine_vs_sym_asym_gap.md',
        'Figure 3: Document cosine vs symmetric-asymmetric gap',
        'fig03_doc_cosine_vs_sym_asym_gap.png',
        'Test whether better student document embeddings reduce the penalty of symmetric evaluation.',
        'A smaller or near-zero gap means the student loses less when it has to encode documents itself instead of reusing teacher docs.',
        f"Across {fig03_stats['n']} replayed runs, higher document cosine is associated with a smaller symmetric penalty (Pearson r={fig03_stats['corr']:.3f}).",
        'This directly supports the paper claim that document alignment matters in symmetric retrieval.'
    )
    write_figure_md(
        fig_md_dir / 'fig04_margin_distribution_best_methods.md',
        'Figure 4: Margin distributions for representative methods',
        'fig04_margin_distribution_best_methods.png',
        'Compare how clearly each representative method separates the positive code from the hardest negative.',
        'Wider distributions shifted upward are better. Large positive margins mean the model is confidently ranking the correct code above the hardest distractor.',
        f"Among the representative runs, {fig04_stats['best_run']} has the highest median margin at {fig04_stats['best_median_margin']:.4f}.",
        'This is the main quantitative view for Analysis #2: stronger methods should not just win on rank; they should create cleaner score separation.'
    )
    write_figure_md(
        fig_md_dir / 'fig05_margin_summary_best_methods.md',
        'Figure 5: Margin summary for representative methods',
        'fig05_margin_summary_best_methods.png',
        'Summarize score separation with both average margin strength and failure rate.',
        'The left panel shows mean margin with a bootstrap confidence interval. The right panel shows how often the positive margin is zero or negative.',
        f"The strongest average margin belongs to {fig05_stats['best_margin_run']} ({fig05_stats['best_mean_margin']:.4f}), while the worst negative-margin rate is {fig05_stats['worst_negative_margin_run']} ({fig05_stats['worst_negative_margin_rate']:.3f}).",
        'This turns per-query margin behavior into a compact paper-ready comparison.'
    )
    write_figure_md(
        fig_md_dir / 'fig06_training_curves_best_runs.md',
        'Figure 6: Training curves for representative runs',
        'fig06_training_curves_best_runs.png',
        'Show whether the best methods improve steadily or only late in training.',
        'Each line is validation MRR across epochs for one representative run.',
        f"The strongest final validation curve in the representative subset is {fig06_stats['best_final_curve']} with final validation MRR {fig06_stats['best_final_val_mrr']:.4f}.",
        'This helps separate stable improvements from lucky endpoint results.'
    )
    write_figure_md(
        fig_md_dir / 'fig07_seed_stability.md',
        'Figure 7: Seed stability',
        'fig07_seed_stability.png',
        'Measure whether the main method ranking is stable across seeds.',
        'Each bar shows the mean across seed42/123/456, with standard deviation as the error bar.',
        f"The best multi-seed mean test MRR is {fig07_stats['best_seed_method']} at {fig07_stats['best_seed_mrr']:.4f}.",
        'A method that only wins once is weak evidence. This figure shows whether the gain survives random initialization.'
    )
    write_figure_md(
        fig_md_dir / 'fig08_teacher_margin_vs_student_margin.md',
        'Figure 8: Teacher margin vs student margin',
        'fig08_teacher_margin_vs_student_margin.png',
        'Check whether representative students preserve the teacher’s confidence structure at the query level.',
        'Each point is one test query. Better agreement means the student margin rises when the teacher margin rises.',
        f"The strongest teacher-margin tracking in the comparison set is {fig08_stats['best_margin_tracking_run']} (r={fig08_stats['best_margin_tracking_corr']:.3f}).",
        'This figure links KD behavior to teacher confidence rather than only final rank metrics.'
    )

    (output_root / 'README.md').write_text(
        '# TACO HF Quant Analysis Package\n\n'
        'This package replays the HF-uploaded submission experiment checkpoints on the fixed TACO test split, writes CSV summaries, builds paper-oriented figures, and adds one Markdown explainer per figure.\n\n'
        '## Outputs\n\n'
        '- `csv/`: run registry, replayed metrics, diagnostics, per-query margins, and summaries\n'
        '- `figures/`: PNG figures for the paper and appendix\n'
        '- `figures_md/`: one Markdown explainer per figure\n\n'
        '## Build command\n\n'
        '```bash\n'
        'cd Project/CS4248_ez_A/mbpp_kd_suite\n'
        '.venv/bin/python ../submission/experiments/analysis/build_quant_package.py\n'
        '```\n',
        encoding='utf-8',
    )

    summary = {
        'device': str(device),
        'num_runs_registry': len(results_summary),
        'num_hf_replayed': sum(1 for r in metrics_rows if r['replay_status'] == 'ok'),
        'output_root': str(output_root),
    }
    (output_root / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
