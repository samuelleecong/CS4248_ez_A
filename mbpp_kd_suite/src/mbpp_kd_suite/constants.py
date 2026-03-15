from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TRAINED_BASELINE_NAME = "supervised_student"

KD_METHOD_ORDER = [
    "score_distill",
    "embed_distill",
    "qed_align",
    "distilcse_lite",
    "pair_distill",
    "adam_lite",
    "hpd",
]

METHOD_ORDER = [TRAINED_BASELINE_NAME, *KD_METHOD_ORDER]

TACO_DATASET_NAMES = {
    "BAAI/TACO",
    "BEE-spoke-data/TACO-hf",
}

MPS_TRAIN_BATCH_CAP = 8
MPS_EVAL_BATCH_CAP = 16
ARTIFACT_ROOT = Path("artifacts")


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    title: str
    venue: str
    year: int
    pdf_name: str
    method_name: str
    implementation_note: str


PAPER_SPECS = [
    PaperSpec(
        paper_id="embed_distill",
        title="EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval",
        venue="arXiv",
        year=2023,
        pdf_name="01_embeddistill_2301.12005.pdf",
        method_name="embed_distill",
        implementation_note="score KL plus query embedding alignment",
    ),
    PaperSpec(
        paper_id="qed_align",
        title="Query Encoder Distillation via Embedding Alignment",
        venue="SustainLP",
        year=2023,
        pdf_name="02_qed_align_2023.sustainlp-1.23.pdf",
        method_name="qed_align",
        implementation_note="teacher-query alignment with retrieval contrastive loss",
    ),
    PaperSpec(
        paper_id="distilcse",
        title="DistilCSE: Effective Knowledge Distillation For Contrastive Sentence Embeddings",
        venue="arXiv",
        year=2023,
        pdf_name="03_distilcse_2112.05638.pdf",
        method_name="distilcse_lite",
        implementation_note="CKD-style query InfoNCE plus retrieval loss",
    ),
    PaperSpec(
        paper_id="pair_distill",
        title="PAIR DISTILL: Pairwise Relevance Distillation for Dense Retrieval",
        venue="EMNLP",
        year=2024,
        pdf_name="04_pairdistill_2024.emnlp-main.1013.pdf",
        method_name="pair_distill",
        implementation_note="pairwise preference distillation over teacher-induced hard negatives",
    ),
    PaperSpec(
        paper_id="adam",
        title="ADAM: Dense Retrieval Distillation with Adaptive Dark Examples",
        venue="Findings of ACL",
        year=2024,
        pdf_name="05_adam_2024.findings-acl.692.pdf",
        method_name="adam_lite",
        implementation_note="embedding-space dark examples with teacher-confidence weighting",
    ),
    PaperSpec(
        paper_id="hpd",
        title="Compressing Sentence Representation for Semantic Retrieval via Homomorphic Projective Distillation",
        venue="Findings of ACL",
        year=2022,
        pdf_name="06_hpd_2022.findings-acl.64.pdf",
        method_name="hpd",
        implementation_note="PCA-compressed target space distillation",
    ),
]
