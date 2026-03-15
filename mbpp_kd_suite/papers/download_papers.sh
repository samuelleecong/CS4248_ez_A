#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

copy_if_present() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "reused local paper: $(basename "$dst")"
    return 0
  fi
  return 1
}

mkdir -p "$ROOT_DIR"

copy_if_present "$PROJECT_DIR/../assignment_details/papers/06_embeddistill_2301.12005.pdf" \
  "$ROOT_DIR/01_embeddistill_2301.12005.pdf" || \
  curl -L "https://arxiv.org/pdf/2301.12005.pdf" -o "$ROOT_DIR/01_embeddistill_2301.12005.pdf"

curl -L "https://aclanthology.org/2023.sustainlp-1.23.pdf" \
  -o "$ROOT_DIR/02_qed_align_2023.sustainlp-1.23.pdf"

copy_if_present "$PROJECT_DIR/../assignment_details/papers/05_distilcse_2112.05638.pdf" \
  "$ROOT_DIR/03_distilcse_2112.05638.pdf" || \
  curl -L "https://arxiv.org/pdf/2112.05638.pdf" -o "$ROOT_DIR/03_distilcse_2112.05638.pdf"

copy_if_present "$PROJECT_DIR/../assignment_details/papers/08_pairdistill_2024.emnlp-main.1013.pdf" \
  "$ROOT_DIR/04_pairdistill_2024.emnlp-main.1013.pdf" || \
  curl -L "https://aclanthology.org/2024.emnlp-main.1013.pdf" -o "$ROOT_DIR/04_pairdistill_2024.emnlp-main.1013.pdf"

copy_if_present "$PROJECT_DIR/../assignment_details/papers/09_adam_2024.findings-acl.692.pdf" \
  "$ROOT_DIR/05_adam_2024.findings-acl.692.pdf" || \
  curl -L "https://aclanthology.org/2024.findings-acl.692.pdf" -o "$ROOT_DIR/05_adam_2024.findings-acl.692.pdf"

curl -L "https://aclanthology.org/2022.findings-acl.64.pdf" \
  -o "$ROOT_DIR/06_hpd_2022.findings-acl.64.pdf"

echo "papers available under: $ROOT_DIR"
ls -lh "$ROOT_DIR"/*.pdf
