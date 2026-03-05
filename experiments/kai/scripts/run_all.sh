#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
cd "$SCRIPT_DIR/../../.."

uv venv --python 3.10 .venv310
uv pip install --python .venv310/bin/python sentence-transformers datasets faiss-cpu accelerate scikit-learn pandas tqdm matplotlib

. .venv310/bin/activate

python experiments/kai/scripts/run_mbpp_experiments.py --output-dir experiments/kai/results --run-id mbpp_full_matrix --device auto --seed 42 --full-matrix --finetune-all-pretrained
python experiments/kai/scripts/run_mbpp_experiments.py --output-dir experiments/kai/results --run-id mbpp_full_matrix --device auto --seed 42 --full-matrix --finetune-all-pretrained --resume
python experiments/kai/scripts/plot_mbpp_results.py --run-dir experiments/kai/results/mbpp_full_matrix
