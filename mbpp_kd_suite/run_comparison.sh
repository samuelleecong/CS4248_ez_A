#!/usr/bin/env bash
# Run baseline + fine-tuning comparison for all 8 models.
# Usage: bash run_comparison.sh
#   (from the mbpp_kd_suite/ directory, or any directory — the script cd's itself)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install the package
if command -v uv &>/dev/null; then
    uv pip install -e .
else
    pip install -e .
fi

echo "=== Step 1/2: Zero-shot baselines ==="
python scratch/baseline_comparison.py

echo ""
echo "=== Step 2/2: Fine-tune all models ==="
python scratch/finetune_comparison.py

echo ""
echo "Done! Results in artifacts/"
