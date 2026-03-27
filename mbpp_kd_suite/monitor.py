"""Live monitor for two-phase KD experiments.

Usage:
    uv run python monitor.py [RUN_DIR]

If RUN_DIR is omitted, watches the most recent run under artifacts/.
Polls every 5 seconds for new history.json and metrics.json files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"
POLL_INTERVAL = 5  # seconds

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"


def find_latest_run(base: Path | None = None) -> Path:
    if base is None:
        candidates = sorted(ARTIFACTS.rglob("config.json"))
        if candidates:
            return candidates[-1].parent
        # No config.json yet — find the newest timestamped dir
        dirs = [
            d
            for parent in ARTIFACTS.iterdir() if parent.is_dir()
            for d in parent.iterdir() if d.is_dir() and d.name[:8].isdigit()
        ]
        if dirs:
            return max(dirs, key=lambda d: d.name)
    raise FileNotFoundError(f"No run directories found under {ARTIFACTS}")


def format_metric(name: str, value: float) -> str:
    if "MRR" in name or "Recall" in name:
        return f"{name}={GREEN}{value:.4f}{RESET}"
    if "loss" in name.lower() or name in ("one_hot", "distill_kl", "align", "pairwise", "relation", "dark_kl"):
        return f"{name}={YELLOW}{value:.4f}{RESET}"
    return f"{name}={value:.4f}"


def print_history(path: Path, label: str) -> int:
    """Print training history and return number of epochs shown."""
    if not path.exists():
        return 0
    with path.open() as f:
        history = json.load(f)
    if not history:
        return 0

    print(f"\n{BOLD}{CYAN}--- {label} ---{RESET}")
    for row in history:
        epoch = row.get("epoch", "?")
        parts = [f"{BOLD}Epoch {epoch}{RESET}"]

        # Training losses
        loss_keys = ["loss", "one_hot", "distill_kl", "align", "pairwise", "relation", "dark_kl"]
        losses = [format_metric(k, row[k]) for k in loss_keys if k in row and row[k] != 0.0]
        if losses:
            parts.append("  ".join(losses))

        # Validation metrics
        val_keys = [k for k in row if k.startswith("val_")]
        vals = [format_metric(k, row[k]) for k in val_keys]
        if vals:
            parts.append("  ".join(vals))

        print("  " + "  |  ".join(parts))
    return len(history)


def print_final_metrics(path: Path, label: str) -> None:
    if not path.exists():
        return
    with path.open() as f:
        metrics = json.load(f)

    test = metrics.get("test", {})
    if not test:
        return

    print(f"\n{BOLD}{GREEN}=== {label} (test) ==={RESET}")
    parts = [format_metric(k, v) for k, v in test.items() if isinstance(v, (int, float))]
    print("  " + "  ".join(parts))


def monitor(run_dir: Path) -> None:
    print(f"{BOLD}Monitoring:{RESET} {run_dir}")
    print(f"{DIM}Polling every {POLL_INTERVAL}s — Ctrl+C to stop{RESET}\n")

    seen_epochs: dict[str, int] = {}
    seen_finals: set[str] = set()

    while True:
        changed = False

        # Scan for history.json files (training in progress)
        for hpath in sorted(run_dir.rglob("history.json")):
            label = "/".join(hpath.relative_to(run_dir).parts[:-1])
            with hpath.open() as f:
                n_epochs = len(json.load(f))
            prev = seen_epochs.get(label, 0)
            if n_epochs > prev:
                print_history(hpath, label)
                seen_epochs[label] = n_epochs
                changed = True

        # Scan for metrics.json files (training completed for a method)
        for mpath in sorted(run_dir.rglob("metrics.json")):
            label = "/".join(mpath.relative_to(run_dir).parts[:-1])
            if label not in seen_finals:
                print_final_metrics(mpath, label)
                seen_finals.add(label)
                changed = True

        # Check if experiment is done (config.json + results_summary.json both exist)
        if (run_dir / "results_summary.json").exists() and (run_dir / "config.json").exists():
            print(f"\n{BOLD}{GREEN}Experiment complete!{RESET}")
            summary_path = run_dir / "results_summary.json"
            with summary_path.open() as f:
                summary = json.load(f)
            print(f"\n{BOLD}Final Results Summary:{RESET}")
            for name, result in summary.items():
                test = result.get("test", {})
                if test:
                    mrr = test.get("MRR", 0)
                    r1 = test.get("Recall@1", 0)
                    r5 = test.get("Recall@5", 0)
                    r10 = test.get("Recall@10", 0)
                    bar = "=" * int(mrr * 40)
                    print(f"  {name:>30} | MRR={GREEN}{mrr:.4f}{RESET} R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}  {DIM}{bar}{RESET}")
            break

        if not changed:
            # Show a dot to indicate we're still polling
            print(f"{DIM}.{RESET}", end="", flush=True)

        time.sleep(POLL_INTERVAL)


def main() -> None:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        try:
            run_dir = find_latest_run()
        except FileNotFoundError as e:
            print(f"{RED}Error: {e}{RESET}")
            sys.exit(1)

    try:
        monitor(run_dir)
    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")


if __name__ == "__main__":
    main()
