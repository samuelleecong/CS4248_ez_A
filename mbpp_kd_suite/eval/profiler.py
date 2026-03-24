from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

import psutil
import torch

T = TypeVar("T")


class StageProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.process = psutil.Process(os.getpid())
        self._rows: list[dict[str, float | str]] = []
        self._total_start = time.perf_counter()

    def profile(self, stage_name: str, fn: Callable[[], T]) -> T:
        rss_before = self.process.memory_info().rss
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        result = fn()
        duration_sec = time.perf_counter() - start
        rss_after = self.process.memory_info().rss
        peak_memory_bytes = self._peak_memory_bytes(rss_before=rss_before, rss_after=rss_after)
        self._rows.append(
            {
                "stage": stage_name,
                "duration_sec": float(duration_sec),
                "peak_memory_bytes": float(peak_memory_bytes),
            }
        )
        return result

    def finalize(self) -> dict[str, Any]:
        total_duration_sec = time.perf_counter() - self._total_start
        total_peak_memory_bytes = 0.0
        if self._rows:
            total_peak_memory_bytes = max(float(row["peak_memory_bytes"]) for row in self._rows)
        self._rows.append(
            {
                "stage": "total_eval",
                "duration_sec": float(total_duration_sec),
                "peak_memory_bytes": float(total_peak_memory_bytes),
            }
        )
        return {
            "stages": list(self._rows),
            "total_duration_sec": float(total_duration_sec),
            "peak_memory_bytes": float(total_peak_memory_bytes),
        }

    def _peak_memory_bytes(self, rss_before: int, rss_after: int) -> float:
        if self.device.type == "cuda":
            return float(torch.cuda.max_memory_allocated(self.device))
        return float(max(rss_before, rss_after))
