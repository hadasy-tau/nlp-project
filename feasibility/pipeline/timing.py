"""Step timing: every pipeline stage runs inside a timed block and the
elapsed times are collected for the final summary and the results JSON."""

from __future__ import annotations

import time
from contextlib import contextmanager


class StepTimer:
    def __init__(self):
        self.records: dict[str, float] = {}
        self._t_start = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.records[name] = elapsed
        print(f"--- {name}: {_fmt(elapsed)}", flush=True)

    def summary(self) -> str:
        total = time.perf_counter() - self._t_start
        width = max(len(n) for n in self.records) if self.records else 10
        lines = ["", "=" * 48, "TIMING SUMMARY", "=" * 48]
        for name, elapsed in self.records.items():
            lines.append(f"{name:<{width}}  {_fmt(elapsed)}")
        lines.append("-" * 48)
        lines.append(f"{'TOTAL':<{width}}  {_fmt(total)}")
        return "\n".join(lines)


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s ({seconds:.0f}s)"
