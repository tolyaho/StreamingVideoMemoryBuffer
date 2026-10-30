#!/usr/bin/env python3
"""Paired comparison of the committed StreamingBench runs (no models, no videos needed).

For every run under results/, counts the questions each system gets right, the discordant pairs
(only hierarchical right / only baseline right) and an exact two-sided McNemar test, i.e. a binomial
test on the discordant pairs with p = 0.5.

    python scripts/paired_stats.py
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("streamingbench_text_3b_notebook", "streamingbench_text_3b_tuned", "streamingbench_vlm_8b")


def exact_mcnemar(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(run: str) -> list[dict]:
    rows = []
    for path in sorted((ROOT / "results" / run / "per_video").glob("sample_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main() -> int:
    print(f"{'run':34} {'n':>4} {'hier':>5} {'base':>5} {'delta':>8} {'only_h':>7} {'only_b':>7} {'p (exact McNemar)':>18}")
    for run in RUNS:
        rows = load(run)
        h = sum(r["hierarchical"]["correct"] for r in rows)
        b = sum(r["baseline"]["correct"] for r in rows)
        only_h = sum(r["hierarchical"]["correct"] and not r["baseline"]["correct"] for r in rows)
        only_b = sum(r["baseline"]["correct"] and not r["hierarchical"]["correct"] for r in rows)
        delta = 100 * (h - b) / len(rows)
        print(f"{run:34} {len(rows):>4} {h:>5} {b:>5} {delta:>+7.1f}pp {only_h:>7} {only_b:>7} {exact_mcnemar(only_h, only_b):>18.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
