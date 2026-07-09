"""Protein benchmark: biospectral vs classical sequence distances.

Database: 7 reference proteins (105--585 AA).  Queries: 350 single-AA
substitution variants (50 per protein).  Methods: spectral (this work),
Hamming, Levenshtein, kmer-Jaccard-3.

Output: top-1 retrieval accuracy, mean rank, mean query time.
"""
from __future__ import annotations

import os
import sys
import time
import random
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biospectral.protein import (
    protein_database_screen,
    REFERENCE_PROTEIN_DATABASE,
)


def hamming_distance(a: str, b: str) -> int:
    n = min(len(a), len(b))
    d = sum(1 for i in range(n) if a[i] != b[i])
    return d + abs(len(a) - len(b))


def levenshtein_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def kmer_set(seq: str, k: int = 3) -> set:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard_distance(a: str, b: str, k: int = 3) -> float:
    sa, sb = kmer_set(a, k), kmer_set(b, k)
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)


AA = "ACDEFGHIKLMNPQRSTVWY"


def make_mutant(seq: str, rng: random.Random) -> str:
    n = len(seq)
    if n < 5:
        pos = rng.randint(0, n - 1)
    else:
        pos = rng.randint(3, n - 4)
    cur = seq[pos]
    new = rng.choice([a for a in AA if a != cur])
    return seq[:pos] + new + seq[pos + 1 :]


def run(n_queries_per_gene: int = 50, seed: int = 42) -> None:
    rng = random.Random(seed)
    database = REFERENCE_PROTEIN_DATABASE
    names = list(database.keys())

    methods = {
        "protein-spectral": lambda q, db: protein_database_screen(q, db),
        "hamming": lambda q, db: sorted(
            [(name, hamming_distance(q, seq)) for name, seq in db.items()],
            key=lambda r: r[1],
        ),
        "levenshtein": lambda q, db: sorted(
            [(name, levenshtein_distance(q, seq)) for name, seq in db.items()],
            key=lambda r: r[1],
        ),
        "kmer-Jaccard-3": lambda q, db: sorted(
            [(name, jaccard_distance(q, seq)) for name, seq in db.items()],
            key=lambda r: r[1],
        ),
    }

    results: Dict[str, Dict[str, List[float]]] = {
        m: {"top1_acc": [], "rank": [], "time_ms": []} for m in methods
    }

    queries: List[Tuple[str, str]] = []
    for name, seq in database.items():
        for _ in range(n_queries_per_gene):
            queries.append((name, make_mutant(seq, rng)))

    print(f"Database: {len(database)} proteins")
    print(f"Sequences: {', '.join(names)}")
    print(f"Queries: {len(queries)} (single-AA substitution)")
    print(f"Methods: {list(methods.keys())}")
    print()

    for true_name, query in queries:
        for method_name, method_fn in methods.items():
            t0 = time.perf_counter()
            ranked = method_fn(query, database)
            t_ms = (time.perf_counter() - t0) * 1000.0
            top1_d = ranked[0][0]
            rank_ = [n for n, _ in ranked].index(true_name) + 1
            results[method_name]["top1_acc"].append(1.0 if top1_d == true_name else 0.0)
            results[method_name]["rank"].append(float(rank_))
            results[method_name]["time_ms"].append(t_ms)

    print(f"{'Method':20s} {'Top-1 acc':>10s} {'Mean rank':>10s} {'Time/query (ms)':>17s}")
    print("-" * 60)
    bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    md = [
        "# Protein benchmark — spectral vs classical distances\n",
        f"Database: {len(database)} proteins ({', '.join(names)}).",
        f"Queries: {len(queries)} single-AA substitution variants.\n",
        "| Method | Top-1 acc | Mean rank | Time/query (ms) |",
        "|-------|----------:|----------:|----------------:|",
    ]
    for method_name in methods:
        top1 = np.mean(results[method_name]["top1_acc"])
        rank_ = np.mean(results[method_name]["rank"])
        t_ms = np.mean(results[method_name]["time_ms"])
        print(f"{method_name:20s} {top1 * 100:9.2f}% {rank_:10.2f} {t_ms:17.3f}")
        md.append(f"| {method_name} | {top1 * 100:.2f}% | {rank_:.2f} | {t_ms:.3f} |")
    md_path = os.path.join(bench_dir, "protein_benchmark.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    run()
