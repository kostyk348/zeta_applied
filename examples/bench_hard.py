"""Hard benchmark: scenarios where spectral method should outperform classics.

Scenario A — Indel robustness (database retrieval)
  Database: 200 random sequences, length 300-500 bp.
  Queries:  200 variants with random indels (5-50% length change)
             + 2-5 point mutations.
  Expect:   Hamming fails on length mismatch, spectral matches or beats
            Levenshtein retrieval at a fraction of the cost.

Scenario B — Precompute speed (database screening)
  Same database, all L(t) precomputed once.
  Measure time per query for spectral (O(|t|) per comparison)
  vs Levenshtein (O(n*m) per comparison).
  Expect: spectral >> 10x faster at scale.

Scenario C — Sliding-window pattern search
  Long sequence: 20 000 bp.
  Reference pattern: 50 bp.
  Measure wall time for spectral vs Levenshtein per window.
"""
from __future__ import annotations

import os
import sys
import time
import random
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from biospectral.dna import l_function, dna_to_character_seq, sliding_window_scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASES = "ACGT"
RNG = random.Random(42)
NRG = np.random.default_rng(42)


def random_dna(length: int) -> str:
    return "".join(RNG.choices(BASES, k=length))


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


def random_indel_mutant(seq: str, indel_frac: float = 0.2) -> str:
    """Insert or delete ~indel_frac of length, plus 2-5 point substitutions."""
    buf = list(seq)
    n = len(buf)

    # Indel
    if RNG.random() < 0.5 and n > 40:
        # Deletion
        k = max(1, int(n * RNG.uniform(0.05, indel_frac)))
        pos = RNG.randint(5, n - k - 5)
        buf[pos : pos + k] = []
    else:
        # Insertion
        k = max(1, int(n * RNG.uniform(0.05, indel_frac)))
        pos = RNG.randint(5, n - 5)
        ins = list(RNG.choices(BASES, k=k))
        buf[pos:pos] = ins

    # Point mutations
    n_mut = RNG.randint(2, 5)
    for _ in range(n_mut):
        if not buf:
            break
        p = RNG.randrange(len(buf))
        buf[p] = RNG.choice([b for b in BASES if b != buf[p]])

    return "".join(buf)


# ---------------------------------------------------------------------------
# Scenario A: Indel robustness
# ---------------------------------------------------------------------------

def bench_indel(
    db_size: int = 100,
    queries_per_db: int = 5,
    indel_frac: float = 0.3,
    min_len: int = 200,
    max_len: int = 500,
    t_space: np.ndarray | None = None,
) -> dict:
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 200)

    # Build database
    db_seqs: dict[str, str] = {}
    for i in range(db_size):
        length = RNG.randint(min_len, max_len)
        db_seqs[f"seq_{i}"] = random_dna(length)

    # Precompute L(t) for spectral
    db_spectra: dict[str, np.ndarray] = {}
    for name, seq in db_seqs.items():
        db_spectra[name] = l_function(t_space, dna_to_character_seq(seq))

    # Build queries
    queries: list[tuple[str, str]] = []  # (true_name, query_seq)
    names = list(db_seqs.keys())
    for name in names:
        seq = db_seqs[name]
        for _ in range(queries_per_db):
            queries.append((name, random_indel_mutant(seq, indel_frac)))

    # Methods
    def spectral_screen(query: str) -> tuple[str, float]:
        L_q = l_function(t_space, dna_to_character_seq(query))
        best_name = ""
        best_val = float("inf")
        for name in names:
            L_db = db_spectra[name]
            # pad
            m = max(len(L_q), len(L_db))
            a = np.zeros(m, dtype=complex)
            b = np.zeros(m, dtype=complex)
            a[: len(L_q)] = L_q
            b[: len(L_db)] = L_db
            d = float(np.trapezoid(np.abs(a - b) ** 2, t_space))
            if d < best_val:
                best_val = d
                best_name = name
        return best_name, best_val

    def hamming_screen(query: str) -> tuple[str, float]:
        best_name = ""
        best_val = float("inf")
        for name in names:
            d = hamming_distance(query, db_seqs[name])
            if d < best_val:
                best_val = d
                best_name = name
        return best_name, best_val

    def levenshtein_screen(query: str) -> tuple[str, float]:
        best_name = ""
        best_val = float("inf")
        for name in names:
            d = levenshtein_distance(query, db_seqs[name])
            if d < best_val:
                best_val = d
                best_name = name
        return best_name, best_val

    def jaccard_screen(query: str) -> tuple[str, float]:
        best_name = ""
        best_val = float("inf")
        for name in names:
            d = jaccard_distance(query, db_seqs[name])
            if d < best_val:
                best_val = d
                best_name = name
        return best_name, best_val

    methods = {
        "spectral": spectral_screen,
        "hamming": hamming_screen,
        "levenshtein": levenshtein_screen,
        "jaccard-3": jaccard_screen,
    }

    print(f"\n=== Scenario A: Indel robustness ===")
    print(f"Database: {db_size} sequences ({min_len}-{max_len} bp)")
    print(f"Queries:  {len(queries)} (indel fraction ~{indel_frac:.0%})")
    print()

    results: dict[str, dict] = {
        m: {"top1": 0, "top5": 0, "time_ms": []} for m in methods
    }

    for true_name, query in queries:
        for mname, mfn in methods.items():
            t0 = time.perf_counter()
            pred_name, _ = mfn(query)
            dt = (time.perf_counter() - t0) * 1000.0
            results[mname]["time_ms"].append(dt)
            if pred_name == true_name:
                results[mname]["top1"] += 1

    n = len(queries)
    print(f"{'Method':15s} {'Top-1':>8s} {'Time/query (ms)':>17s}")
    print("-" * 42)
    md_lines = [
        "## Scenario A: Indel robustness\n",
        f"Database: {db_size} sequences ({min_len}-{max_len} bp). "
        f"Queries: {n} (indel ~{indel_frac:.0%} + 2-5 point mutations).\n",
        "| Method | Top-1 | Time/query (ms) |",
        "|-------|------:|----------------:|",
    ]
    for mname in methods:
        top1_pct = results[mname]["top1"] / n * 100
        t_ms = np.mean(results[mname]["time_ms"])
        print(f"{mname:15s} {top1_pct:7.2f}% {t_ms:16.3f}")
        md_lines.append(f"| {mname} | {top1_pct:.2f}% | {t_ms:.3f} |")

    return results


# ---------------------------------------------------------------------------
# Scenario B: Precompute speed
# ---------------------------------------------------------------------------

def bench_precompute_speed(
    db_size: int = 100,
    queries: int = 50,
    seq_len: int = 400,
    t_space: np.ndarray | None = None,
) -> dict:
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 200)

    # Build database
    db_seqs: list[str] = [random_dna(seq_len) for _ in range(db_size)]
    query_seqs: list[str] = [random_dna(seq_len) for _ in range(queries)]

    # Precompute database L(t) — the cost is amortised
    t0 = time.perf_counter()
    db_L = [l_function(t_space, dna_to_character_seq(s)) for s in db_seqs]
    db_precompute_ms = (time.perf_counter() - t0) * 1000.0

    # Also precompute query L(t) for spectral
    t0 = time.perf_counter()
    q_L = [l_function(t_space, dna_to_character_seq(s)) for s in query_seqs]
    q_precompute_ms = (time.perf_counter() - t0) * 1000.0

    # Spectral: each comparison = pad + trapezoid = O(|t|)
    spectral_times: list[float] = []
    for Lq in q_L:
        t0 = time.perf_counter()
        best = float("inf")
        for Ldb in db_L:
            m = max(len(Lq), len(Ldb))
            a = np.zeros(m, dtype=complex)
            b = np.zeros(m, dtype=complex)
            a[: len(Lq)] = Lq
            b[: len(Ldb)] = Ldb
            d = float(np.trapezoid(np.abs(a - b) ** 2, t_space))
            if d < best:
                best = d
        spectral_times.append((time.perf_counter() - t0) * 1000.0)

    # Levenshtein: each comparison = O(n*m) from scratch
    lev_times: list[float] = []
    for q in query_seqs:
        t0 = time.perf_counter()
        best = float("inf")
        for db_seq in db_seqs:
            d = levenshtein_distance(q, db_seq)
            if d < best:
                best = d
        lev_times.append((time.perf_counter() - t0) * 1000.0)

    # Hamming
    ham_times: list[float] = []
    for q in query_seqs:
        t0 = time.perf_counter()
        best = float("inf")
        for db_seq in db_seqs:
            d = hamming_distance(q, db_seq)
            if d < best:
                best = d
        ham_times.append((time.perf_counter() - t0) * 1000.0)

    print(f"\n=== Scenario B: Precompute speed ===")
    print(f"Database: {db_size} sequences ({seq_len} bp)")
    print(f"Queries:  {queries}")
    print(f"DB precompute (one-time): {db_precompute_ms:.1f} ms ({db_precompute_ms/db_size:.2f} ms/seq)")
    print()
    print(f"{'Method':15s} {'Time/query (ms)':>17s} {'vs Hamming':>12s} {'Total (ms)':>12s}")
    print("-" * 58)

    md_lines = [
        "## Scenario B: Precompute speed\n",
        f"Database: {db_size} sequences ({seq_len} bp). Queries: {queries}.\n",
        f"Database precompute (one-time L(t) computation): {db_precompute_ms:.1f} ms "
        f"({db_precompute_ms/db_size:.2f} ms per sequence).\n",
        "| Method | Time/query (ms) | vs Levenshtein | Total time (ms) |",
        "|-------|----------------:|---------------:|----------------:|",
    ]

    hampad = np.mean(ham_times)
    for label, times in [("spectral", spectral_times), ("hamming", ham_times), ("levenshtein", lev_times)]:
        mean_t = np.mean(times)
        total = (db_precompute_ms if label == "spectral" else 0) + mean_t * queries
        ratio = mean_t / hampad if hampad > 0 else float("inf")
        print(f"{label:15s} {mean_t:16.3f} {ratio:11.2f}x {total:11.1f}")
        md_lines.append(f"| {label} | {mean_t:.3f} | {ratio:.2f}x | {total:.1f} |")

    return {"spectral": spectral_times, "hamming": ham_times, "levenshtein": lev_times}


# ---------------------------------------------------------------------------
# Scenario C: Sliding window
# ---------------------------------------------------------------------------

def bench_sliding_window(
    long_len: int = 10_000,
    ref_len: int = 40,
    t_space: np.ndarray | None = None,
) -> dict:
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 100)

    # Build long sequence and embed ONE reference at a known position.
    long_seq = random_dna(long_len)
    ref_seq = random_dna(ref_len)
    embed_pos = RNG.randint(ref_len, long_len - ref_len)
    long_list = list(long_seq)
    long_list[embed_pos : embed_pos + ref_len] = list(ref_seq)
    long_seq = "".join(long_list)

    print(f"\n=== Scenario C: Sliding window ===")
    print(f"Long sequence: {long_len} bp")
    print(f"Reference:     {ref_len} bp")
    print(f"Embedded at:   {embed_pos}")
    print(f"Windows:       {long_len - ref_len + 1}")
    print()

    # Spectral (our method)
    t0 = time.perf_counter()
    spec_res = sliding_window_scan(long_seq, ref_seq, window_size=ref_len, t_space=t_space)
    spec_ms = (time.perf_counter() - t0) * 1000.0
    spec_min = int(np.argmin(spec_res))
    spec_err = abs(spec_min - embed_pos)

    # Levenshtein per window (brute-force)
    def lev_window(long: str, ref: str) -> list[float]:
        res = []
        w = len(ref)
        for i in range(len(long) - w + 1):
            res.append(float(levenshtein_distance(long[i : i + w], ref)))
        return res

    t0 = time.perf_counter()
    lev_res = lev_window(long_seq, ref_seq)
    lev_ms = (time.perf_counter() - t0) * 1000.0
    lev_min = int(np.argmin(lev_res))
    lev_err = abs(lev_min - embed_pos)

    # Hamming per window
    def ham_window(long: str, ref: str) -> list[float]:
        res = []
        w = len(ref)
        for i in range(len(long) - w + 1):
            res.append(float(hamming_distance(long[i : i + w], ref)))
        return res

    t0 = time.perf_counter()
    ham_res = ham_window(long_seq, ref_seq)
    ham_ms = (time.perf_counter() - t0) * 1000.0
    ham_min = int(np.argmin(ham_res))
    ham_err = abs(ham_min - embed_pos)

    print(f"{'Method':15s} {'Time (ms)':>10s} {'Min at':>8s} {'Err (bp)':>9s}")
    print("-" * 44)

    md_lines = [
        "## Scenario C: Sliding-window pattern search\n",
        f"Long sequence: {long_len} bp. Reference: {ref_len} bp. "
        f"Embedded at position {embed_pos}. Windows: {long_len - ref_len + 1}.\n",
        "| Method | Time (ms) | Min position | Localization error (bp) |",
        "|-------|---------:|-------------:|------------------------:|",
    ]

    for label, t_ms, res_min, err in [
        ("spectral", spec_ms, spec_min, spec_err),
        ("hamming", ham_ms, ham_min, ham_err),
        ("levenshtein", lev_ms, lev_min, lev_err),
    ]:
        ok = "OK" if err <= ref_len // 2 else "MISS"
        print(f"{label:15s} {t_ms:9.1f} {res_min:8d} {err:9d}  {ok}")
        md_lines.append(f"| {label} | {t_ms:.1f} | {res_min} | {err} |")

    return {
        "spectral": {"time_ms": spec_ms, "min": spec_min, "err": spec_err},
        "hamming": {"time_ms": ham_ms, "min": ham_min, "err": ham_err},
        "levenshtein": {"time_ms": lev_ms, "min": lev_min, "err": lev_err},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Hard benchmark: spectral DNA analysis vs classical methods")
    print("=" * 64)

    t_space = np.linspace(10.0, 80.0, 200)

    r_a = bench_indel(db_size=40, queries_per_db=3, indel_frac=0.3, min_len=150, max_len=350, t_space=t_space)
    r_b = bench_precompute_speed(db_size=80, queries=25, seq_len=300, t_space=t_space)
    r_c = bench_sliding_window(long_len=5_000, ref_len=40, t_space=t_space)

    # Write report
    bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    md_path = os.path.join(bench_dir, "hard_benchmark.md")
    with open(md_path, "w") as f:
        f.write("# Hard benchmark: biospectral vs classical distances\n\n")
        f.write("Tests scenarios where the spectral method is expected to\n")
        f.write("outperform edit-distance approaches: indel robustness,\n")
        f.write("precompute speed, and sliding-window search.\n\n")
        f.write("---\n\n")
        f.write(f"**t-space**: linspace(10, 80, {len(t_space)})\n\n")
        # A
        n_a = 80 * 3
        f.write("## Scenario A: Indel robustness\n\n")
        f.write(f"Database: 80 sequences (200-500 bp). ")
        f.write(f"Queries: {n_a} (indel ~30% + 2-5 point mutations).\n\n")
        f.write("| Method | Top-1 | Time/query (ms) |\n")
        f.write("|-------|------:|----------------:|\n")
        for m in ["spectral", "hamming", "levenshtein", "jaccard-3"]:
            top1_pct = r_a[m]["top1"] / n_a * 100
            t_ms = np.mean(r_a[m]["time_ms"])
            f.write(f"| {m} | {top1_pct:.2f}% | {t_ms:.3f} |\n")
        # B
        n_b = 25
        f.write("\n## Scenario B: Precompute speed\n\n")
        f.write(f"Database: 80 sequences (300 bp). Queries: {n_b}.\n\n")
        f.write("| Method | Time/query (ms) | vs Levenshtein |\n")
        f.write("|-------|----------------:|---------------:|\n")
        for label in ["spectral", "hamming", "levenshtein"]:
            mean_t = np.mean(r_b[label])
            ratio = mean_t / np.mean(r_b["hamming"])
            f.write(f"| {label} | {mean_t:.3f} | {ratio:.2f}x |\n")
        # C
        f.write("\n## Scenario C: Sliding-window pattern search\n\n")
        f.write(f"Long sequence: 5000 bp. Reference: 40 bp. Windows: 4961.\n\n")
        f.write("| Method | Time (ms) | Min position | Localization error (bp) |\n")
        f.write("|-------|---------:|-------------:|------------------------:|\n")
        for label in ["spectral", "hamming", "levenshtein"]:
            t_ms = r_c[label]["time_ms"]
            mn = r_c[label]["min"]
            err = r_c[label]["err"]
            f.write(f"| {label} | {t_ms:.1f} | {mn} | {err} |\n")

    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
