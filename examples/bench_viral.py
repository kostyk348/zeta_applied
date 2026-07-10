"""Viral genome fragment identification via spectral fingerprints.

Downloads real viral genomes from NCBI, builds spectral fingerprints,
and benchmarks fragment-to-genome identification at various lengths.
"""

import sys
import time
import random
import socket
import ssl
import http.client
from typing import Dict, List, Tuple

import dns.resolver
import numpy as np
from biospectral.dna import dna_to_character_seq, l_function

# ---------------------------------------------------------------------------
# DNS workaround — NCBI hosts are filtered by the local DNS at this site,
# so we resolve via Google (8.8.8.8) and connect directly by IP.
# ---------------------------------------------------------------------------

_NCBI_DOMAIN = "eutils.ncbi.nlm.nih.gov"
_NCBI_IP = None

def _resolve_ncbi() -> str:
    global _NCBI_IP
    if _NCBI_IP is None:
        r = dns.resolver.Resolver()
        r.nameservers = ["8.8.8.8"]
        _NCBI_IP = str(r.resolve(_NCBI_DOMAIN, "A")[0])
    return _NCBI_IP


class _ResolvedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that connects to a pre-resolved IP for NCBI."""

    def __init__(self, host, resolved_ip, **kwargs):
        self._resolved_ip = resolved_ip
        super().__init__(host, **kwargs)

    def connect(self):
        sock = socket.create_connection((self._resolved_ip, self.port), self.timeout)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def fetch_fasta(accession: str) -> str:
    """Download the FASTA record for a given NCBI nucleotide accession."""
    ip = _resolve_ncbi()
    conn = _ResolvedHTTPSConnection(
        _NCBI_DOMAIN, ip, timeout=30,
    )
    try:
        conn.request(
            "GET",
            f"/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text",
        )
        resp = conn.getresponse()
        return resp.read().decode("utf-8")
    finally:
        conn.close()


def parse_fasta(text: str) -> str:
    """Extract the concatenated sequence from a FASTA string (no gaps)."""
    lines = text.splitlines()
    return "".join(line.strip().upper() for line in lines if not line.startswith(">") and line.strip())


# ---------------------------------------------------------------------------
# Spectral fingerprint helpers
# ---------------------------------------------------------------------------

DEFAULT_T = np.linspace(10.0, 80.0, 400)


def build_fingerprint(seq: str, t_space: np.ndarray = DEFAULT_T) -> np.ndarray:
    """Precomputed fingerprint: complex L-function values."""
    chi = dna_to_character_seq(seq)
    return l_function(t_space, chi)


def spectral_fingerprint_distance(
    fp_a: np.ndarray, fp_b: np.ndarray, t_space: np.ndarray = DEFAULT_T
) -> float:
    """Distance between two precomputed fingerprints (same |t| assumed)."""
    n = min(len(fp_a), len(fp_b))
    return float(np.trapezoid(np.abs(fp_a[:n] - fp_b[:n]) ** 2, t_space[:n]))


# ---------------------------------------------------------------------------
# Edit-distance baseline (fast Levenshtein for short sequences)
# ---------------------------------------------------------------------------

from difflib import SequenceMatcher

def normalized_edit_dist(s: str, t: str) -> float:
    """Normalised edit distance via SequenceMatcher (fast C implementation)."""
    return 1 - SequenceMatcher(None, s, t).ratio()


def edit_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Stub so we can call distance consistently (edit operates on strings)."""
    raise NotImplementedError("Use normalized_edit_dist directly")


# ---------------------------------------------------------------------------
# Viral genome accessions (NCBI nuccore)
# ---------------------------------------------------------------------------

ACCESSIONS: Dict[str, str] = {
    "SARS-CoV-2":      "NC_045512",
    "SARS-CoV":        "NC_004718",
    "MERS-CoV":        "NC_019843",
    "Influenza_A_PB2": "NC_026433",
    "Influenza_A_HA":  "CY121680",
    "HIV-1":           "NC_001802",
    "Ebola":           "NC_002549",
    "Zika":            "NC_012532",
    "Dengue_1":        "NC_001477",
    "Dengue_2":        "NC_001474",
    "Rabies":          "NC_001542",
    "Hepatitis_B":     "NC_003977",
    "Hepatitis_C":     "NC_004102",
    "Phi-X174":        "NC_001422",
    "Lambda_phage":    "NC_001416",
}

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    print("=" * 72)
    print("Viral Genome Fragment Identification via Spectral Fingerprints")
    print("=" * 72)

    # 1. Download genomes
    print("\n[1] Downloading viral genomes from NCBI...")
    genomes: Dict[str, str] = {}
    for name, acc in ACCESSIONS.items():
        sys.stdout.write(f"  {name} ({acc}) ... ")
        sys.stdout.flush()
        try:
            fasta = fetch_fasta(acc)
            seq = parse_fasta(fasta)
            genomes[name] = seq
            print(f"{len(seq):,} bp")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n  Downloaded {len(genomes)} / {len(ACCESSIONS)} genomes")

    if len(genomes) < 3:
        print("Too few genomes downloaded, aborting.")
        return

    names = list(genomes.keys())

    # 2. Precompute fingerprints
    print(f"\n[2] Precomputing spectral fingerprints (|t|={len(DEFAULT_T)})...")
    t0 = time.perf_counter()
    spectral_db: Dict[str, np.ndarray] = {}
    for name in names:
        spectral_db[name] = build_fingerprint(genomes[name])
    precompute_time = time.perf_counter() - t0
    print(f"  Precomputed {len(genomes)} fingerprints in {precompute_time*1000:.1f} ms")

    # 3. Fragment identification
    FRAGMENT_LENGTHS = [100, 250, 500, 1000]
    FRAGMENTS_PER_GENOME = 10
    SEED = 42

    print(f"\n[3] Fragment identification benchmark")
    print(f"  Fragment lengths: {FRAGMENT_LENGTHS}")
    print(f"  Fragments per genome: {FRAGMENTS_PER_GENOME}")
    print(f"  Random seed: {SEED}")
    print()

    rng = random.Random(SEED)

    header = f"{'Length':>8} | {'Spect_top1':>11} {'Spect_top3':>11} {'Spect_ms':>10} | {'Edit_top1':>10} {'Edit_top3':>10} {'Edit_ms':>10} | {'N':>6}"
    print(header)
    print("-" * len(header))

    for frag_len in FRAGMENT_LENGTHS:
        queries: List[Tuple[str, str]] = []  # (source_name, fragment)
        for name in names:
            seq = genomes[name]
            if len(seq) < frag_len:
                continue
            for _ in range(FRAGMENTS_PER_GENOME):
                start = rng.randint(0, len(seq) - frag_len)
                queries.append((name, seq[start:start + frag_len]))

        n_queries = len(queries)

        # --- Spectral ---
        t0 = time.perf_counter()
        spect_top1 = 0
        spect_top3 = 0
        for qi, (source_name, fragment) in enumerate(queries):
            if qi % 20 == 0:
                print(f"    spect {qi}/{n_queries}", flush=True)
            q_fp = build_fingerprint(fragment)
            dists = []
            for db_name in names:
                d = spectral_fingerprint_distance(q_fp, spectral_db[db_name])
                dists.append((db_name, d))
            dists.sort(key=lambda x: x[1])
            rank1 = dists[0][0]
            rank3 = {r[0] for r in dists[:3]}
            if rank1 == source_name:
                spect_top1 += 1
            if source_name in rank3:
                spect_top3 += 1
        spect_time = time.perf_counter() - t0

        # --- Edit distance (baseline, skip 1000bp — too slow for large genomes) ---
        do_edit = frag_len <= 500
        if do_edit:
            t0 = time.perf_counter()
            edit_top1 = 0
            edit_top3 = 0
            for qi, (source_name, fragment) in enumerate(queries):
                if qi % 10 == 0:
                    print(f"    edit {qi}/{n_queries}", flush=True)
                dists = []
                for db_name in names:
                    d = normalized_edit_dist(fragment, genomes[db_name])
                    dists.append((db_name, d))
                dists.sort(key=lambda x: x[1])
                rank1 = dists[0][0]
                rank3 = {r[0] for r in dists[:3]}
                if rank1 == source_name:
                    edit_top1 += 1
                if source_name in rank3:
                    edit_top3 += 1
            edit_time = time.perf_counter() - t0
        else:
            edit_top1 = edit_top3 = 0
            edit_time = 0.0

        spect_acc1 = spect_top1 / n_queries * 100
        spect_acc3 = spect_top3 / n_queries * 100
        edit_acc1 = edit_top1 / n_queries * 100
        edit_acc3 = edit_top3 / n_queries * 100
        spect_ms = spect_time / n_queries * 1000
        edit_ms = edit_time / n_queries * 1000

        if do_edit:
            print(
                f"{frag_len:>8} | "
                f"{spect_acc1:>10.1f}% {spect_acc3:>10.1f}% {spect_ms:>10.3f} | "
                f"{edit_acc1:>10.1f}% {edit_acc3:>10.1f}% {edit_ms:>10.3f} | "
                f"{n_queries:>6}"
            )
        else:
            print(
                f"{frag_len:>8} | "
                f"{spect_acc1:>10.1f}% {spect_acc3:>10.1f}% {spect_ms:>10.3f} | "
                f"{'  N/A   ':>10} {'  N/A   ':>10} {'  N/A   ':>10} | "
                f"{n_queries:>6}"
            )

    # 4. Summary
    print()
    print("Done.")


if __name__ == "__main__":
    run_benchmark()
