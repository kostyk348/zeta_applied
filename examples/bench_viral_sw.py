"""Viral genome fragment identification via spectral sliding window.

For each query fragment, run vectorised sliding_window_scan against every
database genome and pick the best-matching genome (minimum residual).

This is fundamentally different from bench_viral.py: fragment and window
have IDENTICAL positional weighting (1/sqrt(n) for n=1..window_size),
so the comparison is apples-to-apples.
"""

import sys
import time
import random
import socket
import ssl
import http.client
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

import dns.resolver
import numpy as np
from biospectral.dna import dna_to_character_seq, sliding_window_scan

# ---------------------------------------------------------------------------
# DNS workaround
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
    ip = _resolve_ncbi()
    conn = _ResolvedHTTPSConnection(_NCBI_DOMAIN, ip, timeout=30)
    try:
        conn.request("GET",
            f"/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text")
        return conn.getresponse().read().decode("utf-8")
    finally:
        conn.close()


def parse_fasta(text: str) -> str:
    lines = text.splitlines()
    return "".join(l.strip().upper() for l in lines if not l.startswith(">") and l.strip())

# ---------------------------------------------------------------------------

ACCESSIONS: Dict[str, str] = {
    "SARS-CoV-2":      "NC_045512",
    "SARS-CoV":        "NC_004718",
    "MERS-CoV":        "NC_019843",
    "Influenza_PB2":   "NC_026433",
    "Influenza_HA":    "CY121680",
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


def normalized_edit_dist(s: str, t: str) -> float:
    return 1 - SequenceMatcher(None, s, t).ratio()


def run_benchmark() -> None:
    print("=" * 72)
    print("Viral Fragment ID via Spectral Sliding Window")
    print("=" * 72)

    # 1. Download genomes
    print("\n[1] Downloading viral genomes from NCBI...")
    genomes: Dict[str, str] = {}
    for name, acc in ACCESSIONS.items():
        sys.stdout.write(f"  {name} ({acc}) ... ")
        sys.stdout.flush()
        try:
            genomes[name] = parse_fasta(fetch_fasta(acc))
            print(f"{len(genomes[name]):,} bp")
        except Exception as e:
            print(f"FAILED: {e}")

    names = [n for n in ACCESSIONS if n in genomes]
    print(f"\n  Downloaded {len(names)} / {len(ACCESSIONS)} genomes")
    if len(names) < 3:
        return

    # 2. Fragment identification via sliding window
    FRAGMENT_LENGTHS = [100, 250, 500]
    FRAGMENTS_PER_GENOME = 10
    SEED = 42
    T_SPACE = np.linspace(10.0, 80.0, 200)

    print(f"\n[2] Fragment identification via sliding_window_scan")
    print(f"  Fragment lengths: {FRAGMENT_LENGTHS}")
    print(f"  Fragments per genome: {FRAGMENTS_PER_GENOME}")
    print(f"  |t| = {len(T_SPACE)}")
    print()

    rng = random.Random(SEED)

    header = (f"{'Length':>8} | {'SW_top1':>9} {'SW_top3':>9} "
              f"{'SW_q_ms':>9} {'SW_tot_s':>9} | "
              f"{'Edit_top1':>9} {'Edit_top3':>9} {'Edit_tot_s':>9} | "
              f"{'N':>6}")
    print(header)
    print("-" * len(header))

    for frag_len in FRAGMENT_LENGTHS:
        # Generate queries
        queries: List[Tuple[str, str]] = []
        for name in names:
            seq = genomes[name]
            if len(seq) < frag_len:
                continue
            for _ in range(FRAGMENTS_PER_GENOME):
                start = rng.randint(0, len(seq) - frag_len)
                queries.append((name, seq[start:start + frag_len]))

        nq = len(queries)

        # --- Spectral sliding window ---
        t0 = time.perf_counter()
        sw_top1 = 0
        sw_top3 = 0
        for qi, (src_name, fragment) in enumerate(queries):
            if qi % 20 == 0:
                print(f"    sw {qi}/{nq}  frag_len={frag_len}", flush=True)
            scores: List[Tuple[str, float]] = []
            for db_name in names:
                if len(genomes[db_name]) < frag_len:
                    scores.append((db_name, float("inf")))
                    continue
                residuals = sliding_window_scan(
                    genomes[db_name], fragment,
                    window_size=frag_len, t_space=T_SPACE)
                scores.append((db_name, float(np.min(residuals))))
            scores.sort(key=lambda x: x[1])
            if scores[0][0] == src_name:
                sw_top1 += 1
            if src_name in {r[0] for r in scores[:3]}:
                sw_top3 += 1
        sw_time = time.perf_counter() - t0

        # --- Edit distance (baseline, skip 500bp — slow on large genomes) ---
        do_edit = frag_len <= 250
        if do_edit:
            t0 = time.perf_counter()
            edit_top1 = 0
            edit_top3 = 0
            for qi, (src_name, fragment) in enumerate(queries):
                if qi % 10 == 0:
                    print(f"    edit {qi}/{nq}", flush=True)
                scores = [(n, normalized_edit_dist(fragment, genomes[n])) for n in names]
                scores.sort(key=lambda x: x[1])
                if scores[0][0] == src_name:
                    edit_top1 += 1
                if src_name in {r[0] for r in scores[:3]}:
                    edit_top3 += 1
            edit_time = time.perf_counter() - t0
        else:
            edit_top1 = edit_top3 = 0
            edit_time = 0.0

        sw_acc1 = sw_top1 / nq * 100
        sw_acc3 = sw_top3 / nq * 100
        edit_acc1 = edit_top1 / nq * 100 if do_edit else 0
        edit_acc3 = edit_top3 / nq * 100 if do_edit else 0

        edit_tot = f"{edit_time:.1f}" if do_edit else "N/A"
        edit_a1 = f"{edit_acc1:>7.1f}%" if do_edit else "  N/A  "
        edit_a3 = f"{edit_acc3:>7.1f}%" if do_edit else "  N/A  "

        print(
            f"{frag_len:>8} | "
            f"{sw_acc1:>8.1f}% {sw_acc3:>8.1f}% "
            f"{sw_time/nq*1000:>8.3f} {sw_time:>8.1f} | "
            f"{edit_a1:>9} {edit_a3:>9} {edit_tot:>9} | "
            f"{nq:>6}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    run_benchmark()
