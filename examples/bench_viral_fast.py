"""Viral fragment ID via sliding window — optimised with precomputed genome matrices.
"""
import sys, time, random, socket, ssl, http.client
from typing import Dict, List, Tuple

import dns.resolver
import numpy as np
from biospectral.dna import dna_to_character_seq

_NCBI_DOMAIN = "eutils.ncbi.nlm.nih.gov"
_NCBI_IP = None
def _resolve_ncbi():
    global _NCBI_IP
    if _NCBI_IP is None:
        r = dns.resolver.Resolver()
        r.nameservers = ["8.8.8.8"]
        _NCBI_IP = str(r.resolve(_NCBI_DOMAIN, "A")[0])
    return _NCBI_IP

class _ResolvedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, rip, **kw):
        self._rip = rip
        super().__init__(host, **kw)
    def connect(self):
        self.sock = self._context.wrap_socket(
            socket.create_connection((self._rip, self.port), self.timeout),
            server_hostname=self.host)

def fetch_fasta(acc):
    ip = _resolve_ncbi()
    conn = _ResolvedHTTPSConnection(_NCBI_DOMAIN, ip, timeout=30)
    try:
        conn.request("GET", f"/entrez/eutils/efetch.fcgi?db=nuccore&id={acc}&rettype=fasta&retmode=text")
        return conn.getresponse().read().decode("utf-8")
    finally:
        conn.close()

def parse_fasta(text):
    return "".join(l.strip().upper() for l in text.splitlines() if not l.startswith(">") and l.strip())

def l_function_fast(t, char_seq):
    n = np.arange(1, len(char_seq) + 1, dtype=np.float64)
    return np.exp(-1j * np.outer(t, np.log(n))) @ (char_seq.astype(complex) / np.sqrt(n))

def precompute_genome_l_matrix(chi, window_size, t_space):
    n_w = len(chi) - window_size + 1
    if n_w <= 0:
        return None
    windows = np.lib.stride_tricks.as_strided(
        chi, shape=(n_w, window_size),
        strides=(chi.strides[0], chi.strides[0])
    ).astype(np.float64)
    k = np.arange(1, window_size + 1, dtype=np.float64)
    basis = np.exp(-1j * np.outer(np.log(k), t_space))
    return (windows / np.sqrt(k)[None, :]) @ basis  # (n_w, |t|)

ACCESSIONS = {
    "SARS-CoV-2": "NC_045512","SARS-CoV": "NC_004718","MERS-CoV": "NC_019843",
    "Influenza_PB2": "NC_026433","Influenza_HA": "CY121680","HIV-1": "NC_001802",
    "Ebola": "NC_002549","Zika": "NC_012532","Dengue_1": "NC_001477",
    "Dengue_2": "NC_001474","Rabies": "NC_001542","Hepatitis_B": "NC_003977",
    "Hepatitis_C": "NC_004102","Phi-X174": "NC_001422","Lambda_phage": "NC_001416",
}

def run():
    print("="*72)
    print("Viral Fragment ID — Precomputed Sliding Window")
    print("="*72)

    print("\n[1] Downloading genomes...")
    genomes = {}
    for name, acc in ACCESSIONS.items():
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            genomes[name] = parse_fasta(fetch_fasta(acc))
            print(f"{len(genomes[name]):,} bp")
        except Exception as e:
            print(f"FAIL: {e}")
    names = [n for n in ACCESSIONS if n in genomes]
    print(f"\n  Got {len(names)} genomes")

    FRAGS = [100, 250, 500]
    PER_GENOME = 10
    SEED = 42
    T_SPACE = np.linspace(10.0, 80.0, 200)
    rng = random.Random(SEED)

    print(f"\n[2] Precomputing genome L-matrices for each window size...")
    t0 = time.perf_counter()
    precomputed = {}  # frag_len -> {genome_name -> L_matrix}
    for flen in FRAGS:
        precomputed[flen] = {}
        for name in names:
            if len(genomes[name]) < flen:
                continue
            chi = dna_to_character_seq(genomes[name])
            L = precompute_genome_l_matrix(chi, flen, T_SPACE)
            precomputed[flen][name] = L
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    print(f"\n[3] Fragment ID benchmark")
    header = f"{'Len':>6} | {'top1':>7} {'top3':>7} {'q_ms':>8} {'total_s':>8} | {'N':>5}"
    print(header)
    print("-"*len(header))

    for flen in FRAGS:
        queries = []
        for name in names:
            seq = genomes[name]
            if len(seq) < flen: continue
            for _ in range(PER_GENOME):
                s = rng.randint(0, len(seq) - flen)
                queries.append((name, seq[s:s+flen]))

        nq = len(queries)
        t0 = time.perf_counter()
        top1 = top3 = 0

        for qi, (src, frag) in enumerate(queries):
            if qi % 20 == 0:
                print(f"    q {qi}/{nq}  flen={flen}", flush=True)
            q_chi = dna_to_character_seq(frag)
            q_L = l_function_fast(T_SPACE, q_chi)
            best_name = None
            best_score = float("inf")
            scores = []
            for db_name in names:
                L_db = precomputed[flen].get(db_name)
                if L_db is None:
                    scores.append((db_name, float("inf")))
                    continue
                n_min = min(len(q_L), L_db.shape[1])
                diff = q_L[:n_min] - L_db[:, :n_min]
                residual = np.trapezoid(np.abs(diff)**2, T_SPACE[:n_min], axis=1)
                min_r = float(np.min(residual))
                scores.append((db_name, min_r))
            scores.sort(key=lambda x: x[1])
            if scores[0][0] == src:
                top1 += 1
            if src in {r[0] for r in scores[:3]}:
                top3 += 1

        total_s = time.perf_counter() - t0
        print(f"{flen:>6} | {top1/nq*100:>6.1f}% {top3/nq*100:>6.1f}% "
              f"{total_s/nq*1000:>7.1f} {total_s:>7.1f} | {nq:>5}")

    print("\nDone.")

if __name__ == "__main__":
    run()
