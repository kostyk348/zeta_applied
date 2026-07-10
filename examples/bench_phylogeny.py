"""Viral genome phylogeny from spectral fingerprints.

Downloads real viral genomes, computes all-vs-all spectral distances,
builds a distance matrix, and compares clustering accuracy to
known taxonomic relationships.
"""

import sys
import time
import socket
import ssl
import http.client
from typing import Dict, List, Tuple

import dns.resolver
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from biospectral.dna import dna_to_character_seq, l_function

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

# Known taxonomic group for each virus (for sanity checking)
TAXONOMY: Dict[str, str] = {
    "SARS-CoV-2":    "Coronaviridae",
    "SARS-CoV":      "Coronaviridae",
    "MERS-CoV":      "Coronaviridae",
    "Influenza_PB2": "Orthomyxoviridae",
    "Influenza_HA":  "Orthomyxoviridae",
    "HIV-1":         "Retroviridae",
    "Ebola":         "Filoviridae",
    "Zika":          "Flaviviridae",
    "Dengue_1":      "Flaviviridae",
    "Dengue_2":      "Flaviviridae",
    "Rabies":        "Rhabdoviridae",
    "Hepatitis_B":   "Hepadnaviridae",
    "Hepatitis_C":   "Flaviviridae",
    "Phi-X174":      "Microviridae",
    "Lambda_phage":  "Siphoviridae",
}


def spectral_distance(fp_a: np.ndarray, fp_b: np.ndarray, t_space: np.ndarray) -> float:
    n = min(len(fp_a), len(fp_b))
    return float(np.trapezoid(np.abs(fp_a[:n] - fp_b[:n]) ** 2, t_space[:n]))


def all_vs_all_matrix(names: List[str], fingerprints: Dict[str, np.ndarray],
                       t_space: np.ndarray) -> np.ndarray:
    """Compute pairwise spectral distance matrix."""
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = spectral_distance(fingerprints[names[i]], fingerprints[names[j]], t_space)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def run_benchmark() -> None:
    print("=" * 72)
    print("Viral Genome Phylogeny from Spectral Fingerprints")
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

    if len(names) < 5:
        print("Too few genomes, aborting.")
        return

    # 2. Precompute fingerprints
    T_SPACE = np.linspace(10.0, 80.0, 400)
    print(f"\n[2] Precomputing spectral fingerprints (|t|={len(T_SPACE)})...")
    t0 = time.perf_counter()
    fingerprints: Dict[str, np.ndarray] = {}
    for name in names:
        fingerprints[name] = l_function(T_SPACE, dna_to_character_seq(genomes[name]))
    print(f"  Done in {(time.perf_counter()-t0)*1000:.1f} ms")

    # 3. All-vs-all spectral distance
    print("\n[3] All-vs-all spectral distance matrix...")
    t0 = time.perf_counter()
    spec_mat = all_vs_all_matrix(names, fingerprints, T_SPACE)
    print(f"  Done in {time.perf_counter()-t0:.2f} s")
    print(f"  Matrix shape: {spec_mat.shape}")

    # 4. Clustering quality (UPGMA)
    print("\n[4] UPGMA clustering quality...")
    condensed = squareform(spec_mat, checks=False)
    Z = linkage(condensed, method="average")
    coph, _ = cophenet(Z, condensed)
    print(f"  Cophenetic correlation: {coph:.4f}  (1.0 = perfect tree fit)")

    # 5. Intra-family vs inter-family distances
    print("\n[5] Taxonomic group separation...")
    groups: Dict[str, List[str]] = {}
    for name in names:
        group = TAXONOMY.get(name, "Unknown")
        groups.setdefault(group, []).append(name)

    # Mean intra-group distance vs mean inter-group for each family with ≥2 members
    intra_dists: List[float] = []
    inter_dists: List[float] = []
    for g, members in groups.items():
        if len(members) < 2:
            continue
        # Intra
        intra = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                intra.append(spec_mat[names.index(members[i]), names.index(members[j])])
        intra_dists.append(np.mean(intra))
        # Inter: mean distance from group members to ALL others
        other_indices = [names.index(n) for n in names if n not in members]
        if other_indices:
            member_indices = [names.index(n) for n in members]
            inter = [spec_mat[mi, oi] for mi in member_indices for oi in other_indices]
            inter_dists.append(np.mean(inter))
        print(f"  {g:20s}  intra={intra_dists[-1]:.2e}  inter={inter_dists[-1]:.2e}  "
              f"ratio={intra_dists[-1]/inter_dists[-1]:.4f}")

    ratio_overall = np.mean(intra_dists) / np.mean(inter_dists) if inter_dists else 0
    print(f"\n  Overall intra/inter ratio: {ratio_overall:.4f}  "
          f"(<1.0 = groups are closer within than across)")

    # 6. Pairwise distance table (most/least similar)
    print("\n[6] Closest and most distant pairs:")
    triu = [(i, j, spec_mat[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    triu.sort(key=lambda x: x[2])
    print("  Closest 5:")
    for i, j, d in triu[:5]:
        g1, g2 = TAXONOMY.get(names[i], "?"), TAXONOMY.get(names[j], "?")
        print(f"    {names[i]:20s} <-> {names[j]:20s}  d={d:.4e}  ({g1})")
    print("  Most distant 5:")
    for i, j, d in triu[-5:]:
        g1, g2 = TAXONOMY.get(names[i], "?"), TAXONOMY.get(names[j], "?")
        print(f"    {names[i]:20s} <-> {names[j]:20s}  d={d:.4e}  ({g1}, {g2})")

    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()
