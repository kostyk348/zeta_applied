# biospectral

[![CI](https://github.com/kostyk348/biospectral/actions/workflows/ci.yml/badge.svg)](https://github.com/kostyk348/biospectral/actions/workflows/ci.yml)


Dirichlet L-function spectral analysis of DNA and protein sequences.

A sequence over a finite alphabet is mapped to a real Dirichlet character
indexed by position.  Its L-function on the critical line,

    L_seq(t) = sum_n chi_n * n^{-1/2 - it}

encodes the sequence as a complex spectral signature.  Single-point
mutations produce measurable shifts in |L_seq|^2, enabling exact
discrimination between healthy and pathogenic variants.

## Modules

| Module | Description |
|--------|-------------|
| `biospectral.dna` | DNA/RNA mapping (A=+1, T=-2, C=+2, G=-1), L-function, SNP detection, database screening, sliding-window scan |
| `biospectral.protein` | 20-AA mapping by Grantham physicochemical groups, L-function, spectral distance, protein database screening |

## Quick start

```python
import numpy as np
from biospectral import (
    dna_to_character_seq, l_function, spectral_distance, database_screen,
    REFERENCE_GENE_DATABASE,
)

# Spectral fingerprint of a DNA sequence

[![CI](https://github.com/kostyk348/biospectral/actions/workflows/ci.yml/badge.svg)](https://github.com/kostyk348/biospectral/actions/workflows/ci.yml)

t = np.linspace(10.0, 80.0, 400)
chi = dna_to_character_seq("ATGGTGCACCTGACTCCTGAGG")
L = l_function(t, chi)

# Screen a query against the reference database

[![CI](https://github.com/kostyk348/biospectral/actions/workflows/ci.yml/badge.svg)](https://github.com/kostyk348/biospectral/actions/workflows/ci.yml)

residuals = database_screen("ATGGTGCACCTGACTCCTGTGG", REFERENCE_GENE_DATABASE)
print(f"Best match: {residuals[0][0]} (residual {residuals[0][1]:.2e})")
```

## Benchmarks

```bash
python examples/bench_dna.py          # DNA: spectral vs Hamming/Levenshtein/kmer
python examples/bench_protein.py      # Protein: spectral vs classical distances
python examples/bench_hard.py         # Indel robustness, precompute speed, sliding window
python examples/bench_viral_sw.py     # Real viral fragment ID via sliding window
python examples/bench_phylogeny.py    # Viral genome phylogeny from spectral distances
```

### DNA (500 SNP queries, 5 reference genes)

| Method | Top-1 acc | Mean rank | Time/query (ms) |
|--------|----------:|----------:|----------------:|
| spectral | 99.60% | 1.00 | 3.8 |
| Hamming | 99.80% | 1.00 | 0.02 |
| Levenshtein | 99.80% | 1.00 | 3.0 |
| kmer-Jaccard-3 | 80.00% | 1.20 | 0.07 |

### Protein (350 single-AA substitution queries, 7 reference proteins)

| Method | Top-1 acc | Mean rank | Time/query (ms) |
|--------|----------:|----------:|----------------:|
| spectral | 100.00% | 1.00 | 39.8 |
| Hamming | 100.00% | 1.00 | 0.10 |
| Levenshtein | 100.00% | 1.00 | 67.0 |
| kmer-Jaccard-3 | 100.00% | 1.00 | 0.55 |

### Viral genome fragment identification (15 real genomes from NCBI)

A 100–250 bp fragment from one of 15 diverse viral genomes (SARS-CoV-2,
SARS-CoV, MERS-CoV, Influenza A, HIV-1, Ebola, Zika, Dengue 1/2, Rabies,
Hepatitis B/C, Phi-X174, Lambda phage) is matched against the full database.
The spectral sliding-window method compares the fragment against windows of
the *same length* — the only approach that works for fragment-vs-genome
matching.

| Fragment length | top-1 | top-3 | Time/query (ms) |
|----------------:|------:|------:|----------------:|
| 100 bp | 94.7% | 100% | 807 |
| 250 bp | 96.0% | 100% | 1010 |

Edit distance (SequenceMatcher) achieves ≤8% top-1 on the same task — it fails
because normalisation over the full genome length masks the local match. The
sliding-window approach naturally avoids this issue.

### Viral genome phylogeny (15 genomes, all-vs-all spectral distance)

Spectral fingerprints of complete viral genomes produce a distance matrix
that recovers known taxonomic relationships without alignment.

| Metric | Value |
|--------|-------|
| Cophenetic correlation (UPGMA) | 0.84 |
| Intra-group / inter-group distance ratio | 0.78 |
| Closest pair | Dengue 1 ↔ Dengue 2 |
| Genomic neighbours correctly identified | SARS2↔SARS, Influenza PB2↔HA, Zika↔Dengue |

## Tests

```bash
pip install -e .[tests]
pytest tests/ -v
```

## License

Apache 2.0
