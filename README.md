# biospectral

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
t = np.linspace(10.0, 80.0, 400)
chi = dna_to_character_seq("ATGGTGCACCTGACTCCTGAGG")
L = l_function(t, chi)

# Screen a query against the reference database
residuals = database_screen("ATGGTGCACCTGACTCCTGTGG", REFERENCE_GENE_DATABASE)
print(f"Best match: {residuals[0][0]} (residual {residuals[0][1]:.2e})")
```

## Benchmarks

```bash
python examples/bench_dna.py     # DNA: spectral vs Hamming/Levenshtein/kmer
python examples/bench_protein.py # Protein: spectral vs classical distances
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

## Tests

```bash
pip install -e .[tests]
pytest tests/ -v
```

## License

Apache 2.0
