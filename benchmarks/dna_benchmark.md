# DNA benchmark — zeta-DNA vs classical distances

Database: 5 genes (64 bp each). Queries: 500 single-base SNP variants.

| Method | Top-1 acc | Mean rank | Time/query (ms) |
|-------|----------:|----------:|----------------:|
| zeta-DNA | 99.60% | 1.00 | 3.787 |
| hamming | 99.80% | 1.00 | 0.019 |
| levenshtein | 99.80% | 1.00 | 3.010 |
| kmer-Jaccard-3 | 80.00% | 1.20 | 0.069 |
