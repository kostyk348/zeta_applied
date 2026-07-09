# Hard benchmark: biospectral vs classical distances

Tests scenarios where the spectral method is expected to
outperform edit-distance approaches: indel robustness,
precompute speed, and sliding-window search.

---

**t-space**: linspace(10, 80, 200)

## Scenario A: Indel robustness

Database: 80 sequences (200-500 bp). Queries: 240 (indel ~30% + 2-5 point mutations).

| Method | Top-1 | Time/query (ms) |
|-------|------:|----------------:|
| spectral | 48.33% | 3.857 |
| hamming | 42.92% | 0.539 |
| levenshtein | 50.00% | 418.208 |
| jaccard-3 | 24.17% | 2.370 |

## Scenario B: Precompute speed

Database: 80 sequences (300 bp). Queries: 25.

| Method | Time/query (ms) | vs Levenshtein |
|-------|----------------:|---------------:|
| spectral | 1.149 | 0.65x |
| hamming | 1.771 | 1.00x |
| levenshtein | 1203.871 | 679.65x |

## Scenario C: Sliding-window pattern search

Long sequence: 5000 bp. Reference: 40 bp. Windows: 4961.

| Method | Time (ms) | Min position | Localization error (bp) |
|-------|---------:|-------------:|------------------------:|
| spectral | 27.3 | 3344 | 0 |
| hamming | 15.4 | 3344 | 0 |
| levenshtein | 1308.8 | 3344 | 0 |
