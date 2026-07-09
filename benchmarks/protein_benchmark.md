# Protein benchmark — protein-spectral vs classical distances

Database: 7 proteins (CytC_Human, Myoglobin_Human, HemA_Human, HemB_Human, INS_Human, ALBU_Human, OVA_Chicken).
Queries: 350 single-AA substitution variants.

| Method | Top-1 acc | Mean rank | Time/query (ms) |
|-------|----------:|----------:|----------------:|
| protein-spectral | 100.00% | 1.00 | 39.841 |
| hamming | 100.00% | 1.00 | 0.096 |
| levenshtein | 100.00% | 1.00 | 66.962 |
| kmer-Jaccard-3 | 100.00% | 1.00 | 0.548 |
