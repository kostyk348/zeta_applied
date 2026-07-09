"""biospectral — Dirichlet L-function spectral analysis of biological sequences.

The library encodes DNA or amino-acid sequences as real Dirichlet
characters and computes their L-function on the critical line.  The
resulting complex spectrum serves as a fingerprint for exact sequence
discrimination, mutation detection, and protein database screening.
"""

from .dna import (
    char_to_character_seq,
    dna_to_character_seq,
    l_function,
    dna_l_function,
    spectral_distance,
    snp_spectral_shift,
    database_screen,
    sliding_window_scan,
    REFERENCE_GENE_DATABASE,
)
from .protein import (
    protein_to_character_seq,
    protein_l_function,
    protein_spectral_distance,
    protein_database_screen,
    sliding_window_scan_protein,
    REFERENCE_PROTEIN_DATABASE,
    REFERENCE_PROTEIN_NAMES,
)

__version__ = "1.0.0"
