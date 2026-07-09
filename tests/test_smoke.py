"""Smoke tests for biospectral.

Run with `pytest tests/` after `pip install -e .`.
"""
from __future__ import annotations

import numpy as np
import pytest

from biospectral import dna as _dna
from biospectral import protein as _protein
from biospectral.dna import (
    DNA_MAPPING,
    char_to_character_seq,
    dna_to_character_seq,
    l_function,
    spectral_distance,
    snp_spectral_shift,
    database_screen,
    sliding_window_scan,
    REFERENCE_GENE_DATABASE,
)
from biospectral.protein import (
    AA_MAPPING,
    protein_to_character_seq,
    protein_l_function,
    protein_spectral_distance,
    protein_database_screen,
    sliding_window_scan_protein,
    REFERENCE_PROTEIN_DATABASE,
    REFERENCE_PROTEIN_NAMES,
)


def test_dna_mapping():
    assert DNA_MAPPING["A"] == 1
    assert DNA_MAPPING["C"] == 2
    assert DNA_MAPPING["G"] == -1
    assert DNA_MAPPING["T"] == -2


def test_dna_to_character_seq():
    chi = dna_to_character_seq("ACGTN")
    assert chi.tolist() == [1, 2, -1, -2, 0]


def test_l_function_shape():
    t = np.linspace(10.0, 80.0, 64)
    chi = dna_to_character_seq("ACGTACGT")
    L = l_function(t, chi)
    assert L.shape == (64,)
    assert np.all(np.isfinite(L))


def test_spectral_distance_zero_for_identical():
    chi = char_to_character_seq("ACGTACGT", DNA_MAPPING)
    t = np.linspace(10.0, 80.0, 256)
    d = spectral_distance(t, chi, chi)
    assert d < 1e-10


def test_spectral_distance_positive_for_diff():
    t = np.linspace(10.0, 80.0, 256)
    chi_a = char_to_character_seq("ACGTACGTACGTACGTACGTACGTACGTACGT", DNA_MAPPING)
    chi_b = char_to_character_seq("ACGTACGTTCGTACGTACGTACGTACGTACGT", DNA_MAPPING)
    d = spectral_distance(t, chi_a, chi_b)
    assert d > 0.0


def test_snp_spectral_shift():
    t = np.linspace(10.0, 80.0, 256)
    e, _, _ = snp_spectral_shift(t, "ACGTACGTACGTACGTACGTACGTACGTACGT",
                                    "ACGTACGTTCGTACGTACGTACGTACGTACGT")
    assert e > 0.0


def test_database_screen():
    db = {"A": "ACGT", "B": "ACGG"}
    result = database_screen("ACGT", db)
    assert result[0][0] == "A"


def test_sliding_window_scan():
    long_seq = "ACGTACGTACGTACGT"
    ref = "ACGT"
    res = sliding_window_scan(long_seq, ref, window_size=4, t_space=np.linspace(10, 80, 64))
    assert res.shape == (13,)
    assert np.isfinite(res).all()


def test_protein_mapping():
    assert AA_MAPPING["A"] == 2
    assert AA_MAPPING["K"] == -1
    assert AA_MAPPING["D"] == -2
    assert AA_MAPPING["G"] == 1
    assert AA_MAPPING["X"] == 0


def test_protein_to_character_seq():
    chi = protein_to_character_seq("ACDEFGHIKLMNPQRSTVWY")
    assert len(chi) == 20
    assert chi.dtype == np.int64


def test_protein_l_function_shape():
    t = np.linspace(10.0, 80.0, 64)
    chi = protein_to_character_seq("MGDVEKGKKIFIMKCSQCHTVEK")
    L = protein_l_function(t, chi)
    assert L.shape == (64,)
    assert np.all(np.isfinite(L))


def test_protein_spectral_distance_zero_for_identical():
    seq = "MGDVEKGKKIFIMKCSQCHTVEK"
    d = protein_spectral_distance(seq, seq)
    assert d < 1e-10


def test_protein_spectral_distance_positive_for_diff():
    d = protein_spectral_distance("MGDVEKGKKIFIMKCSQCHTVEK",
                                   "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNL")
    assert d > 0.0


def test_protein_database_screen():
    db = {"A": "MGDVEK", "B": "MGDVKK"}
    result = protein_database_screen("MGDVEK", db)
    assert result[0][0] == "A"


def test_protein_ref_database():
    assert len(REFERENCE_PROTEIN_DATABASE) == 7
    assert "CytC_Human" in REFERENCE_PROTEIN_DATABASE


def test_protein_ref_names():
    assert "CytC_Human" in REFERENCE_PROTEIN_NAMES


def test_ref_gene_database():
    assert "HBB_Normal" in REFERENCE_GENE_DATABASE
    assert len(REFERENCE_GENE_DATABASE) == 5


def test_char_to_character_seq_generic():
    custom_map = {"X": 10, "Y": -10}
    chi = char_to_character_seq("XYX", custom_map)
    assert chi.tolist() == [10, -10, 10]
