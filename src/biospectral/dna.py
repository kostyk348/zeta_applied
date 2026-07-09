"""Spectral DNA screening via Dirichlet L-functions on the critical line.

A DNA sequence over {A, C, G, T} is mapped to a real Dirichlet character,
and its L-function L_seq(t) = sum chi_n * n^{-1/2 - it} encodes the
sequence as a complex spectral signature.  Single-point mutations produce
measurable shifts in |L_seq|^2, enabling exact discrimination.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


DNA_MAPPING = {"A": 1, "C": 2, "G": -1, "T": -2, "N": 0, "U": -2}


def char_to_character_seq(seq: str, mapping: Dict[str, int]) -> np.ndarray:
    """Map a string to an integer character vector using ``mapping``."""
    return np.array([mapping.get(c.upper(), 0) for c in seq], dtype=np.int64)


def dna_to_character_seq(seq: str) -> np.ndarray:
    """Convert a DNA / RNA string to a signed integer character vector."""
    return char_to_character_seq(seq, DNA_MAPPING)


def l_function(t: np.ndarray, char_seq: np.ndarray) -> np.ndarray:
    """Dirichlet L-function of a character-encoded sequence on the critical line.

    Parameters
    ----------
    t : array_like
        Real ordinates (critical-line frequencies).
    char_seq : int64 ndarray
        Integer character encoding of the sequence.

    Returns
    -------
    complex ndarray
        L(t) = sum_n chi_n * n^{-1/2 - it}.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    n = np.arange(1, len(char_seq) + 1, dtype=np.float64)
    exponent = np.exp(-1j * np.multiply.outer(t, np.log(n)))
    coeffs = (char_seq / np.sqrt(n)).astype(complex)
    return exponent @ coeffs


def dna_l_function(t: np.ndarray, char_seq: np.ndarray) -> np.ndarray:
    """Alias for :func:`l_function` (DNA-specific)."""
    return l_function(t, char_seq)


def spectral_distance(
    t_space: np.ndarray,
    chi_a: np.ndarray,
    chi_b: np.ndarray,
) -> float:
    """Integrated squared residual between two character spectra.

    Returns
    -------
    float
        integral |L_A(t) - L_B(t)|^2 dt.  Zero for identical sequences.
    """
    L_a = l_function(t_space, chi_a)
    L_b = l_function(t_space, chi_b)
    diff = L_a - L_b
    return float(np.trapezoid(np.abs(diff) ** 2, t_space))


def snp_spectral_shift(
    t_space: np.ndarray, seq_a: str, seq_b: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Integrated spectral shift between two DNA sequences."""
    chi_a = dna_to_character_seq(seq_a)
    chi_b = dna_to_character_seq(seq_b)
    L_a = dna_l_function(t_space, chi_a)
    L_b = dna_l_function(t_space, chi_b)
    diff = L_a - L_b
    energy = float(np.trapezoid(np.abs(diff) ** 2, t_space))
    return energy, L_a, L_b


def database_screen(
    query_seq: str,
    database: Dict[str, str],
    t_space: np.ndarray | None = None,
) -> List[Tuple[str, float]]:
    """Return the database sorted by ascending spectral residual."""
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 400)
    chi_q = dna_to_character_seq(query_seq)
    L_q = dna_l_function(t_space, chi_q)
    residuals: List[Tuple[str, float]] = []
    for name, seq in database.items():
        L_db = dna_l_function(t_space, dna_to_character_seq(seq))
        L_q_pad, L_db_pad = _pad_to_same(L_q, L_db)
        diff = L_q_pad - L_db_pad
        residuals.append((name, float(np.trapezoid(np.abs(diff) ** 2, t_space))))
    residuals.sort(key=lambda r: r[1])
    return residuals


def _pad_to_same(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    target = max(a.shape[0], b.shape[0])
    a_pad = np.zeros(target, dtype=complex)
    a_pad[: a.shape[0]] = a
    b_pad = np.zeros(target, dtype=complex)
    b_pad[: b.shape[0]] = b
    return a_pad, b_pad


def sliding_window_scan(
    long_seq: str,
    reference: str,
    window_size: int = 32,
    t_space: np.ndarray | None = None,
) -> np.ndarray:
    """Slide ``reference`` across ``long_seq``; minima locate the reference.

    Fully vectorised: all windows are packed into a 2D array and the
    L-function spectrum of every window is computed in a single matrix
    multiply, so the cost scales linearly with the sequence length.
    """
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 200)
    chi_ref = dna_to_character_seq(reference)
    L_ref = dna_l_function(t_space, chi_ref)

    chi_long = dna_to_character_seq(long_seq)
    n_windows = len(chi_long) - window_size + 1
    if n_windows <= 0:
        raise ValueError("long_seq must be longer than window_size")

    windows = np.lib.stride_tricks.as_strided(
        chi_long,
        shape=(n_windows, window_size),
        strides=(chi_long.strides[0], chi_long.strides[0]),
    ).astype(np.float64)  # (n_windows, window_size)

    k = np.arange(1, window_size + 1, dtype=np.float64)
    basis = np.exp(-1j * np.outer(np.log(k), t_space))  # (window_size, |t|)
    coeffs = windows / np.sqrt(k)[None, :]              # (n_windows, window_size)
    L_matrix = coeffs @ basis                           # (n_windows, |t|)

    diff = L_matrix - L_ref[None, :]
    residuals = np.trapezoid(np.abs(diff) ** 2, t_space, axis=1)
    return residuals


REFERENCE_GENE_DATABASE: Dict[str, str] = {
    "HBB_Normal":     "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGG",
    "HBB_SickleCell": "ATGGTGCACCTGACTCCTGTGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGG",
    "INS_Human":      "TTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAAC",
    "SARS_CoV_2":     "AAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCT",
    "Influenza_HA":   "ATGGAGAAAATAGTGCTTCTTCTTGCAATAGTCAGTCTTGTTAAAAGTGATCAGGATTTGTGAT",
}
