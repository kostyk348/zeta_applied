"""Spectral protein screening via Dirichlet L-functions.

Extends the DNA spectral method to the 20-letter amino acid alphabet.
Amino acids are mapped to Dirichlet characters based on standard
physicochemical classification (Grantham groups), and the L-function
spectrum is used as a sequence fingerprint.

References
----------
Grantham, R. (1974). "Amino acid difference formula to help explain
protein evolution." Science, 185(4154), 862-864.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .dna import _pad_to_same, dna_l_function

__all__ = [
    "AA_MAPPING",
    "protein_to_character_seq",
    "protein_l_function",
    "protein_spectral_distance",
    "protein_database_screen",
    "sliding_window_scan_protein",
]


# Amino acid -> Dirichlet character based on Grantham physicochemical groups:
#   +2 : non-polar hydrophobic (aliphatic + aromatic + proline)
#   +1 : polar uncharged
#   -1 : basic (positive charge at pH 7)
#   -2 : acidic (negative charge at pH 7)
#    0 : ambiguous / special (selenocysteine, stop, etc.)
AA_MAPPING: Dict[str, int] = {
    # Non-polar hydrophobic
    "A": 2,  # Alanine
    "V": 2,  # Valine
    "I": 2,  # Isoleucine
    "L": 2,  # Leucine
    "M": 2,  # Methionine
    "F": 2,  # Phenylalanine
    "W": 2,  # Tryptophan
    "P": 2,  # Proline
    # Polar uncharged
    "G": 1,  # Glycine
    "S": 1,  # Serine
    "T": 1,  # Threonine
    "C": 1,  # Cysteine
    "Y": 1,  # Tyrosine
    "N": 1,  # Asparagine
    "Q": 1,  # Glutamine
    # Basic (positive charge)
    "K": -1,  # Lysine
    "R": -1,  # Arginine
    "H": -1,  # Histidine
    # Acidic (negative charge)
    "D": -2,  # Aspartic acid
    "E": -2,  # Glutamic acid
    # Ambiguous
    "B": 0,  # Asparagine or aspartic acid
    "Z": 0,  # Glutamine or glutamic acid
    "X": 0,  # Unknown
    "*": 0,  # Stop
    "-": 0,  # Gap
}


def protein_to_character_seq(seq: str) -> np.ndarray:
    """Convert a protein (amino acid) string to a signed integer character vector."""
    return np.array([AA_MAPPING.get(c.upper(), 0) for c in seq], dtype=np.int64)


def protein_l_function(t: np.ndarray, char_seq: np.ndarray) -> np.ndarray:
    """Dirichlet L-function of a protein sequence on the critical line.

    Parameters
    ----------
    t : array_like
        Real ordinates on the critical line.
    char_seq : int64 ndarray
        Dirichlet-character encoding of a protein sequence.

    Returns
    -------
    complex ndarray
        :math:`L_{\\text{seq}}(t) = \\sum_n \\chi_n n^{-1/2 - i t}`.
    """
    return dna_l_function(t, char_seq)


def protein_spectral_distance(
    seq_a: str,
    seq_b: str,
    t_space: np.ndarray | None = None,
) -> float:
    """Integrated squared residual between two protein L-function spectra.

    Parameters
    ----------
    seq_a, seq_b : str
        Amino acid sequences.
    t_space : optional array
        Critical-line ordinates (defaults to ``linspace(10, 80, 400)``).

    Returns
    -------
    float
        :math:`\\int |L_A(t) - L_B(t)|^2 dt`.  Zero for identical sequences.
    """
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 400)
    chi_a = protein_to_character_seq(seq_a)
    chi_b = protein_to_character_seq(seq_b)
    L_a = protein_l_function(t_space, chi_a)
    L_b = protein_l_function(t_space, chi_b)
    diff = L_a - L_b
    return float(np.trapezoid(np.abs(diff) ** 2, t_space))


def protein_database_screen(
    query_seq: str,
    database: Dict[str, str],
    t_space: np.ndarray | None = None,
) -> List[Tuple[str, float]]:
    """Return the protein database sorted by ascending spectral residual.

    Parameters
    ----------
    query_seq : str
        Amino acid query sequence.
    database : dict
        Mapping ``name -> sequence``.
    t_space : optional array
        Critical-line ordinates (defaults to ``linspace(10, 80, 400)``).

    Returns
    -------
    list of (name, residual_energy)
        Best match first.
    """
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 400)
    chi_q = protein_to_character_seq(query_seq)
    L_q = protein_l_function(t_space, chi_q)
    residuals: List[Tuple[str, float]] = []
    for name, seq in database.items():
        L_db = protein_l_function(t_space, protein_to_character_seq(seq))
        L_q_pad, L_db_pad = _pad_to_same(L_q, L_db)
        diff = L_q_pad - L_db_pad
        residuals.append((name, float(np.trapezoid(np.abs(diff) ** 2, t_space))))
    residuals.sort(key=lambda r: r[1])
    return residuals


def sliding_window_scan_protein(
    long_seq: str,
    reference: str,
    window_size: int = 25,
    t_space: np.ndarray | None = None,
) -> np.ndarray:
    """Slide ``reference`` across ``long_seq``; minima locate the reference.

    Fully vectorised: all windows are packed into a 2D array and the
    L-function spectrum of every window is computed in a single matrix
    multiply.
    """
    if t_space is None:
        t_space = np.linspace(10.0, 80.0, 200)
    chi_ref = protein_to_character_seq(reference)
    L_ref = protein_l_function(t_space, chi_ref)

    chi_long = protein_to_character_seq(long_seq)
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


# ---------------------------------------------------------------------------
# Reference protein database: small, well-known proteins with various lengths
# and functions.  Used for benchmarking and demo.
# ---------------------------------------------------------------------------

REFERENCE_PROTEIN_DATABASE: Dict[str, str] = {
    "CytC_Human":       "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",
    "Myoglobin_Human":  "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG",
    "HemA_Human":       "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "HemB_Human":       "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    "INS_Human":        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "ALBU_Human":       "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL",
    "OVA_Chicken":      "MSIGAASMEFCFDVFKELKVHHANENIFYCPIAIMSALAMVYLGAKDSTRTQINKVVRFDKLPGFGDSIEAQCGTSVNVHSSLRDILNQITKPNDVYSFSLASRLYAEERYPILPEYLQCVKELYRGGLEPINFQTAADQARELINSWVESQTNGIIRNVLQPSSVDSQTAMVLVNAIVFKGLWEKAFKDEDTQAMPFRVTEQESKPVQMMYQIGLFRVASMASEKMKILELPFASGDMSMLVLLPDEVSGLEQLESLINFEKLTEWTSSNVMEERKIKVYLPRMKMEEKYNLTSVLMAMGITDVFSSSANLSGISSAESLKISQAVHAAHAEINEAGREVVGSAEAGVDAASVSEEFRADHPFLFCIKHIATNAVLFFGRCVSP",
}

REFERENCE_PROTEIN_NAMES: Dict[str, str] = {
    "CytC_Human":      "Cytochrome C (human, 105 AA)",
    "Myoglobin_Human": "Myoglobin (human, 154 AA)",
    "HemA_Human":      "Hemoglobin subunit alpha (human, 141 AA)",
    "HemB_Human":      "Hemoglobin subunit beta (human, 146 AA)",
    "INS_Human":       "Preproinsulin (human, 110 AA)",
    "ALBU_Human":      "Serum albumin (human, 585 AA)",
    "OVA_Chicken":     "Ovalbumin (chicken, 386 AA)",
}
