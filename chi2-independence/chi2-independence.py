import numpy as np

def chi2_independence(C):
    """
    Compute chi-square statistic and expected frequencies.
    """
    C = np.asarray(C, dtype=float)

    row_sum = C.sum(axis=1, keepdims=True)
    col_sum = C.sum(axis=0, keepdims=True)
    total = C.sum()

    expected = row_sum @ col_sum / total

    chi2 = np.sum((C - expected) ** 2 / expected)

    return chi2, expected