import numpy as np

def cohens_kappa(rater1, rater2):
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)

    n = len(rater1)

    # Observed agreement
    po = np.mean(rater1 == rater2)

    # All possible labels
    labels = set(rater1) | set(rater2)

    # Expected agreement
    pe = 0.0
    for label in labels:
        p1 = np.sum(rater1 == label) / n
        p2 = np.sum(rater2 == label) / n
        pe += p1 * p2

    # If denominator is zero, both raters agree on every sample
    if 1 - pe == 0:
        return 1.0

    return float((po - pe) / (1 - pe))