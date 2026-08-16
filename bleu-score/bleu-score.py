import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """

    if not candidate or not reference:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        # Generate n-grams
        candidate_ngrams = Counter(
            tuple(candidate[i:i+n])
            for i in range(len(candidate) - n + 1)
        )

        reference_ngrams = Counter(
            tuple(reference[i:i+n])
            for i in range(len(reference) - n + 1)
        )

        # Total candidate n-grams
        total = sum(candidate_ngrams.values())

        if total == 0:
            return 0.0

        # Modified precision
        clipped_count = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in candidate_ngrams.items()
        )

        precision = clipped_count / total

        if precision == 0:
            return 0.0

        precisions.append(precision)

    # Brevity penalty
    c = len(candidate)
    r = len(reference)

    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    # Geometric mean of precisions
    log_precision = sum(math.log(p) for p in precisions) / max_n

    return float(bp * math.exp(log_precision))