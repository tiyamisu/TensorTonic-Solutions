import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    if not relevance_scores:
        return 0.0

    k = min(k, len(relevance_scores))

    def dcg(scores):
        return sum(
            (2 ** score - 1) / math.log2(i + 2)
            for i, score in enumerate(scores[:k])
        )

    actual_dcg = dcg(relevance_scores)

    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return float(actual_dcg / ideal_dcg)