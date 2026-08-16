def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """

    if not recommendations:
        return 0.0

    hits = 0

    for recs, relevant in zip(recommendations, ground_truth):
        top_k = set(recs[:k])
        relevant_items = set(relevant)

        if top_k & relevant_items:
            hits += 1

    return float(hits / len(recommendations))