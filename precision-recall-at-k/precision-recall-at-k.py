def precision_recall_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))

    precision = hits / k
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0

    return [precision, recall]