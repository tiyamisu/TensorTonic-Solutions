import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    try:
        # Empty corpus
        if docs is None:
            return None
        if len(docs) == 0:
            return np.array([], dtype=float)

        N = len(docs)
        avgdl = sum(len(doc) for doc in docs) / N

        # Avoid division by zero if all documents are empty
        if avgdl == 0:
            return np.zeros(N, dtype=float)

        # Document frequencies
        df = {}
        for term in query_tokens:
            df[term] = sum(term in doc for doc in docs)

        scores = []

        for doc in docs:
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0

            for term in query_tokens:
                if tf[term] == 0:
                    continue

                idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

                numerator = tf[term] * (k1 + 1)
                denominator = tf[term] + k1 * (1 - b + b * (dl / avgdl))

                score += idf * (numerator / denominator)

            scores.append(score)

        return np.array(scores, dtype=float)

    except:
        return None