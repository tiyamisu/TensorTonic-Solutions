import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """

    # Tokenize
    docs = [doc.split() for doc in documents]
    N = len(docs)

    # Vocabulary (sorted for deterministic output)
    vocabulary = sorted(set(word for doc in docs for word in doc))

    # Document frequency
    df = {}
    for word in vocabulary:
        df[word] = sum(word in doc for doc in docs)

    # IDF
    idf = {word: math.log(N / df[word]) for word in vocabulary}

    # TF-IDF matrix
    tfidf_matrix = []

    for doc in docs:
        counts = Counter(doc)
        total = len(doc)

        row = []
        for word in vocabulary:
            tf = counts[word] / total if total > 0 else 0
            row.append(tf * idf[word])

        tfidf_matrix.append(row)

    return np.array(tfidf_matrix, dtype=float), vocabulary