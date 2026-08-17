import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    vector = np.zeros(len(vocab), dtype=int)

    # Create word -> index mapping
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Count only words present in vocabulary
    for token in tokens:
        if token in word_to_index:
            vector[word_to_index[token]] += 1

    return vector