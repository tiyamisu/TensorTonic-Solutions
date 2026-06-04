import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """

    predictions = np.array(predictions)
    result = []

    for i in range(predictions.shape[1]):
        votes = predictions[:, i]
        counts = np.bincount(votes)
        result.append(int(np.argmax(counts)))

    return result