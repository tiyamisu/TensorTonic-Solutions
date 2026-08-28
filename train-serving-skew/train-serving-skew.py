import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    result = {}
    for feature in train_dist:
        train = np.asarray(train_dist[feature], dtype=float) + eps
        serving = np.asarray(serving_dist[feature], dtype=float) + eps
        psi = round(float(np.sum((serving - train) * np.log(serving / train))), 6)
        result[feature] = {"psi": psi, "skewed": psi >= threshold}
    return result
