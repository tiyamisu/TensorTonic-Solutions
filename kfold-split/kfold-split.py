import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N)

    if shuffle:
        if rng is None:
            np.random.shuffle(indices)
        else:
            rng.shuffle(indices)

    fold_sizes = np.full(k, N // k)
    fold_sizes[:N % k] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        start = current
        end = current + fold_size

        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        folds.append((train_idx, val_idx))
        current = end

    return folds