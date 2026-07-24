import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x)

    mean = float(np.mean(x))
    median = float(np.median(x))

    counts = Counter(x)
    max_freq = max(counts.values())
    mode = min(num for num, freq in counts.items() if freq == max_freq)

    return (mean, median, mode)