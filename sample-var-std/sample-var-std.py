import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype=float)

    variance = np.var(x, ddof=1)
    std = np.std(x, ddof=1)

    return variance, std