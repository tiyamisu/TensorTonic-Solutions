import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x, dtype=float)

    mean = np.mean(x)
    s = np.std(x, ddof=1)
    n = len(x)

    t = (mean - mu0) / (s / np.sqrt(n))

    return t