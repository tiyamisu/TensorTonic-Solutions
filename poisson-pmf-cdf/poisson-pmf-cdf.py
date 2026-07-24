import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    pmf = (math.exp(-lam)*(lam**k))/math.factorial(k)

    cdf = 0.0
    for i in range(k+1):
        cdf += (math.exp(-lam)*(lam**i))/math.factorial(i)

    return (float(pmf), float(cdf))