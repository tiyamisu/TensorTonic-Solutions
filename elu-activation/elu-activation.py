import math

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []

    for val in x:
        if val > 0:
            result.append(val)
        else:
            result.append(alpha * (math.exp(val) - 1))

    return result