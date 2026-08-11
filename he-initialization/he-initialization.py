import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    L = math.sqrt(6 / fan_in)

    return [
        [w * 2 * L - L for w in row]
        for row in W
    ]