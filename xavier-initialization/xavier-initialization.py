import math

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    L = math.sqrt(6 / (fan_in + fan_out))

    return [
        [w * 2 * L - L for w in row]
        for row in W
    ]