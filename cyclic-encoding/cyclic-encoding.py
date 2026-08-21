import math

def cyclic_encoding(values, period):
    result = []

    for v in values:
        theta = 2 * math.pi * v / period
        result.append([math.sin(theta), math.cos(theta)])

    return result