import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list of floats.
    """
    center = size // 2

    kernel = []
    total = 0.0

    for i in range(size):
        row = []

        for j in range(size):
            x = j - center
            y = i - center

            weight = math.exp(
                -(x * x + y * y) / (2 * sigma * sigma)
            )

            row.append(weight)
            total += weight

        kernel.append(row)

    # Normalize so that all weights sum to 1.
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total

    return kernel