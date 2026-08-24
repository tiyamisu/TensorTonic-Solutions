import math

def rolling_std(values: list, window_size: int) -> list:
    if window_size <= 0 or window_size > len(values):
        return []

    result = []

    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]

        mean = sum(window) / window_size

        variance = sum(
            (x - mean) ** 2 for x in window
        ) / window_size

        result.append(round(math.sqrt(variance), 6))

    return result