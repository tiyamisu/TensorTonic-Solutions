def moving_median(values: list, window_size: int) -> list:
    result = []

    for i in range(len(values) - window_size + 1):
        window = sorted(values[i:i + window_size])
        n = len(window)

        if n % 2 == 1:
            median = window[n // 2]
        else:
            median = (window[n // 2 - 1] + window[n // 2]) / 2

        result.append(float(median))

    return result