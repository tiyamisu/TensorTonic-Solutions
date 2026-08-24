def autocorrelation(series: list, max_lag: int) -> list:
    n = len(series)

    if n == 0:
        return []

    mean = sum(series) / n

    gamma0 = sum((x - mean) ** 2 for x in series)

    result = [1.0]

    for k in range(1, max_lag + 1):
        if gamma0 == 0:
            result.append(0.0)
        else:
            autocovariance = sum(
                (series[t] - mean) * (series[t + k] - mean)
                for t in range(n - k)
            )
            result.append(round(autocovariance / gamma0, 6))

    return result