def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    if not series:
        return []

    if len(series) == 1:
        return [float(series[0])]

    level = float(series[0])
    trend = float(series[1] - series[0])

    result = [level]

    for i in range(1, len(series)):
        previous_level = level

        level = alpha * series[i] + (1 - alpha) * (previous_level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend

        result.append(level)

    return result