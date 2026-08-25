def differencing(series: list, order: int) -> list:
    for _ in range(order):
        series = [series[i] - series[i - 1] for i in range(1, len(series))]

    return series