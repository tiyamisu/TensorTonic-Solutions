def seasonal_average(series: list, period: int) -> list:
    averages = []

    for p in range(period):
        values = series[p::period]
        averages.append(sum(values) / len(values))

    return averages