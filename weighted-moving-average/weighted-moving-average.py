def weighted_moving_average(values: list, weights: list) -> list:
    k = len(weights)
    weight_sum = sum(weights)

    if k == 0 or weight_sum == 0 or k > len(values):
        return []

    result = []

    for i in range(len(values) - k + 1):
        weighted_sum = sum(
            values[i + j] * weights[j]
            for j in range(k)
        )

        result.append(round(weighted_sum / weight_sum, 6))

    return result