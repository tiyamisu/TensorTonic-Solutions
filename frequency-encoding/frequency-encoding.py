def frequency_encoding(values):
    frequency = {}
    n = len(values)

    for value in values:
        frequency[value] = frequency.get(value, 0) + 1

    return [frequency[value] / n for value in values]