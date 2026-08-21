def polynomial_features(values, degree):
    result = []

    for x in values:
        row = []
        for power in range(degree + 1):
            row.append(x ** power)
        result.append(row)

    return result