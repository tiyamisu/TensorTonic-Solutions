def interaction_features(X):
    result = []

    for row in X:
        new_row = row.copy()

        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                new_row.append(row[i] * row[j])

        result.append(new_row)

    return result