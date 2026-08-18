def min_max_scaling(data):
    rows = len(data)
    cols = len(data[0])

    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        column = [data[i][j] for i in range(rows)]
        mn = min(column)
        mx = max(column)

        for i in range(rows):
            if mx == mn:
                result[i][j] = 0.0
            else:
                result[i][j] = (data[i][j] - mn) / (mx - mn)

    return result