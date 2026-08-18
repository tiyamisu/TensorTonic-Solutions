def mean_rating_imputation(ratings_matrix, mode):
    matrix = [row[:] for row in ratings_matrix]
    rows = len(matrix)
    cols = len(matrix[0])

    if mode == "user":
        for i in range(rows):
            values = [matrix[i][j] for j in range(cols) if matrix[i][j] != 0]
            mean = sum(values) / len(values) if values else 0

            for j in range(cols):
                if matrix[i][j] == 0:
                    matrix[i][j] = mean

    else:  # item mode
        for j in range(cols):
            values = [matrix[i][j] for i in range(rows) if matrix[i][j] != 0]
            mean = sum(values) / len(values) if values else 0

            for i in range(rows):
                if matrix[i][j] == 0:
                    matrix[i][j] = mean

    return matrix