def rank_transform(values):
    sorted_values = sorted(values)
    ranks = {}

    i = 0

    while i < len(sorted_values):
        j = i

        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1

        average_rank = (i + 1 + j) / 2

        ranks[sorted_values[i]] = average_rank
        i = j

    return [ranks[value] for value in values]