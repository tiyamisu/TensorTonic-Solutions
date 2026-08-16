def binning(values, num_bins):
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0 for _ in values]

    width = (max_val - min_val) / num_bins

    return [
        min(int((x - min_val) / width), num_bins - 1)
        for x in values
    ]