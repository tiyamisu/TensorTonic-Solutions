def winsorize(values, lower_pct, upper_pct):
    arr = sorted(values)
    n = len(arr)

    def percentile(p):
        k = (n - 1) * p / 100
        lower = int(k)
        upper = min(lower + 1, n - 1)
        fraction = k - lower

        return arr[lower] + fraction * (arr[upper] - arr[lower])

    lower_bound = percentile(lower_pct)
    upper_bound = percentile(upper_pct)

    return [
        max(lower_bound, min(x, upper_bound))
        for x in values
    ]