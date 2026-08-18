def robust_scaling(values):
    if not values:
        return []

    arr = sorted(values)
    n = len(arr)

    def median(a):
        if not a:
            return 0.0
        m = len(a)
        if m % 2 == 0:
            return (a[m // 2 - 1] + a[m // 2]) / 2
        return a[m // 2]

    med = median(arr)

    # Split into lower and upper halves
    if n % 2 == 0:
        lower = arr[:n // 2]
        upper = arr[n // 2:]
    else:
        lower = arr[:n // 2]
        upper = arr[n // 2 + 1:]

    q1 = median(lower)
    q3 = median(upper)

    iqr = q3 - q1

    if iqr == 0:
        return [x - med for x in values]

    return [(x - med) / iqr for x in values]