def linear_interpolation(values):
    result = values[:]

    for i in range(len(result)):
        if result[i] is None:
            left = i - 1
            while result[left] is None:
                left -= 1

            right = i + 1
            while result[right] is None:
                right += 1

            result[i] = result[left] + (
                (i - left) / (right - left)
            ) * (result[right] - result[left])

    return result