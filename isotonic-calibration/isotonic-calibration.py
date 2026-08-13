def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    # Sort by predicted probability
    data = sorted(zip(cal_probs, cal_labels))
    probs = [x[0] for x in data]
    labels = [float(x[1]) for x in data]

    # Pool Adjacent Violators Algorithm (PAVA)
    blocks = []

    for i, label in enumerate(labels):
        blocks.append([i, i, label, 1])

        while len(blocks) >= 2:
            avg1 = blocks[-2][2] / blocks[-2][3]
            avg2 = blocks[-1][2] / blocks[-1][3]

            if avg1 > avg2:
                b2 = blocks.pop()
                b1 = blocks.pop()

                blocks.append([
                    b1[0],
                    b2[1],
                    b1[2] + b2[2],
                    b1[3] + b2[3]
                ])
            else:
                break

    # Fitted calibrated values
    fitted = [0.0] * len(labels)

    for start, end, total, count in blocks:
        avg = total / count
        for i in range(start, end + 1):
            fitted[i] = avg

    # Interpolate new probabilities
    result = []

    for q in new_probs:
        if q <= probs[0]:
            result.append(fitted[0])
        elif q >= probs[-1]:
            result.append(fitted[-1])
        else:
            for i in range(len(probs) - 1):
                if probs[i] <= q <= probs[i + 1]:
                    if probs[i] == probs[i + 1]:
                        result.append(fitted[i])
                    else:
                        value = fitted[i] + (
                            (q - probs[i]) /
                            (probs[i + 1] - probs[i])
                        ) * (fitted[i + 1] - fitted[i])

                        result.append(value)
                    break

    return result