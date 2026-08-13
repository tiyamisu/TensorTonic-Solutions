def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    n = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        lower = b / n_bins
        upper = (b + 1) / n_bins

        # Last bin includes probability 1.0
        if b == n_bins - 1:
            indices = [
                i for i, p in enumerate(y_pred)
                if lower <= p <= upper
            ]
        else:
            indices = [
                i for i, p in enumerate(y_pred)
                if lower <= p < upper
            ]

        if len(indices) == 0:
            continue

        accuracy = sum(y_true[i] for i in indices) / len(indices)
        confidence = sum(y_pred[i] for i in indices) / len(indices)

        ece += (len(indices) / n) * abs(accuracy - confidence)

    return float(ece)