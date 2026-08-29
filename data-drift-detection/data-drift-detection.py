def detect_drift(reference_counts: list, production_counts: list, threshold: float) -> dict:
    """
    Returns a dictionary with score and drift_detected.
    """
    reference_total = sum(reference_counts)
    production_total = sum(production_counts)

    n = max(len(reference_counts), len(production_counts))

    if reference_total == 0:
        reference_probs = [0.0] * len(reference_counts)
    else:
        reference_probs = [
            count / reference_total for count in reference_counts
        ]

    if production_total == 0:
        production_probs = [0.0] * len(production_counts)
    else:
        production_probs = [
            count / production_total for count in production_counts
        ]

    # Pad in case lengths differ.
    reference_probs += [0.0] * (n - len(reference_probs))
    production_probs += [0.0] * (n - len(production_probs))

    score = 0.5 * sum(
        abs(p - q)
        for p, q in zip(reference_probs, production_probs)
    )

    # Avoid floating-point boundary errors for strict comparison.
    score = round(score, 12)

    return {
        "score": float(score),
        "drift_detected": score > threshold
    }