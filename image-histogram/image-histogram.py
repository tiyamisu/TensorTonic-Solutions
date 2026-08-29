def image_histogram(image: list) -> list:
    """
    Returns a list of intensity and count pairs.
    """
    counts = {}

    for row in image:
        for intensity in row:
            counts[intensity] = counts.get(intensity, 0) + 1

    return [
        [intensity, counts[intensity]]
        for intensity in sorted(counts)
    ]