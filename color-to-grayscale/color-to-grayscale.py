def color_to_grayscale(image):
    """
    Returns the luminance value of every RGB pixel.
    """
    H, W = len(image), len(image[0])
    result = []
    for i in range(H):
        row = []
        for j in range(W):
            r, g, b = image[i][j][0], image[i][j][1], image[i][j][2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            row.append(round(gray, 6))
        result.append(row)
    return result
