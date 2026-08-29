def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns a two-dimensional list with shape (new_h, new_w).
    """
    H = len(image)
    W = len(image[0])

    output = []

    for i in range(new_h):

        # Corner-aligned source y coordinate.
        if new_h == 1:
            y = 0.0
        else:
            y = i * (H - 1) / (new_h - 1)

        y0 = int(y)
        y1 = min(y0 + 1, H - 1)
        dy = y - y0

        row = []

        for j in range(new_w):

            # Corner-aligned source x coordinate.
            if new_w == 1:
                x = 0.0
            else:
                x = j * (W - 1) / (new_w - 1)

            x0 = int(x)
            x1 = min(x0 + 1, W - 1)
            dx = x - x0

            # Horizontal interpolation on top row.
            v0 = (
                image[y0][x0] * (1 - dx)
                + image[y0][x1] * dx
            )

            # Horizontal interpolation on bottom row.
            v1 = (
                image[y1][x0] * (1 - dx)
                + image[y1][x1] * dx
            )

            # Vertical interpolation.
            value = v0 * (1 - dy) + v1 * dy

            row.append(value)

        output.append(row)

    return output