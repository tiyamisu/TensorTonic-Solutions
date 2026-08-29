def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    """
    Returns a two-dimensional list.
    """
    H = len(image)
    W = len(image[0])

    Kh = len(kernel)
    Kw = len(kernel[0])

    # Output dimensions.
    out_h = (H + 2 * padding - Kh) // stride + 1
    out_w = (W + 2 * padding - Kw) // stride + 1

    output = []

    for i in range(out_h):
        row = []

        for j in range(out_w):
            total = 0

            # Top-left corner in the original image
            start_i = i * stride - padding
            start_j = j * stride - padding

            for a in range(Kh):
                for b in range(Kw):
                    img_i = start_i + a
                    img_j = start_j + b

                    # Zero padding.
                    if 0 <= img_i < H and 0 <= img_j < W:
                        total += image[img_i][img_j] * kernel[a][b]

            row.append(total)

        output.append(row)

    return output