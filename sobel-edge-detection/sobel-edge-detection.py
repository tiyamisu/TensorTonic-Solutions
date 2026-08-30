import math

def sobel_edges(image: list) -> list:
    """
    Returns the zero-padded Sobel gradient magnitude at every pixel.
    """
    H = len(image)
    W = len(image[0])

    Kx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    Ky = [
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ]

    output = []

    for i in range(H):
        row = []

        for j in range(W):
            gx = 0
            gy = 0

            # Center the 3x3 kernels at (i, j).
            for a in range(3):
                for b in range(3):
                    r = i + a - 1
                    c = j + b - 1

                    # Zero padding outside the image.
                    pixel = 0
                    if 0 <= r < H and 0 <= c < W:
                        pixel = image[r][c]

                    gx += Kx[a][b] * pixel
                    gy += Ky[a][b] * pixel

            magnitude = math.sqrt(gx * gx + gy * gy)
            row.append(magnitude)

        output.append(row)

    return output