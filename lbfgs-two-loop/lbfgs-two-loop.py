def dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """

    m = len(s_list)

    if m == 0:
        return [-g for g in grad]

    q = grad[:]
    alpha = [0.0] * m
    rho = [0.0] * m

    # First loop (backward)
    for i in range(m - 1, -1, -1):
        rho[i] = 1.0 / dot(y_list[i], s_list[i])
        alpha[i] = rho[i] * dot(s_list[i], q)
        q = [qj - alpha[i] * yj for qj, yj in zip(q, y_list[i])]

    # Initial scaling
    sy = dot(s_list[-1], y_list[-1])
    yy = dot(y_list[-1], y_list[-1])
    gamma = sy / yy if yy != 0 else 1.0

    r = [gamma * qi for qi in q]

    # Second loop (forward)
    for i in range(m):
        beta = rho[i] * dot(y_list[i], r)
        r = [
            rj + s_list[i][j] * (alpha[i] - beta)
            for j, rj in enumerate(r)
        ]

    # Search direction
    return [-x for x in r]