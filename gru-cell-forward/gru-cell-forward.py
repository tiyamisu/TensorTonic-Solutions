def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """

    Wx = params["Wz"]
    Wr = params["Wr"]
    Wh = params["Wh"]

    Uz = params["Uz"]
    Ur = params["Ur"]
    Uh = params["Uh"]

    bz = params["bz"]
    br = params["br"]
    bh = params["bh"]

    # Convert to 2D if needed
    x, x_single = _as2d(x, Wx.shape[0])
    h_prev, h_single = _as2d(h_prev, Uz.shape[0])

    # Update gate
    z = _sigmoid(x @ Wx + h_prev @ Uz + bz)

    # Reset gate
    r = _sigmoid(x @ Wr + h_prev @ Ur + br)

    # Candidate hidden state
    h_tilde = np.tanh(
        x @ Wh + (r * h_prev) @ Uh + bh
    )

    # Final hidden state
    h = (1 - z) * h_prev + z * h_tilde

    # Return original shape if input was 1D
    if x_single and h_single:
        return h.squeeze(0)

    return h