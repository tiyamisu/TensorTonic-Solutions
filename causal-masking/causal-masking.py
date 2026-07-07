def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)

    Return: masked scores (same shape, dtype=float)
    """

    scores = np.array(scores, dtype=float, copy=True)

    T = scores.shape[-1]

    # True for positions above the main diagonal
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    # Apply mask to the last two dimensions
    scores[..., mask] = mask_value

    return scores