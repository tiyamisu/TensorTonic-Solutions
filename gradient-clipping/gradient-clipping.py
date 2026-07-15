import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    try:
        g = np.asarray(g, dtype=float)

        # If max_norm is non-positive, return gradients unchanged
        if max_norm <= 0:
            return g

        norm = np.linalg.norm(g)

        # No clipping needed
        if norm == 0 or norm <= max_norm:
            return g

        # Scale gradients
        scale = max_norm / norm
        return g * scale

    except:
        return None