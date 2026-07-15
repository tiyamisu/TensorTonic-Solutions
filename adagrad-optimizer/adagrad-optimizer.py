import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    try:
        w = np.asarray(w, dtype=float)
        g = np.asarray(g, dtype=float)
        G = np.asarray(G, dtype=float)

        if w.shape != g.shape or w.shape != G.shape:
            return None

        # Accumulate squared gradients
        G_new = G + g ** 2

        # Parameter update
        w_new = w - lr * g / np.sqrt(G_new + eps)

        return w_new, G_new

    except:
        return None