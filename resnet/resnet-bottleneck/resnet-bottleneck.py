import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    W3 = np.asarray(W3)

    # Main path
    h1 = np.maximum(0, x @ W1)
    h2 = np.maximum(0, h1 @ W2)
    main = h2 @ W3

    # Shortcut
    if Ws is None:
        shortcut = x
    else:
        Ws = np.asarray(Ws)
        shortcut = x @ Ws

    # Add shortcut, then final ReLU
    out = np.maximum(0, main + shortcut)

    return out