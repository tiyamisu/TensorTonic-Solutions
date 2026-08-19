import numpy as np

def conv_block(x, W1, W2, Ws):
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    Ws = np.asarray(Ws)

    # Main path
    h = np.maximum(0, x @ W1)
    z = h @ W2

    # Projection shortcut
    s = x @ Ws

    # Add shortcut, then ReLU
    y = np.maximum(0, z + s)

    return y