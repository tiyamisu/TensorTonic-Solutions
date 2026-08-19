import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    x = np.asarray(x, dtype=float)
    conv1 = np.asarray(conv1, dtype=float)
    W1_b1 = np.asarray(W1_b1, dtype=float)
    W2_b1 = np.asarray(W2_b1, dtype=float)
    W1_b2 = np.asarray(W1_b2, dtype=float)
    W2_b2 = np.asarray(W2_b2, dtype=float)
    Ws_b2 = np.asarray(Ws_b2, dtype=float)
    fc = np.asarray(fc, dtype=float)

    # Initial convolution + ReLU
    out = np.maximum(0, x @ conv1)

    # Basic Block 1
    h = np.maximum(0, out @ W1_b1)
    h = h @ W2_b1
    out = np.maximum(0, h + out)

    # Basic Block 2 with projection shortcut
    h = np.maximum(0, out @ W1_b2)
    h = h @ W2_b2

    shortcut = out @ Ws_b2

    out = np.maximum(0, h + shortcut)

    # Final fully connected classification layer
    logits = out @ fc

    return logits