import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Update first moment
    m = beta1 * m + (1 - beta1) * grad

    # Update second moment
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # AdamW update (decoupled weight decay)
    w = w - lr * weight_decay * w - lr * m / (np.sqrt(v) + eps)

    return w, m, v