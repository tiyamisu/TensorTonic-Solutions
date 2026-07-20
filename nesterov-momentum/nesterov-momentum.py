import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    w = np.array(w, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    # Look-ahead position
    w_look = w - momentum * v

    # Update velocity (grad is already the look-ahead gradient)
    v = momentum * v + lr * grad

    # Update weights
    w = w - v

    return w.tolist(), v.tolist()