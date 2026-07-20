import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)

    # Step 1: Update running average of squared gradients
    E_grad_sq = rho * E_grad_sq + (1 - rho) * (grad ** 2)

    # Step 2: Compute parameter update
    delta = - (np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq + eps)) * grad

    # Step 3: Update running average of squared updates
    E_update_sq = rho * E_update_sq + (1 - rho) * (delta ** 2)

    # Step 4: Update parameters
    w = w + delta

    return w, E_grad_sq, E_update_sq