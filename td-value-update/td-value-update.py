import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    V_new = V.copy()

    # TD error
    delta = r + gamma * V[s_next] - V[s]

    # One-step TD(0) update
    V_new[s] = V[s] + alpha * delta

    return V_new