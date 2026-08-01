import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q_new = np.array(Q, dtype=float).copy()

    td_target = r + gamma * np.max(Q_new[s_next])
    td_error = td_target - Q_new[s][a]

    Q_new[s][a] = Q_new[s][a] + alpha * td_error

    return Q_new