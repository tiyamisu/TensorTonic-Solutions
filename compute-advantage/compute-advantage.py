import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """

    n = len(rewards)
    G = np.zeros(n)

    ret = 0
    for t in reversed(range(n)):
        ret = rewards[t] + gamma * ret
        G[t] = ret

    A = np.zeros(n)
    for t in range(n):
        A[t] = G[t] - V[states[t]]

    return A