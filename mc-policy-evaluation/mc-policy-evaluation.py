import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """

    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        # Compute return G for each timestep
        G = 0
        returns_per_timestep = [0] * len(episode)

        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + gamma * G
            returns_per_timestep[t] = G

        # First-visit check
        visited = set()
        for t, (state, reward) in enumerate(episode):
            if state not in visited:
                visited.add(state)
                returns[state].append(returns_per_timestep[t])

    V = np.zeros(n_states)

    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])

    return V