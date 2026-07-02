import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    values = np.asarray(values, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    n_states = len(values)
    new_values = np.zeros(n_states)

    for s in range(n_states):
        action_values = []
        for a in range(transitions.shape[1]):
            q = rewards[s][a] + gamma * np.dot(transitions[s][a], values)
            action_values.append(q)
        new_values[s] = max(action_values)

    return new_values.tolist()