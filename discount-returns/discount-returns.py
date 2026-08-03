def discount_returns(rewards, gamma):
    n = len(rewards)
    returns = [0] * n

    G = 0
    for i in range(n - 1, -1, -1):
        G = rewards[i] + gamma * G
        returns[i] = G

    return returns