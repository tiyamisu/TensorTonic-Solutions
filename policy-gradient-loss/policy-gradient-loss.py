def policy_gradient_loss(log_probs, rewards, gamma):
    n = len(rewards)

    returns = [0] * n
    G = 0
    for i in range(n - 1, -1, -1):
        G = rewards[i] + gamma * G
        returns[i] = G

    baseline = sum(returns) / n

    loss = 0
    for lp, G in zip(log_probs, returns):
        advantage = G - baseline
        loss += -lp * advantage

    return loss / n