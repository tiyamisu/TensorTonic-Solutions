def priority_replay_sample(priorities, alpha, beta):
    powered = [p ** alpha for p in priorities]

    total = sum(powered)
    probs = [p / total for p in powered]

    N = len(priorities)
    weights = [(N * p) ** (-beta) for p in probs]

    max_w = max(weights)
    weights = [w / max_w for w in weights]

    return [probs, weights]