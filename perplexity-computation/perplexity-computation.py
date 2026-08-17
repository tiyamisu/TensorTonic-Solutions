import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence
    given predicted probability distributions.
    """

    total_log_prob = 0.0
    N = len(actual_tokens)

    for probs, token in zip(prob_distributions, actual_tokens):

        p = probs[token]

        if p <= 0:
            return float("inf")

        total_log_prob += math.log(p)

    cross_entropy = -total_log_prob / N

    return math.exp(cross_entropy)