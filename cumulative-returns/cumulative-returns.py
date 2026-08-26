def cumulative_returns(returns: list) -> list:
    wealth = 1.0
    result = []

    for r in returns:
        wealth *= (1 + r)
        result.append(wealth - 1)

    return result