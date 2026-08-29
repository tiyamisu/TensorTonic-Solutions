def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    remaining_budget = config["budget"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    max_staleness = config["max_staleness"]
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]

    last_retrain_day = None
    days_since_retrain = 0

    retrain_days = []

    for stat in daily_stats:
        day = stat["day"]

        # Must increment BEFORE checking the current day.
        days_since_retrain += 1

        needs_retraining = (
            stat["drift_score"] > drift_threshold
            or stat["performance"] < performance_threshold
            or days_since_retrain >= max_staleness
        )

        cooldown_ok = (
            last_retrain_day is None
            or day - last_retrain_day >= cooldown
        )

        budget_ok = remaining_budget >= retrain_cost

        if needs_retraining and cooldown_ok and budget_ok:
            retrain_days.append(day)

            remaining_budget -= retrain_cost
            last_retrain_day = day
            days_since_retrain = 0

    return retrain_days