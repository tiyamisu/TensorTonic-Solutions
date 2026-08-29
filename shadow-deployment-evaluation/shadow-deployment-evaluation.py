import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    n = len(production_log)

    if n == 0:
        return {
            "promote": False,
            "metrics": {
                "shadow_accuracy": 0.0,
                "production_accuracy": 0.0,
                "accuracy_gain": 0.0,
                "shadow_latency_p95": 0,
                "agreement_rate": 0.0
            }
        }

    production_accuracy = (
        sum(
            row["prediction"] == row["actual"]
            for row in production_log
        ) / n
    )

    shadow_accuracy = (
        sum(
            row["prediction"] == row["actual"]
            for row in shadow_log
        ) / n
    )

    accuracy_gain = shadow_accuracy - production_accuracy

    # Nearest-rank P95:
    # index = ceil(0.95 * n) - 1
    sorted_latencies = sorted(
        row["latency_ms"]
        for row in shadow_log
    )

    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = sorted_latencies[p95_index]

    # Same ordered request positions.
    agreement_rate = (
        sum(
            production_log[i]["prediction"]
            == shadow_log[i]["prediction"]
            for i in range(n)
        ) / n
    )

    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate
        }
    }