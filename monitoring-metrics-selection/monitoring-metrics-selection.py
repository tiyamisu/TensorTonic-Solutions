import math

def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    n = len(y_true)

    if system_type == "classification":
        tp = sum(
            1 for yt, yp in zip(y_true, y_pred)
            if yt == 1 and yp == 1
        )

        tn = sum(
            1 for yt, yp in zip(y_true, y_pred)
            if yt == 0 and yp == 0
        )

        fp = sum(
            1 for yt, yp in zip(y_true, y_pred)
            if yt == 0 and yp == 1
        )

        fn = sum(
            1 for yt, yp in zip(y_true, y_pred)
            if yt == 1 and yp == 0
        )

        accuracy = (tp + tn) / n if n else 0.0

        precision = (
            tp / (tp + fp)
            if (tp + fp) != 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) != 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    elif system_type == "regression":
        if n == 0:
            return {
                "mae": 0.0,
                "rmse": 0.0
            }

        errors = [
            yp - yt
            for yt, yp in zip(y_true, y_pred)
        ]

        mae = sum(abs(error) for error in errors) / n

        rmse = math.sqrt(
            sum(error ** 2 for error in errors) / n
        )

        return {
            "mae": mae,
            "rmse": rmse
        }

    elif system_type == "ranking":
        # Python's sorted() is stable, so tied scores
        # retain their original input order.
        ranked = sorted(
            zip(y_true, y_pred),
            key=lambda item: item[1],
            reverse=True
        )

        top_three = ranked[:3]

        relevant_at_3 = sum(
            1 for target, _ in top_three
            if target
        )

        total_relevant = sum(
            1 for target in y_true
            if target
        )

        precision_at_3 = relevant_at_3 / 3

        recall_at_3 = (
            relevant_at_3 / total_relevant
            if total_relevant != 0
            else 0.0
        )

        return {
            "precision_at_3": precision_at_3,
            "recall_at_3": recall_at_3
        }

    return {}