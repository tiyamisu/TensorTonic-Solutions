import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    if average == "micro":
        tp = np.sum(y_true == y_pred)
        total = len(y_true)
        precision = recall = f1 = tp / total if total else 0.0

    elif average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

    else:  # macro or weighted
        labels = np.unique(np.concatenate([y_true, y_pred]))
        precisions, recalls, f1s, weights = [], [], [], []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))

            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            weights.append(np.sum(y_true == label))

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        weights = np.array(weights)

        if average == "weighted":
            precision = np.average(precisions, weights=weights)
            recall = np.average(recalls, weights=weights)
            f1 = np.average(f1s, weights=weights)
        else:  # macro
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }