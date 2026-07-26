import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Start with +inf so the first point is (0,0)
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(y_score))[::-1]))

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    fpr = []
    tpr = []

    for th in thresholds:
        pred = y_score >= th

        TP = np.sum((pred == 1) & (y_true == 1))
        FP = np.sum((pred == 1) & (y_true == 0))

        tpr.append(TP / P if P else 0.0)
        fpr.append(FP / N if N else 0.0)

    return np.array(fpr), np.array(tpr), thresholds