import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((num_classes, num_classes), dtype=float)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if normalize == 'true':
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum

    elif normalize == 'pred':
        col_sum = cm.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1
        cm = cm / col_sum

    elif normalize == 'all':
        total = cm.sum()
        if total != 0:
            cm = cm / total

    return cm