import numpy as np

def auc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    return np.trapezoid(tpr, fpr)