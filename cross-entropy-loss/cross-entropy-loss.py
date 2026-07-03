import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    loss = 0.0

    for i in range(len(y_true)):
        loss += -np.log(y_pred[i][y_true[i]])

    return loss / len(y_true)