import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    loss = 0.0
    n = len(predictions)

    for p, y in zip(predictions, targets):
        if y == 1:
            pt = p
        else:
            pt = 1 - p

        loss += -alpha * ((1 - pt) ** gamma) * math.log(pt)

    return loss / n