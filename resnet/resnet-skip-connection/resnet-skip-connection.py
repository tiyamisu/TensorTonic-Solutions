import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    gradient = np.asarray(x)

    for J in gradients_F:
        J = np.asarray(J)
        I = np.eye(J.shape[0])
        gradient = gradient @ (J + I)

    return gradient


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    gradient = np.asarray(x)

    for J in gradients_F:
        J = np.asarray(J)
        gradient = gradient @ J

    return gradient