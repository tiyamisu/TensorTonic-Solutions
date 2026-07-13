import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        matrix = np.asarray(matrix, dtype=float)

        # Must be exactly 2D
        if matrix.ndim != 2:
            return None

        if axis not in (None, 0, 1):
            return None

        if norm_type == 'l2':
            if axis is None:
                norm = np.linalg.norm(matrix)
            else:
                norm = np.linalg.norm(matrix, axis=axis, keepdims=True)

        elif norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)

        elif norm_type == 'max':
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

        else:
            return None

        # Avoid division by zero: zero rows/columns remain zero
        norm = np.where(norm == 0, 1, norm)

        return matrix / norm

    except:
        return None