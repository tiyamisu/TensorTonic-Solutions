import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.array(matrix, dtype=float)
        # Check if square matrix
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return None
        eigenvalues = np.linalg.eigvals(matrix)
        return np.sort(eigenvalues)
        
    except:
        return None