def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    return [
        [sum(X[i][k] * W[k][j] for k in range(len(W))) + b[j]
         for j in range(len(W[0]))]
        for i in range(len(X))
    ]