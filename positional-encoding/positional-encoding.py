import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.power(base, (2 * (np.arange(d_model) // 2)) / d_model)

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(position / div_term[0::2])
    pe[:, 1::2] = np.cos(position / div_term[1::2])

    return pe