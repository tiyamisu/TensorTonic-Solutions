import numpy as np

def replay_buffer_sample(buffer, batch_size, seed):
    np.random.seed(seed)
    idx = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[i] for i in idx]
    