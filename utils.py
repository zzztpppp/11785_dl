# Sequence search
import numpy as np
from itertools import groupby


def greedy_search(symbol_set, y_probs):
    """
    Greedily search the genenrated probs and output the sequence with the maximum
    probability.

    :param symbol_set: List of all symbols wihtout blank
    :param y_probs: (symbols, seq_length, batch_size)
    :return:
    """

    best_symbols = y_probs.argmax(axis=0)
    seq_length, batch_size = best_symbols.shape
    symbol_indices = [[key for key, _ in groupby(best_symbols[:, i]) if key != 0]
                      for i in range(batch_size)]
    forward_path = [
        ''.join([symbol_set[key] for key in symbol_indices[i]])
        for i in range(batch_size)
    ]
    forward_prob = np.prod(np.array([
        [y_probs[key, i, j] for i, key in enumerate(best_symbols[:, j])]
        for j in range(batch_size)
    ]), axis=0
    )

    return forward_path, forward_prob