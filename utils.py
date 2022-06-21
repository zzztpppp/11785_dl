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


def initialize_paths(symbol_set, y_probs):
    paths_t_blank = {}
    paths_t_symbol = {}
    for i, probs in enumerate(y_probs):
        if i == 0:
            paths_t_blank[' '] = y_probs[0]
            continue
        paths_t_symbol[symbol_set[i - 1]] = y_probs[i]

    return paths_t_blank, paths_t_symbol


def prune(paths_t_blank, paths_t_symbol, beam_width):
    # Collect scores and find the cutoff of beam width
    scores = []
    scores.extend(paths_t_blank.values())
    scores.extend(paths_t_symbol.values())
    sorted_scores = sorted(scores, reverse=True)  # In descending order
    cutoff = sorted_scores[beam_width - 1] if beam_width < len(scores) else sorted_scores[-1]

    # Remove paths with probs less than cut off
    pruned_t_blank = {k: v for k, v in paths_t_blank.items() if v >= cutoff}
    pruned_t_symbol = {k: v for k, v in paths_t_symbol.items() if v >= cutoff}

    return pruned_t_blank, pruned_t_symbol


def extend_with_blank(paths_t_blank, paths_t_symbol, y_prob):
    # Uion two sets of paths
    updated_t_blank = {}
    for k, v in paths_t_blank.items():
        # Appending to a path terminating with blank a blank doesn't
        # change the path representation
        updated_t_blank[k] = y_prob[0] * v

    for k, v in paths_t_symbol.items():
        # If the path already exists after the appending
        new_k = k + ' '
        if updated_t_blank.get(new_k) is not None:
            updated_t_blank[new_k] += (y_prob[0] * v)
        else:
            updated_t_blank[new_k] = y_prob[0] * v

    return updated_t_blank


def extend_with_symbol(paths_t_blank, paths_t_symbol, symbol_set, y_prob):
    updated_t_symbol = {}
    for k, v in paths_t_symbol.items():
        for i, s in enumerate(symbol_set):
            new_k = k + s if k[-1] != s else k  # Merge identical characters
            if updated_t_symbol.get(new_k) is not None:
                updated_t_symbol[new_k] += (v * y_prob[i + 1])
            else:
                updated_t_symbol[new_k] = (v * y_prob[i + 1])

    for k, v in paths_t_blank.items():
        for i, s in enumerate(symbol_set):
            # Remove the trailing blank
            new_k = k[:-1] + s

            if updated_t_symbol.get(new_k) is None:
                updated_t_symbol[new_k] = v * y_prob[i + 1]
            else:
                updated_t_symbol[new_k] += v * y_prob[i + 1]

    return updated_t_symbol


def merge_paths(paths_t_blank, paths_t_symbol):
    """
    Merge paths only differ by the last blank
    """
    for k, v in paths_t_blank.items():
        eq_k = k[:-1]  # Remove the final blank
        if paths_t_symbol.get(eq_k) is not None:
            paths_t_symbol[eq_k] += v
        else:
            paths_t_symbol[eq_k] = v
    return paths_t_symbol


def beam_search(symbol_sets, y_probs, beam_width):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    _, seq_len, _ = y_probs.shape
    new_paths_t_blank, new_paths_t_symbol = initialize_paths(symbol_sets, y_probs[:, 0, 0])

    for t in range(1, seq_len):
        paths_t_blank, paths_t_symbol = prune(new_paths_t_blank, new_paths_t_symbol, beam_width)
        new_paths_t_blank = extend_with_blank(paths_t_blank, paths_t_symbol, y_probs[:, t, 0])
        new_paths_t_symbol = extend_with_symbol(paths_t_blank, paths_t_symbol, symbol_sets, y_probs[:, t, 0])

    merged_paths = merge_paths(new_paths_t_blank, new_paths_t_symbol)
    best_path_idx = np.argmax(list(merged_paths.values()))
    best_path = list(merged_paths.keys())[best_path_idx]
    return best_path, merged_paths


def beam_search_batch(symbol_sets, y_probs_batch, beam_width):
    """
    A simple wrap of beam_search

    :param symbol_sets:
    :param y_probs_batch:
    :param beam_width:
    :return:
    """
    best_path_list = []
    merged_path_list = []
    _, _, batch_size = y_probs_batch.shape
    for i in range(batch_size):
        best_path, merged_path = beam_search(symbol_sets,
                                             y_probs_batch[:, :, [i]],  # Square bracket to keep dimension
                                             beam_width)
        best_path_list.append(best_path)
        merged_path_list.append(merged_path)
    return best_path_list, merged_path_list
