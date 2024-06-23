import typing
from typing import List

import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import math

if typing.TYPE_CHECKING:
    from models import ASRModel

VOCAB = [
    '<pad>', '<sos>', '<eos>',
    'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X',
    'Y', 'Z', "'", ' ',
]

VOCAB_MAP = {VOCAB[i]: i for i in range(0, len(VOCAB))}

PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]


def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices:  # This loops through all the indices
        if int(i) == SOS_TOKEN:  # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN:  # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return tokens


def calc_edit_distance(predictions, y, y_len, vocab, print_example=False):
    dist = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):
        y_sliced = indices_to_chars(y[batch_idx, 0:y_len[batch_idx]], vocab)
        pred_sliced = indices_to_chars(predictions[batch_idx], vocab)

        # Strings - When you are using characters from the AudioDataset
        y_string = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)

        # dist        += Levenshtein.distance(pred_string, y_string)
        # Comment the above abd uncomment below for toy dataset
        dist += Levenshtein.distance(y_sliced, pred_sliced)

    if print_example:
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print("\nGround Truth : ", y_string)
        print("Prediction   : ", pred_string)

    dist /= batch_size
    return dist


def plot_attention(attention):
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def beam_search(beam_with: int, model: 'ASRModel', x: torch.Tensor, x_len: int):
    model.eval()
    max_len = 600
    listener = model.listener
    speller = model.speller
    attender = model.attend
    sequence_embedding, _ = listener(x, x_len)
    attender.set_key_value(sequence_embedding)
    embedding_size = speller.embedding_size
    # log-probs, current sequence, attention_context, hidden_states, is-ended
    beams = [[0.0,  [SOS_TOKEN], torch.zeros(size=(1, embedding_size)).to(DEVICE), [], False]]
    finished_paths = []
    # Beam search
    for t in range(max_len):
        new_beams = []
        for path in beams:
            if path[1][-1] == EOS_TOKEN:
                path[-1] = True
                finished_paths.append(path)
            else:
                with torch.inference_mode():
                    char_embedding = speller.embedding(torch.ones(size=(1, ), dtype=torch.long).to(DEVICE) * path[1][-1])
                    lstm_input = torch.concat([char_embedding, path[2]], dim=1)
                    hidden_state_t = speller.lstm_step(lstm_input, path[3])
                    path[3].append(hidden_state_t)
                    context, _ = attender.compute_context(hidden_state_t[-1][0])
                    cdn_input = torch.concat([hidden_state_t[-1][0], context], dim=1)
                    raw_pred = speller.cdn(cdn_input)
                    log_probs = torch.nn.functional.log_softmax(raw_pred, dim=-1).squeeze(0)
                candidate_tokens = torch.argsort(log_probs, descending=True)[:beam_with]
                for token in candidate_tokens:
                    path[1].append(token.item())
                    new_path = [
                        path[0] + log_probs[token].item(),
                        path[1],
                        context,
                        [hidden_state_t],
                        False,
                    ]
                    new_beams.append(new_path)
            beams = sorted(new_beams, key=lambda p: p[0], reverse=True)[:beam_with]
        if len(beams) == 0:
            break

    result = max(finished_paths, key=lambda p: p[0])[1]

    return result
