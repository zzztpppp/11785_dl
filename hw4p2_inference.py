import torch
import numpy as np
from phonetics import SOS_TOKEN, EOS_TOKEN
from hw4p2_train import test_x_dir, device, UnlabeledDataset
from las import LAS
from hw4p2_train import translate, get_unlabeled_dataloader, test_x_dir
from phonetics import VOCAB


def beam_search(model: LAS, data_loader, beam_width):
    pass


def sum_log_probs(batch_log_probs, output_lengths, normalize=False):
    _, max_length = batch_log_probs.shape
    output_lengths = torch.tensor(output_lengths)
    mask = torch.arange(0, max_length)[None, :] >= torch.tensor(output_lengths)[:, None]
    batch_log_probs = batch_log_probs.masked_fill(mask, 0)
    batch_total_log_probs = batch_log_probs.sum(dim=1)
    if normalize:
        batch_total_log_probs = batch_log_probs / output_lengths
    return batch_total_log_probs


def _pick(batch_output_seqs, batch_output_log_probs):
    batch_n_strings = np.array(batch_output_seqs)
    batch_n_probs = torch.stack(batch_output_log_probs).cpu().numpy()
    return batch_n_strings[:, batch_n_probs.argmax(dim=1)]


def random_search(model: LAS, data_loader, n_runs):
    model.eval()
    all_strings = []
    with torch.inference_mode():
        for batch_x, seq_length in data_loader:
            seq_embeddings, seq_embedding_lengths = model.listener.forward(batch_x, seq_length)
            batch_output_seqs, batch_output_log_probs = [], []
            for n in n_runs:
                _, (output_seqs, output_log_probs) = model.speller.forward(seq_embeddings, seq_embedding_lengths)
                batch_output_seqs.append(output_seqs)
                batch_output_log_probs.append(output_log_probs)

                y_hat_strings, y_hat_length = translate(output_seqs)
                y_hat_string_log_probs = sum_log_probs(output_log_probs, y_hat_length, True)
                batch_output_seqs.append(y_hat_strings)
                batch_output_log_probs.append(y_hat_string_log_probs)
            # Pick the one with the largest log prob.
            all_strings.append(_pick(batch_output_seqs, batch_output_log_probs))
    return np.concatenate(all_strings)


def inference(params):
    model_checkpoint = params["model_checkpoint"]
    model = LAS(
        params["embedding_size"],
        params["context_size"],
        len(VOCAB),
        plstm_layers=params["plstm_layers"],
        teacher_force_rate=1,
        encoder_dropout=0.5,
        decoder_dropout=0.5,
        freq_mask=0,
        time_mask=0
    )
    dataloader = get_unlabeled_dataloader(
        params["data_root"],
        test_x_dir,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=False
    )

    searching_method = params["searching_method"]
    model.load_state_dict(torch.load(model_checkpoint))
    if searching_method == "random":
        predicted_strings = random_search(model, dataloader, params["n_runs"])
    else:
        raise NotImplementedError()

