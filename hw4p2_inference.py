import torch
import numpy as np
from phonetics import SOS_TOKEN, EOS_TOKEN
from hw4p2_train import test_x_dir, device, UnlabeledDataset
from las import LAS
from hw4p2_train import translate, get_unlabeled_dataloader
from phonetics import VOCAB


def beam_search(model: LAS, data_loader, beam_width):
    pass


def sum_log_probs(batch_log_probs, output_lengths, normalize=False):
    _, max_length = batch_log_probs.shape
    output_lengths = torch.tensor(output_lengths)
    mask = torch.arange(0, max_length)[None, :] >= output_lengths[:, None]
    batch_log_probs = batch_log_probs.masked_fill(mask, 0)
    batch_total_log_probs = batch_log_probs.sum(dim=1)
    if normalize:
        batch_total_log_probs = batch_total_log_probs / output_lengths
    return batch_total_log_probs


def _pick(batch_output_seqs, batch_output_log_probs):
    batch_n_strings = np.array(batch_output_seqs).T  # (Batch, N_runs)
    batch_size, _ = batch_n_strings.shape
    print(batch_n_strings.shape)
    batch_n_probs = torch.stack(batch_output_log_probs, dim=1).cpu().numpy()
    return batch_n_strings[np.arange(0, batch_size), batch_n_probs.argmax(axis=1)]


def random_search(model: LAS, data_loader, n_runs):
    model.eval()
    all_strings = []
    with torch.inference_mode():
        for batch_x, seq_length in data_loader:
            seq_embeddings, seq_embedding_lengths = model.listener.forward(batch_x, seq_length)
            batch_output_seqs, batch_output_log_probs = [], []
            for _ in range(n_runs):
                _, (output_seqs, output_log_probs) = model.speller.forward(seq_embeddings, seq_embedding_lengths)

                y_hat_strings, y_hat_length = translate(output_seqs)
                y_hat_string_log_probs = sum_log_probs(output_log_probs, y_hat_length, True)
                batch_output_seqs.append(y_hat_strings)
                batch_output_log_probs.append(y_hat_string_log_probs)
            # Pick the one with the largest log prob.
            all_strings.append(_pick(batch_output_seqs, batch_output_log_probs))
    return np.concatenate(all_strings)


def inference(params):
    model_checkpoint = params["model_path"]
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
        num_workers=4,
        shuffle=False
    )

    searching_method = params["searching_method"]
    model.load_state_dict(torch.load(model_checkpoint))
    if searching_method == "random":
        predicted_strings = random_search(model, dataloader, params["n_runs"])
    else:
        raise NotImplementedError()

    print(predicted_strings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("mfcc_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--searching_method", type=str, default="random")
    parser.add_argument("--n_runs", type=int, default=100, help="Number of random runs if searching method is random")
    parser.add_argument("--num_dataloader_workers", type=int, default=2)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--context_size", type=int, default=512)
    parser.add_argument("--plstm_layers", type=int, default=3)
    args = parser.parse_args()
    inference(vars(args))
