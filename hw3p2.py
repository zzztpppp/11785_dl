import os
import sys

import numpy as np
import torch
import argparse
import pandas as pd
import Levenshtein
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from utils import greedy_search, beam_search_batch

from torch.nn.functional import log_softmax, softmax
from torch.utils.data import Dataset, DataLoader
from phonemes import PHONEME_MAP, PHONEMES
from ctcdecode import CTCBeamDecoder

train_data_dir = os.path.join("hw3p2_data", "train", "mfcc")
train_label_dir = os.path.join("hw3p2_data", "train", "transcript")

dev_data_dir = os.path.join("hw3p2_data", "dev", "mfcc")
dev_label_dir = os.path.join("hw3p2_data", "dev", "transcript")

test_data_dir = os.path.join("hw3p2_data", "test", "mfcc")
test_data_order_dir = os.path.join('hw3p2_data', "test", "test_order.csv")

# Check if cuda is available and set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

num_workers = 8 if cuda else 0

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers), " system version = ", sys.version)

phoneme2int = dict(zip(PHONEMES, range(len(PHONEMES))))


# Define dataset class
class LabeledDataset(Dataset):
    # load the dataset
    def __init__(self, x, y):
        # X and y are the directories containing training data and label
        x_file_list = [os.path.join(x, p) for p in os.listdir(x)]
        y_file_list = [os.path.join(y, p) for p in os.listdir(y)]
        self.X = [np.load(p, allow_pickle=True) for p in x_file_list]
        self.Y = [np.load(p, allow_pickle=True) for p in y_file_list]
        self.seq_lengths = [x.shape[0] for x in self.X]
        self.target_lengths = [y.shape[0] for y in self.Y]

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

    # get row item at some index
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        # Remove <sos> an <eos>
        y = y[1: -1]
        # Transform string to integer encoding
        y = np.array([phoneme2int[p] for p in y])
        seq_length = self.seq_lengths[index]
        target_length = self.target_lengths[index] - 2  # Minus the length of <eos> an <sos>
        return (x, y), (seq_length, target_length)

    @staticmethod
    def collate_fn(batch):
        batch_x = [torch.tensor(x) for (x, _), _ in batch]
        batch_y = [torch.tensor(y) for (_, y), _ in batch]
        batch_seq_lengths = [l for _, (l, _) in batch]
        batch_target_lengths = [t for _, (_, t) in batch]

        # Pad variable length sequences
        padded_x = pad_sequence(batch_x, batch_first=True)
        padded_y = pad_sequence(batch_y, batch_first=True)
        return (padded_x, padded_y), (batch_seq_lengths, batch_target_lengths)


# Define dataset class
class TestDataSet(Dataset):
    # load the dataset
    # TODO: replace x and y with dataset path and load data from here -> more efficient
    def __init__(self, x, test_order_path):
        with open(test_order_path, "r") as f:
            file_order = f.read().splitlines()
        # Remove header
        file_order = file_order[1:]
        x_files = [os.path.join(x, p) for p in file_order]
        self.X = [np.load(p, allow_pickle=True) for p in x_files]
        self.seq_lengths = [x.shape[0] for x in self.X]

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        seq_length = self.seq_lengths[index]
        return x, seq_length

    @staticmethod
    def collate_fn(batch):
        batch_x = [torch.tensor(x) for (x, _) in batch]
        batch_seq_lengths = [l for (_, l) in batch]
        return pad_sequence(batch_x, batch_first=True), batch_seq_lengths


training_batch_size = 16
val_batch_size = 128
test_batch_size = 128  # Save memory for beam search


def get_labeled_data_loader(x_dir, y_dir, batch_size):
    train_set = LabeledDataset(x_dir, y_dir)
    train_args = {"shuffle": True, "collate_fn": LabeledDataset.collate_fn}
    train_loader = DataLoader(train_set, **train_args, pin_memory=True, num_workers=4, batch_size=batch_size)
    return train_loader


def get_unlabeled_data_loader(x_dir, data_order_dir, batch_size):
    test = TestDataSet(x_dir, data_order_dir)
    test_args = {"batch_size": batch_size, "collate_fn": TestDataSet.collate_fn}
    test_loader = DataLoader(test, shuffle=False, **test_args)
    return test_loader


def pick_and_translate_beams(beam_results: torch.tensor, beam_scores: torch.tensor, lengths: torch.tensor):
    """
    Pick the best result from beam search and translate to phonemes
    :param beam_results: batch_size * n_beams * T
    :param beam_scores: batch_size * n_beams
    :param lengths: batch_size * n_beams
    :return:
    """
    batch_size, _, _ = beam_results.shape
    result = []
    best_beam_index = beam_scores.argmax(dim=1)
    best_beams = torch.vstack([beam_results[i, best_beam_index[i], :].squeeze() for i in range(batch_size)])
    best_lengths = torch.vstack([lengths[i, best_beam_index[i]].squeeze() for i in range(batch_size)])
    batch_size, _ = best_beams.shape
    # Translate beams
    for b in range(batch_size):
        current_beam_length = best_lengths[b]
        current_beam = best_beams[b]
        phonemes = [PHONEME_MAP[c] for c in current_beam[:current_beam_length]]
        result.append(''.join(phonemes))
    return result


# BaseLine model
# cnn --> lstm --> mlp
class Model1(torch.nn.Module):
    def __init__(self, cnn_channels, n_lstm_layers, bidirectional, mlp_sizes, dropout, **kwargs):
        super(Model1, self).__init__()
        self.cnn_1 = torch.nn.Conv1d(in_channels=13, out_channels=cnn_channels, kernel_size=3, padding='same')
        self.cnn_2 = torch.nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3,
                                     stride=2)
        self.lstm = torch.nn.LSTM(input_size=cnn_channels * 2, hidden_size=cnn_channels * 2, num_layers=n_lstm_layers,
                                  bidirectional=bidirectional, batch_first=True, dropout=dropout)

        # Use the last hidden output as the final lstm layer output
        factor = 2 if bidirectional else 1
        mlp = [torch.nn.Linear(cnn_channels * factor * 2, mlp_sizes[0]), torch.nn.ReLU(), torch.nn.Dropout(p=dropout)]
        for i in range(len(mlp_sizes) - 1):
            mlp.append(torch.nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
            mlp.append(torch.nn.Dropout(p=dropout))
            mlp.append(torch.nn.ReLU())

        # Output layer
        mlp.extend([torch.nn.Linear(mlp_sizes[-1], len(PHONEMES))])
        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x, seq_lengths):
        # x is of shape (N, T, 13)
        # But conv1d requires (N, 13, T)
        embed = self.cnn_1.forward(x.transpose(1, 2))
        embed = torch.nn.functional.relu(embed)
        embed = self.cnn_2.forward(embed)
        embed =  torch.nn.functional.relu(embed)
        # pack_padded_sequence requires (N, T, 13)
        # Since we are doning down-sampling ,the seq lengths change correspondingly.
        down_sampled_seq_length = [(x - 3) // 2 for x in seq_lengths]
        packed_embed = pack_padded_sequence(embed.transpose(1, 2), down_sampled_seq_length, batch_first=True, enforce_sorted=False)

        seq_out, _ = self.lstm.forward(packed_embed)  # (N, L, D*H_out)
        # Unpack
        unpacked_seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        out = self.mlp.forward(unpacked_seq_out)

        # Shape of (N, L, 41)
        return out, down_sampled_seq_length


def predict_string(model, batch_x, batch_seq_length, decoder):
    result, _ = predict(model, batch_x, batch_seq_length)
    beam_results, beam_scores, _, seq_len = decoder.decode(softmax(result, dim=-1))
    best_seqs = pick_and_translate_beams(beam_results, beam_scores, seq_len)
    return best_seqs


def output_result(model, beam_width=10):
    """
    Predict the result of our test set.
    Determine the output sequence by beam-searching.
    param test_data_loader:
    :param model:
    :return:
    """
    decoder = CTCBeamDecoder(PHONEME_MAP, beam_width=beam_width, num_processes=8)
    test_data_loader = get_unlabeled_data_loader(test_data_dir, test_data_order_dir, test_batch_size)
    output_seqs = []
    for batch_x, batch_seq_length in test_data_loader:
        batch_string = predict_string(model, batch_x, batch_seq_length, decoder)
        output_seqs.extend(batch_string)

    ids = np.arange(len(output_seqs))
    data = np.vstack([ids, np.array(output_seqs)])
    df = pd.DataFrame(data=data.T, columns=['id', 'predictions'])
    df.to_csv("hw3p2_submission.csv", index=False)


def train_epoch(training_loader, model, criterion, optimizer):
    (batch_x, batch_y), (batch_seq_lengths, batch_target_sizes) = next(iter(training_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    model.to(device)
    model.train()
    optimizer.zero_grad()
    outputs, new_seq_length = model(batch_x, batch_seq_lengths)
    # CTC loss requires (L B, C) so we transpose
    loss = criterion(log_softmax(outputs.transpose(0, 1), dim=-1), batch_y, new_seq_length, batch_target_sizes)
    print(f"Training loss {loss}")
    loss.backward()
    optimizer.step()


def predict(model, x, x_lengths):
    model.eval()
    y, new_length = model(x, x_lengths)
    return y, new_length


def validate(model, validation_loader, criterion, decoder=None):
    total_loss = 0.0
    total_distance = 0.0
    total_size = 0
    decoder = CTCBeamDecoder(PHONEME_MAP, beam_width=5, num_processes=8)
    with torch.no_grad():
        for batch_num, ((batch_x, batch_y), (batch_seq_lengths, batch_target_lengths)) in enumerate(validation_loader):
            batch_size, _ = batch_y.shape
            total_size += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # Output from model is (N, L, D) but ctcloss accepts (L, N, D)
            logit, new_seq_length = predict(model, batch_x, batch_seq_lengths)
            y_hat = log_softmax(logit, dim=-1)
            loss = criterion(y_hat.transpose(0, 1), batch_y, new_seq_length, batch_target_lengths)
            batch_string = predict_string(model, batch_x, batch_seq_lengths, decoder)
            for y_hat_string, y in zip(batch_string, batch_y):
                y_string = ''.join([PHONEME_MAP[c] for c in y])
                total_distance += Levenshtein.distance(y_hat_string, y_string)
            total_loss += (loss.item() * batch_size)

    # Return average loss
    return total_loss / total_size, total_distance / total_size


def get_model(model_name, model_param):
    if model_name == "Model1":
        pass


def train(model, optimizer_params):
    n_epochs = 500
    print(model)

    lr = optimizer_params.get("lr")
    warmup_epoch = optimizer_params.get("warmup")
    if warmup_epoch is None:
        warmup_epoch = 150
    weight_decay = optimizer_params.get("weight_decay")
    if lr is None:
        lr = 0.002
    beam_width = optimizer_params.get("beam_width")
    if beam_width is None:
        beam_width = 10
    train_loader = get_labeled_data_loader(train_data_dir, train_label_dir, training_batch_size)
    dev_loader = get_labeled_data_loader(dev_data_dir, dev_label_dir, val_batch_size)
    validation_decoder = CTCBeamDecoder(PHONEME_MAP, beam_width=beam_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=0, verbose=True)
    ctc_loss = torch.nn.CTCLoss()
    for i in range(n_epochs):
        train_epoch(train_loader, model, ctc_loss, optimizer)
        validation_loss, validation_distance = validate(model, dev_loader, ctc_loss)
        print(f"Epoch {i}, validation_loss: {validation_loss}, validation_distance: {validation_distance}")
        # Warmup for the first 150 epochs:
        scheduler.step(validation_distance)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'test'])
    parser.add_argument("model")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--model_path")

    args = parser.parse_args()
    training_params = vars(args)
    torch.cuda.empty_cache()
    model_params = {"cnn_channels": 128, "n_lstm_layers": 4, "bidirectional": True, "mlp_sizes": [2048], "dropout": 0.3}
    model_params_str = f"({', '.join(['='.join([str(k), str(v)]) for k, v in model_params.items()])})"
    model_name = args.model
    training_model = eval(model_name + model_params_str)

    if args.mode == 'train':
        training_model = train(training_model, training_params)
        torch.save(training_model.state_dict(), "saved_model_hw3p2")
    else:
        path = args.model_path
        if path is None:
            path = "saved_model_hw3p2"
        training_model.load_state_dict(torch.load(path))

    output_result(training_model)
