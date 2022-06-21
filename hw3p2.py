import os
import sys

import numpy as np
import torch
import argparse
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from utils import greedy_search, beam_search_batch

from torch.nn.functional import log_softmax, softmax
from torch.utils.data import Dataset, DataLoader
from phonemes import PHONEME_MAP, PHONEMES


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

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers),  " system version = ", sys.version)

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
        batch_target_lengths =[t for _, (_, t) in batch]

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
        batch_x = [torch.tensor(x) for (x, _)in batch]
        batch_seq_lengths =[l for (_, l) in batch]
        return pad_sequence(batch_x, batch_first=True), batch_seq_lengths


training_batch_size = 32
val_batch_size = 128
test_batch_size = 128  # Save memory for beam search


# training data
train_set = LabeledDataset(train_data_dir, train_label_dir)
train_args = {"batch_size": training_batch_size, "shuffle": True, "collate_fn": LabeledDataset.collate_fn}
train_loader = DataLoader(train_set, **train_args, pin_memory=True, num_workers=4)

# validation data
dev = LabeledDataset(dev_data_dir, dev_label_dir)
dev_args = {"batch_size": val_batch_size, "shuffle": True, "collate_fn": LabeledDataset.collate_fn}
dev_loader = DataLoader(dev, **dev_args, pin_memory=True, num_workers=4)

# test data
test = TestDataSet(test_data_dir, test_data_order_dir)

test_args = {"batch_size": test_batch_size, "collate_fn": TestDataSet.collate_fn}
test_loader = DataLoader(test, shuffle=False, **test_args)


# BaseLine model
# cnn --> lstm --> mlp
class Model1(torch.nn.Module):
    def __init__(self, cnn_channels, n_lstm_layers, bidirectional, mlp_sizes):
        super(Model1, self).__init__()
        self.cnn = torch.nn.Conv1d(in_channels=13, out_channels=cnn_channels, kernel_size=13, padding="same")
        self.lstm = torch.nn.LSTM(input_size=cnn_channels, hidden_size=cnn_channels, num_layers=n_lstm_layers,
                                  bidirectional=bidirectional, batch_first=True)

        # Use the last hidden output as the final lstm layer output
        factor = 2 if bidirectional else 1
        mlp = [torch.nn.Linear(cnn_channels * factor, mlp_sizes[0]), torch.nn.ReLU()]
        for i in range(len(mlp_sizes) - 1):
            mlp.append(torch.nn.Linear(mlp_sizes[i], mlp_sizes[i+1]))
            mlp.append(torch.nn.ReLU())

        # Output layer
        mlp.extend([torch.nn.Linear(mlp_sizes[-1], len(PHONEMES))])
        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x, seq_lengths):
        # x is of shape (N, T, 13)
        # But conv1d requires (N, 13, T)
        embed = self.cnn.forward(x.transpose(1, 2))
        embed = torch.nn.functional.relu(embed)
        # pack_padded_sequence requires (N, T, 13)
        packed_embed = pack_padded_sequence(embed.transpose(1, 2), seq_lengths, batch_first=True, enforce_sorted=False)

        seq_out, _ = self.lstm.forward(packed_embed)  # (N, L, D*H_out)
        # Unpack
        unpacked_seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        out = self.mlp.forward(unpacked_seq_out)

        # Shape of (N, L, 41)
        return out


def output_result(model, test_data_loader, search="beam", beam_width=10):
    """
    Predict the result of our test set.
    Determine the output sequence by beam-searching.
    :param test_data_loader:
    :param model:
    :return:
    """
    output_seqs = []
    model.to("cpu")
    for batch_x, batch_seq_length in test_data_loader:
        result = predict(model, batch_x, batch_seq_length)
        if search == 'beam':
            best_seqs, _ =  beam_search_batch(PHONEME_MAP[1:],
                                              softmax(result, dim=-1).detach().numpy().transpose(2, 1, 0), beam_width)
        else:
            best_seqs, _ = greedy_search(PHONEME_MAP, softmax(result, dim=-1).detach().numpy().transpose(2, 1, 0))

        output_seqs.extend(best_seqs)
    ids = np.arange(len(output_seqs))
    data = np.vstack([ids, np.array(output_seqs)])
    df = pd.DataFrame(data=data.T, columns=['id', 'predictions'])
    df.to_csv("hw3p2_submission.csv", index=False)


def train_epoch(training_loader, model, criterion, optimizer):
    (batch_x, batch_y), (batch_seq_lengths, batch_target_sizes) = next(iter(training_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    model.train()
    optimizer.zero_grad()
    outputs = model(batch_x, batch_seq_lengths)
    # CTC loss requires (L B, C) so we transpose
    loss = criterion(log_softmax(outputs.transpose(0, 1), dim=-1), batch_y, batch_seq_lengths, batch_target_sizes)
    print(f"Training loss {loss}")
    loss.backward()
    optimizer.step()


def predict(model, x, x_lengths):
    model.eval()
    y = model(x, x_lengths)
    return y


def validate(model, validation_loader, criterion):
    total_loss = 0.0
    total_size = 0
    with torch.no_grad():
        for batch_num, ((batch_x, batch_y), (batch_seq_lengths, batch_target_lengths)) in enumerate(validation_loader):
            batch_size, _ = batch_y.shape
            total_size += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # Output from model is (N, L, D) but ctcloss accepts (L, N, D)
            y_hat = log_softmax(predict(model, batch_x, batch_seq_lengths), dim=-1)
            loss = criterion(y_hat.transpose(0, 1), batch_y, batch_seq_lengths, batch_target_lengths)
            total_loss += (loss.item() * batch_size)

    # Return average loss
    return total_loss / total_size


def train():
    n_epochs = 100
    model = Model1(cnn_channels=256, n_lstm_layers=1, bidirectional=True, mlp_sizes=[2048])
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=0, verbose=True)
    ctc_loss = torch.nn.CTCLoss()
    model.to(device)
    for i in range(n_epochs):
        train_epoch(train_loader, model, ctc_loss, optimizer)
        validation_loss = validate(model, dev_loader, ctc_loss)
        print(f"Epoch {i}, validation_loss: {validation_loss}")
        scheduler.step(validation_loss)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'test'])
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    torch.cuda.empty_cache()

    if args.mode == 'train':
        trained_model = train()
        torch.save(trained_model.state_dict(), "saved_model_hw3p2")
    else:
        path = args.model_path
        if path is None:
            path = "saved_model_hw3p2"
        trained_model = Model1(cnn_channels=256, n_lstm_layers=4, bidirectional=True, mlp_sizes=[512, 512])
        trained_model.load_state_dict(torch.load(path))

    output_result(trained_model, test_loader, "greedy")
