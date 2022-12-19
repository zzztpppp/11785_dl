import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from phonetics import Alphabet
import os

training_x_dir = os.path.join("hw4p2_data", "train-clean-100", "mfcc")
training_y_dir = os.path.join("hw4p2_data", "train-clean-100", "transcript")

dev_x_dir = os.path.join("hw4p2_data", "dev-clean", "mfcc")
dev_y_dir = os.path.join("hw4p2_data", "dev-clean", "transcript")

test_x_dir = os.path.join("hw4p2_data", "test-clean", "mfcc")


# Copied from hw3p2
class LabeledDataset(Dataset):
    # load the dataset
    def __init__(self, x, y):
        # X and y are the directories containing training data and labelkk
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
        # Transform string to integer encoding
        y = np.array([Alphabet.where(p.lower()) for p in y])
        seq_length = self.seq_lengths[index]
        target_length = self.target_lengths[index]

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


class UnlabeledDataset(Dataset):
    def __init__(self, x):
        x_files = [os.path.join(x, p) for p in os.listdir(x)]
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

class PyramidLSTM(nn.Module):
    """
    The pyramid-lstm block reduce input's time-step by half and feed into a
    bidirectional lstm layer
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, batch_x, seq_lengths):
        # Reduce the sequence length by 2
        max_seq_length = batch_x.shape[1]



class Listener(nn.Module):
    """
    Listener consists of 1D cnn and some specified layers of lstms
    """

    def __init__(self, lstm_layers, input_size, initial_hidden_size, reduce=None):
        super().__init__()
        # The first layer doesn't reduce the number of times steps/
        # The rest each reduces the number of time steps by a factor


