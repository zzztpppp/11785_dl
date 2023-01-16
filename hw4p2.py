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
        batch_x_resized, seq_lengths_resized = self.reduce_length(batch_x, seq_lengths)
        packed_data = pack_padded_sequence(batch_x_resized, seq_lengths_resized, batch_first=True, enforce_sorted=False)
        packed_out = self.layer.forward(packed_data)
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return padded_out, seq_lengths_resized

    @staticmethod
    def reduce_length(batch_x: torch.tensor, seq_lengths: list) -> (torch.tensor, list):
        batch_size, max_length, hidden_size = batch_x.shape
        batch_resized = torch.zeros(
            batch_size,
            max_length // 2,
            hidden_size * 2,
            dtype=batch_x.dtype,
            layout=batch_x.layout,
            device=batch_x.device,
            requires_grad=True
        )

        resized_lengths = []
        for i, length in enumerate(seq_lengths):
            # Drop last step if length is odd
            length = length - (length % 2)
            resized_length = length // 2
            resized = batch_x[i, :length, :].reshape((resized_length, hidden_size * 2))   # (L // 2, H * 2)
            batch_resized[i, :, :]  = resized
            resized_lengths.append(resized_length)

        return batch_resized, resized_lengths


class PyLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        """
        The final output size is (hidden_size * (2 ** layers ))
        :param input_size:
        :param hidden_size:
        :param layers:
        """
        super().__init__()
        self.b_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True, batch_first=True)
        # Since each p-lstm is also bidirectional, the output size
        # is x4 the input size given that adjacent time-steps are concatenated.
        self.p_lstms = nn.ModuleList()
        for i in range(layers):
            p_hidden_size = hidden_size * (2 ** i)
            self.p_lstms.append(
                nn.LSTM(
                    input_size=p_hidden_size,
                    hidden_size= p_hidden_size // 2,
                    batch_first=True,
                    bidirectional=True
                )
            )

    def forward(self, batch_x, seq_lengths):
        packed_x = pack_padded_sequence(batch_x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_b_out, _ = self.b_lstm.forward(packed_x)
        padded_b_out, _ = pad_packed_sequence(packed_b_out, batch_first=True)
        p_input, p_size = padded_b_out, seq_lengths
        for p_lstm in self.p_lstms:
            p_input, p_size = p_lstm.forward(p_input, p_size)
        return p_input, p_size


class Listener(nn.Module):
    """
    Listener consists of 1D cnn and some specified layers of lstms
    """

    def __init__(self, lstm_layers, input_size, initial_hidden_size, reduce=None):
        super().__init__()
        # The first layer doesn't reduce the number of times steps/
        # The rest each reduces the number of time steps by a factor


