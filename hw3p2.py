import os
import sys

import numpy as np
import torch
import argparse
import pandas as pd
import Levenshtein
import torch.nn.functional as F
import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from utils import greedy_search, beam_search_batch
from torch import nn
from torchaudio.transforms import TimeMasking, FrequencyMasking, SlidingWindowCmn

from torch.nn.functional import log_softmax, softmax
from torch.utils.data import Dataset, DataLoader
from phonemes import PHONEME_MAP, PHONEMES
from ctcdecode import CTCBeamDecoder


logger = logging.getLogger()
file_handler = logging.FileHandler("training.log")
logger.addHandler(file_handler)
logger.setLevel("DEBUG")

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


training_batch_size = 32
val_batch_size = 128
test_batch_size = 128  # Save memory for beam sear64


def get_labeled_data_loader(x_dir, y_dir, batch_size):
    train_set = LabeledDataset(x_dir, y_dir)
    train_args = {"shuffle": True, "collate_fn": LabeledDataset.collate_fn}
    train_loader = DataLoader(train_set, **train_args, pin_memory=True, num_workers=4, batch_size=batch_size)
    return train_loader


def get_unlabeled_data_loader(x_dir, data_order_dir, batch_size):
    test = TestDataSet(x_dir, data_order_dir)
    test_args = {"batch_size": batch_size, "collate_fn": TestDataSet.collate_fn}
    test_loader = DataLoader(test, shuffle=False, **test_args, num_workers=4, pin_memory=True)
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


def freeze_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False


def unfreeze_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True


def alter_mlp_layer(model, mlp_module):
    """
    Alter the specific layer of a pretrained model
    and freeze the rest.

    :param model:
    :param mlp_module:
    :return: A function that unfreeze old layers when needed
    """
    model.mlp = mlp_module
    freeze_layer(model.embed_layer)
    freeze_layer(model.lstm_layer)
    def unfreeze_other():
        unfreeze_layer(model.embed_layer)
        unfreeze_layer(model.lstm_layer)

    return unfreeze_other

def alter_lstm_layer(model, lstm_module):
    model.lstm_layer = lstm_module
    freeze_layer(model.mlp)
    freeze_layer(model.embed_layer)

    def unfreeze_other():
        unfreeze_layer(model.mlp)
        unfreeze_layer(model.embed_layer)

    return unfreeze_other

def alter_embedding_layer(model, embed_module):
    model.embed_layer = embed_module
    freeze_layer(model.lstm_layer)
    freeze_layer(model.mlp)
    def unfreeze_other():
        unfreeze_layer(model.mlp)
        unfreeze_layer(model.lstm_layer)
    return unfreeze_other


class ResidualBlock1D(torch.nn.Module):
    """"
    Residual block that makes up the embedding layer
    """

    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(output_channels)
        )

        # Transform the input to match the size of the output
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_layer(x)
        residual = self.shortcut(x)
        return torch.nn.functional.relu(out + residual)


class GaussianNoise(torch.nn.Module):
    def __init__(self, p=0.3, mean=0, std=0.5):
        super(GaussianNoise, self).__init__()
        self.p = p
        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Add noise to a fraction of p samples
        :param x:
        :return:
        """
        noise = torch.randn(x.size()).to(device)
        batch_size = x.size()[0]
        noise = noise * self.std + self.mean
        mask = torch.bernoulli(torch.ones(batch_size, dtype=torch.float) * torch.tensor(0.3)).to(device)
        # Unsqueeze to match the dimension of x
        for _ in range(len(x.size()) - 1):
            mask = mask[:, None]

        mask = mask.broadcast_to(x.size())
        x = x + noise * mask
        return x


class EmbeddingLayer0(torch.nn.Module):
    def __init__(self):
        """
        Increase embedding size to 256. No dropout
        """
        super(EmbeddingLayer0, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 128,3),
            ResidualBlock1D(128, 256, 3)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer0Flat(torch.nn.Module):
    def __init__(self):
        """
        Increase embedding size to 256. No dropout
        """
        super(EmbeddingLayer0Flat, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 128,3),
            ResidualBlock1D(128, 256, 3)
        )

    def forward(self, x):
        return self.layers(x)

class EmbeddingLayer1(torch.nn.Module):
    """
    2x down-sampling cnn
    """
    def __init__(self):
        super(EmbeddingLayer1, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 128,3),
            ResidualBlock1D(128, 256, 3, 2)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer1Down4(nn.Module):
    """
    4x down-sampling cnn
    """
    def __init__(self):
        super(EmbeddingLayer1Down4, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 128, 3),
            ResidualBlock1D(128, 256, 3, 2),
            ResidualBlock1D(256, 512, 3, 2),
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer1Shallower(torch.nn.Module):
    """
    2x down-sampling cnn
    """
    def __init__(self):
        super(EmbeddingLayer1Shallower, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 128, 3),
            ResidualBlock1D(128, 256, 5, 2),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer1Deeper(torch.nn.Module):
    """
    2x down-sampling cnn
    """
    def __init__(self):
        super(EmbeddingLayer1Deeper, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 64, 3),
            ResidualBlock1D(64, 128,3),
            ResidualBlock1D(128, 128,3),
            ResidualBlock1D(128, 256, 5, 2), 
            ResidualBlock1D(256, 256, 3), 
        )

    def forward(self, x):
        return self.layers(x)

class EmbeddingLayer1Wide(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLayer1Wide, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 128, 3),
            ResidualBlock1D(128, 256, 5, 2),
            ResidualBlock1D(256, 512, 3),
            ResidualBlock1D(512, 1024, 3)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer2(torch.nn.Module):
    """
    A deeper embedding with 4x down sample
    """
    def __init__(self):
        super(EmbeddingLayer2, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 64, 3),
            ResidualBlock1D(64, 128, 3, 2),
            ResidualBlock1D(128, 128, 3),
            ResidualBlock1D(128, 256, 3, 2),
            ResidualBlock1D(256, 256, 3)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer3(torch.nn.Module):
    """
    2x down-sampling cnn with dropout
    """
    def __init__(self):
        super(EmbeddingLayer3, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 128,3),
            nn.Dropout(0.2),
            ResidualBlock1D(128, 256, 3, 2),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayer4(torch.nn.Module):
    """
    2x down-sampling cnn
    """
    def __init__(self):
        super(EmbeddingLayer4, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock1D(13, 32, 3),
            ResidualBlock1D(32, 32, 3),
            ResidualBlock1D(32, 64, 3),
            ResidualBlock1D(64, 64, 3),
            ResidualBlock1D(64, 128,3),
            ResidualBlock1D(128, 128, 3),
            ResidualBlock1D(128, 256, 3),
            ResidualBlock1D(256, 256, 3),
            ResidualBlock1D(256, 512, 3, 2)
        )

    def forward(self, x):
        return self.layers(x)


class EmbeddingLayerWide(torch.nn.Module):
    """
    A very wide embedding layer
    """
    def __init__(self):
        super(EmbeddingLayerWide, self).__init__()
        self.layer = nn.Sequential(
               ResidualBlock1D(13, 256, 3),
               ResidualBlock1D(256, 512, 3),
               ResidualBlock1D(512, 1024, 5, 2),
       )

    def forward(self, x):
        return self.layer(x)



class MLPLayer(torch.nn.Module):
    def __init__(self, input_size, mlp_size, dropout, **kwargs):
        super(MLPLayer, self).__init__()
        mlp = [nn.Linear(input_size, mlp_size[0]), nn.ReLU(), nn.Dropout(dropout)]
        mlp_size = mlp_size[1:]
        for i in range(len(mlp_size) - 1):
            mlp.append(nn.Linear(mlp_size[i], mlp_size[i + 1]))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))

        mlp.append(nn.Linear(mlp_size[-1], 41))
        self.layer = nn.Sequential(*mlp)

    def forward(self, x):
        return self.layer(x)


class Model2(torch.nn.Module):
    def __init__(self, dropout, **kwargs):
        super(Model2, self).__init__()
        self.embed_layer = EmbeddingLayer0()
        self.lstm_layer = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=dropout)

        self.mlp = MLPLayer(256 * 2, [2048, 2048], dropout)

    def forward(self, x, seq_lengths):
        embeddings = self.embed_layer(x.transpose(1, 2))
        packed_embed = pack_padded_sequence(embeddings.transpose(1, 2), seq_lengths, batch_first=True,
                                            enforce_sorted=False)

        seq_out, _ = self.lstm_layer.forward(packed_embed)  # (N, L, D*H_out)
        # Unpack
        unpacked_seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        out = self.mlp.forward(unpacked_seq_out)

        # Shape of (N, L, 41)
        return out, seq_lengths


class Model3(Model2):
    def __init__(self, dropout, **kwargs):
        super(Model3, self).__init__(dropout, **kwargs)
        self.lstm_layer = nn.LSTM(input_size=256, hidden_size=256, num_layers=4, bidirectional=True, batch_first=True,
                                  dropout=dropout)


class Model4(Model3):
    def __init__(self, dropout, **kwargs):
        super(Model4, self).__init__(dropout, **kwargs)
        self.embed_layer = EmbeddingLayer1()

    def forward(self, x, seq_lengths):
        down_sampled_lenghts = [(l - 3) // 2 + 1 for l in seq_lengths]
        out, _ = super(Model4, self).forward(x, down_sampled_lenghts)
        return out, down_sampled_lenghts


class Model4Noisy(Model4):
    def __init__(self, dropout, **kwargs):
        super(Model4Noisy, self).__init__(dropout, **kwargs)
        self.noise = GaussianNoise()

    def forward(self, x, seq_lengths):
        if self.training:
            x = self.noise(x)
        return super(Model4Noisy, self).forward(x, seq_lengths)


class Model4NoisyMasked(Model4Noisy):
    def __init__(self, dropout, noise_std, noise_prob, freq_mask, time_mask, **kwargs):
        super(Model4NoisyMasked, self).__init__(dropout, **kwargs)
        # Apply both time mask and frequency mask
        self.norm = SlidingWindowCmn(cmn_window=300, min_cmn_window=50)
        self.mask = nn.Sequential(
            FrequencyMasking(freq_mask),
            TimeMasking(time_mask)
        )
        self.lstm_layer = nn.LSTM(input_size=512, hidden_size=512, num_layers=4, bidirectional=True, batch_first=True, dropout=dropout)

        self.embed_layer = EmbeddingLayer1Down4()
        self.noise = GaussianNoise(p=noise_prob, std=noise_std)
        self.mlp = MLPLayer(1024, [4096, 4096, 4096, 4096], dropout)

    def forward(self, x, seq_lengths):
        seq_lengths = [((l - 1) // 2 + 1)  for l in seq_lengths]
        seq_lengths = [((l - 1) // 2 + 1)  for l in seq_lengths]
        # x = self.norm.forward(x)
        if self.training:
            x = self.noise.forward(x)
            x = self.mask(x.transpose(1, 2)).transpose(1, 2)

        embeddings = self.embed_layer(x.transpose(1, 2))
        packed_embed = pack_padded_sequence(embeddings.transpose(1, 2), seq_lengths, batch_first=True,
                                            enforce_sorted=False)

        seq_out, _ = self.lstm_layer.forward(packed_embed)  # (N, L, D*H_out)
        # Unpack
        unpacked_seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        out = self.mlp.forward(unpacked_seq_out)

        # Shape of (N, L, 41)
        return out, seq_lengths


class Model4NoisyMaskedWide(Model4NoisyMasked):
    def __init__(self, dropout, **kwargs):
        super(Model4NoisyMaskedWide, self).__init__(dropout, **kwargs)
        self.embed_layer = EmbeddingLayer1Wide()
        self.mask = nn.Sequential(
                FrequencyMasking(2),
                TimeMasking(100)
        )
        self.noise = GaussianNoise(std=0.4)
        self.lstm_layer = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=4, bidirectional=True, batch_first=True,
                                  dropout=dropout)
        self.mlp = MLPLayer(1024 * 2, [2048, 2048], dropout)

    def forward(self, x, seq_lengths):
        return super(Model4NoisyMaskedWide, self).forward(x, seq_lengths)


class Model5(Model3):
    def __init__(self, dropout, **kwargs):
        super(Model5, self).__init__(dropout, **kwargs)
        self.embed_layer = EmbeddingLayer3()

    def forward(self, x, seq_lengths):
        down_sampled_lenghts = [(l - 3) // 4 + 1 for l in seq_lengths]
        out, _ = super(Model5, self).forward(x, down_sampled_lenghts)
        return out, down_sampled_lenghts


class Model6(Model4Noisy):
    def __init__(self, dropout, **kwargs):
        super(Model6, self).__init__(dropout, **kwargs)
        self.mlp = MLPLayer(256 * 2, [2048, 2048, 2048], dropout)

    def forward(self, x, seq_lengths):
        return super(Model6, self).forward(x, seq_lengths)


class Model7(Model6):
    def  __init__(self, dropout, **kwargs):
        super(Model7, self).__init__(dropout, **kwargs)
        self.embed_layer = EmbeddingLayer4()
        self.noise = GaussianNoise(std=0.4)
        self.lstm_layer = nn.LSTM(input_size=512, hidden_size=512, num_layers=6, bidirectional=True, batch_first=True,
                                  dropout=dropout)
        self.mlp = MLPLayer(512 * 2, [4096, 4096, 4096], dropout)

    def forward(self, x, seq_lengths):
       return super(Model7, self).forward(x, seq_lengths)


# class Model4(Model3):
#     def __init__(self, dropout, **kwargs):
#         """
#         Noisy version of Model3
#         :param dropout:
#         :param kwargs:
#         """
#         super(Model4, self).__init__(dropout, **kwargs)
#         self.noise_layer = GaussianNoise()
#
#     def forward(self, x, seq_lengths):
#         if self.training:
#             x = self.noise_layer(x)
#
#         return super().forward(x, seq_lengths)


class Model0(torch.nn.Module):
    """
    Raw LSTM model without embedding layer
    """
    def __init__(self, *args, **kwargs):
        super(Model0, self).__init__()
        self.hidden_size = 256
        lstm_layers = 2
        self.bidirectional = True
        self.lstm = torch.nn.LSTM(
            input_size=13, hidden_size=self.hidden_size, num_layers=lstm_layers, bidirectional=True, batch_first=True,
            dropout=0.5
        )
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2048), torch.nn.ReLU(),

                                       torch.nn.Linear(2048, 41))

    def forward(self, x, seq_lengths):
        packed_x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        seq_out, _  = self.lstm.forward(packed_x)
        unpacked_seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        if self.bidirectional:
            unpacked_seq_out = (
                    unpacked_seq_out[:, :, self.hidden_size: ] + unpacked_seq_out[:, :, : self.hidden_size]
            )
        unpacked_seq_out = torch.relu(unpacked_seq_out)
        out = self.mlp.forward(unpacked_seq_out)

        # Return seq_lengths for interface compatibility
        return out, seq_lengths


class Network(nn.Module):

    def __init__(self, dropout_embed=0.15, dropout_lstm=0.35,
                 dropout_classification=0.2, *args, **kwargs):  # You can add any extra arguments as you wish

        super(Network, self).__init__()

        # Embedding layer converts the raw input into features which may (or may not) help the LSTM to learn better
        # For the very low cut-off you dont require an embedding layer. You can pass the input directly to the  LSTM
        self.embedding = nn.Sequential(nn.Conv1d(in_channels=13, out_channels=128, kernel_size=5, stride=2),
                                       nn.BatchNorm1d(128), nn.Dropout(dropout_embed), nn.ReLU(inplace=True),
                                       nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm1d(256), nn.Dropout(dropout_embed), nn.ReLU(inplace=True))

        self.lstm = nn.LSTM(256, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True,
                            dropout=dropout_lstm)  # TODO: # Create a single layer, uni-directional LSTM with hidden_size = 256
        # Use nn.LSTM() Make sure that you give in the proper arguments as given in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        self.classification = nn.Sequential(nn.Linear(512 * 2, 2048), nn.Dropout(dropout_classification), nn.ReLU(),
                                            nn.Linear(2048,
                                                      41))  # TODO: Create a single classification layer using nn.Linear()

    def forward(self, x, len_x):  # TODO: You need to pass atleast 1 more parameter apart from self and x

        # x is returned from the dataloader. So it is assumed to be padded with the help of the collate_fn
        # print("x ",x.shape )
        input_for_cnn = torch.permute(x, (0, 2, 1))
        # print("input_for_cnn ", input_for_cnn.shape)
        input_after_cnn = self.embedding(input_for_cnn)
        # print("input_after_cnn ", input_after_cnn.shape)
        x = torch.permute(input_after_cnn, (2, 0, 1))
        # print("input_after_cnn ", x.shape)
        len_x = torch.clamp(torch.tensor(len_x), max=x.shape[0])
        packed_input = pack_padded_sequence(x, len_x,
                                            enforce_sorted=False)  # TODO: Pack the input with pack_padded_sequence. Look at the parameters it requires

        out1, (out2, out3) = self.lstm(packed_input)  # TODO: Pass packed input to self.lstm
        # As you may see from the LSTM docs, LSTM returns 3 vectors. Which one do you need to pass to the next function?
        out, lengths = pad_packed_sequence(out1, batch_first=True)  # TODO: Need to 'unpack' the LSTM output using pad_packed_sequence
        # print("lengths", lengths)
        out = self.classification(out)  # TODO: Pass unpacked LSTM output to the classification layer
        # out_l = F.log_softmax(out, dim=2)  # Optional: Do log softmax on the output. Which dimension?
        # print("out_l", out_l.shape)
        # print("lengths", lengths.shape)
        return out, lengths.tolist()  # TODO: Need to return 2 variables


# BaseLine model
# cnn --> lstm --> mlp
class Model1(torch.nn.Module):
    def __init__(self, cnn_channels, n_lstm_layers, bidirectional, mlp_sizes, dropout, **kwargs):
        super(Model1, self).__init__()
        self.cnn_1 = torch.nn.Conv1d(in_channels=13, out_channels=cnn_channels, kernel_size=3, padding='same')
        self.cnn_2 = torch.nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3,
                                     stride=4)
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
        down_sampled_seq_length = [((x + 1) // 2 +1) //2 for x in seq_lengths]
        packed_embed = pack_padded_sequence(embed.transpose(1, 2), down_sampled_seq_length, batch_first=True,
                                            enforce_sorted=False)

        seq_out, _ = self.lstm.forward(packed_embed)  # (N, L, D*H_out)
        # Unpack
        unpacked_seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        out = self.mlp.forward(unpacked_seq_out)

        # Shape of (N, L, 41)
        return out, down_sampled_seq_length


def predict_string(model, batch_x, batch_seq_length, decoder):
    result, new_seq_length = predict(model, batch_x, batch_seq_length)
    beam_results, beam_scores, _, seq_len = decoder.decode(F.softmax(result, dim=-1), torch.tensor(new_seq_length))
    best_seqs = pick_and_translate_beams(beam_results, beam_scores, seq_len)
    return best_seqs


def output_result(model, beam_width=35):
    """
    Predict the result of our test set.
    Determine the output sequence by beam-searching.
    param test_data_loader:
    :param model:
    :return:
    """
    decoder = CTCBeamDecoder(PHONEME_MAP, beam_width=beam_width, num_processes=8, cutoff_top_n=41)
    test_data_loader = get_unlabeled_data_loader(test_data_dir, test_data_order_dir, test_batch_size)
    output_seqs = []
    for batch_x, batch_seq_length in test_data_loader:
        model.to("cpu")
        batch_string = predict_string(model, batch_x, batch_seq_length, decoder)
        output_seqs.extend(batch_string)

    ids = np.arange(len(output_seqs))
    data = np.vstack([ids, np.array(output_seqs)])
    df = pd.DataFrame(data=data.T, columns=['id', 'predictions'])
    df.to_csv("hw3p2_submission.csv", index=False)


def train_epoch(training_loader, model, criterion, optimizer, scaler, current_epoch, scheduler=None):
    total_training_loss = 0.0
    total_batches = len(training_loader)
    b = 0
    for (batch_x, batch_y), (batch_seq_lengths, batch_target_sizes) in training_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        model.to(device)
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs, new_seq_length = model(batch_x, batch_seq_lengths)
            # CTC loss requires (L B, C) so we transpose
            loss = criterion(log_softmax(outputs.transpose(0, 1), dim=2), batch_y, new_seq_length, batch_target_sizes)
        total_training_loss += float(loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            current_epoch = current_epoch + b / total_batches
            scheduler.step(current_epoch)
        b += 1
    training_loss_sum = "Training loss ", total_training_loss / len(training_loader)
    logger.debug(training_loss_sum)
    print(training_loss_sum)


def predict(model, x, x_lengths):
    model.eval()
    y, new_length = model(x, x_lengths)
    return y, new_length


def validate(model, validation_loader, criterion, decoder=None, compute_distance=False):
    total_loss = 0.0
    total_distance = 0.0
    total_size = 0
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
            if compute_distance:
                batch_string = predict_string(model, batch_x, batch_seq_lengths, decoder)
                for y_hat_string, y, target_length in zip(batch_string, batch_y, batch_target_lengths):
                    y_string = ''.join([PHONEME_MAP[c] for c in y[:target_length]])
                    dist =  Levenshtein.distance(y_hat_string, y_string)
                    total_distance += dist
            total_loss += (loss.item() * batch_size)

    # Return average loss
    return total_loss / total_size, total_distance / total_size


def get_model(model_name, model_param):
    if model_name == "Model1":
        pass


def get_scheduler(optimizer, scheduler):
    if scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, verbose=True)
    return scheduler


def train(model, optimizer_params):
    n_epochs = optimizer_params['n_epochs']

    lr = optimizer_params.get("lr")
    warmup_epoch = optimizer_params.get("warmup")
    if warmup_epoch is None:
        warmup_epoch = 150
    weight_decay = optimizer_params.get("weight_decay")
    if lr is None:
        lr = 0.002
    beam_width = optimizer_params.get("beam_width")
    if beam_width is None:
        beam_width = 15
    train_loader = get_labeled_data_loader(train_data_dir, train_label_dir, training_batch_size)
    dev_loader = get_labeled_data_loader(dev_data_dir, dev_label_dir, val_batch_size)
    validation_decoder = CTCBeamDecoder(PHONEME_MAP, beam_width=beam_width, cutoff_top_n=41)

    # When using pretrained model
    # unfreezer = alter_lstm_layer(model,
    #                             nn.LSTM(input_size=512, hidden_size=512, num_layers=8, bidirectional=True, batch_first=True, dropout=optimizer_params['dropout'])
    #                         )
        
    print(model)
    logger.debug(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, optimizer_params['scheduler'])
    scaler = torch.cuda.amp.GradScaler()
    ctc_loss = torch.nn.CTCLoss()

    for i in range(n_epochs):
        # Increase dropout by 0.1 every 15 epochs
        in_epoch_scheduler = scheduler if optimizer_params['scheduler'] == 'Cosine' else None
        train_epoch(train_loader, model, ctc_loss, optimizer, scaler, current_epoch=i, scheduler=in_epoch_scheduler)
        # if i == 15:
        #     unfreezer()

        if i % 10  == 0:
            torch.save(training_model.state_dict(), "saved_model_hw3p2" + optimizer_params['model'])

        if optimizer_params['scheduler'] == 'Cosine' and (i +1) % 5 != 0:
            validation_loss, validation_distance = None, None
        else:
            validation_loss, validation_distance = validate(model, dev_loader, ctc_loss, validation_decoder, True)
            # Early stop
#            if validation_loss < 0.229:
#                break

        validation_summary = f"Epoch {i}, validation_loss: {validation_loss}, validation_distance: {validation_distance}"
        print(validation_summary)
        logger.debug(validation_summary)
        # Warmup for the first 150 epochs:
        if (i + 1) % 5 == 0 and optimizer_params['scheduler'] == 'Cosine':
            scheduler.base_lrs[0] = scheduler.base_lrs[0] * 0.8
            print(f"Half the learning rate {scheduler.base_lrs}")
        elif optimizer_params['scheduler'] != 'Cosine':
            print("Other scheduler")
            scheduler.step(validation_distance)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'retrain', 'test'])
    parser.add_argument("model")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--model_path")
    parser.add_argument("--scheduler", default="StepLR")
    parser.add_argument("--scheduler step size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--noise_prob", type=float, default=0.3)
    parser.add_argument("--freq_mask", type=int, default=3)
    parser.add_argument("--time_mask", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=50)

    args = parser.parse_args()
    training_params = vars(args)
    torch.cuda.empty_cache()
    model_params = {"cnn_channels": 128, "n_lstm_layers": 4, "bidirectional": True, "mlp_sizes": [2048],
                    "dropout": args.dropout, "noise_std": args.noise_std}

    model_params_str = f"({', '.join(['='.join([str(k), str(v)]) for k, v in model_params.items()])})"
    model_name = args.model
    training_model = eval(model_name + model_params_str)
    logger.debug("===================================================================================================")
    logger.debug(training_params)
    print(training_params)

    path = args.model_path
    if path is None:
        path = "saved_model_hw3p2" + args.model

    if args.mode == 'train':
        training_model = train(training_model, training_params)
        torch.save(training_model.state_dict(), "saved_model_hw3p2" + args.model)
    elif args.mode == 'retrain':
        training_model.load_state_dict(torch.load(path))
        training_model = train(training_model, training_params)
        torch.save(training_model.state_dict(), "saved_model_hw3p2" + args.model)
    else:
        training_model.load_state_dict(torch.load(path))

    output_result(training_model)
