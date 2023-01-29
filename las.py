import numpy as np
import torch
import torch.nn as nn
import typing as t
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import softmax
from phonetics import SOS_TOKEN, EOS_TOKEN


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
        packed_out, _ = self.layer.forward(packed_data)
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
            device=batch_x.device
        )

        resized_lengths = []
        for i, length in enumerate(seq_lengths):
            # Drop last step if length is odd
            length = length - (length % 2)
            resized_length = length // 2
            resized = batch_x[i, :length, :].reshape((resized_length, hidden_size * 2))   # (L // 2, H * 2)
            batch_resized[i, :resized_length, :] = resized
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
            p_hidden_size = hidden_size * (2 ** (i + 1))
            self.p_lstms.append(
                PyramidLSTM(
                    input_size=p_hidden_size,
                    hidden_size= p_hidden_size // 2,
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


class Attention(nn.Module):
    def __init__(self, hidden_dim, key_dim):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._key_dim = key_dim
        self._key_mlp = nn.Linear(hidden_dim, key_dim)
        self._value_mlp = nn.Linear(hidden_dim, key_dim)

    def forward(self, query: torch.Tensor, embedding_seq: torch.Tensor, batch_seq_lengths: list):
        """
        :param query: tensor of size (batch, hidden)
        :param embedding_seq: (batch, max_length, hidden_dim)
        :param batch_seq_lengths: the length of each sequence in the batch
        :return:
        """
        batch_size, max_length, hidden_size = embedding_seq.shape
        keys = self._key_mlp.forward(embedding_seq)  # (batch, max_length, key_dim)
        values = self._value_mlp.forward(embedding_seq)  # (batch, max_length, val_dim)
        weights = torch.zeros(batch_size, max_length, device=query.device, dtype=query.dtype, requires_grad=True)
        # For each query in the batch, compute
        # its context.
        for b in range(batch_size):
            seq_length = batch_seq_lengths[b]
            query_b = query[b][:, None]
            key_b = keys[b, :seq_length, :]
            weights_b = softmax(torch.matmul(key_b, query_b) / torch.sqrt(hidden_size), dim=0) # (seq_length, 1)
            weights[b, :seq_length] = weights_b.squeeze()
        context = (values * weights[:, :, None]).sum(dim=1)
        return context, weights


class Listener(nn.Module):
    """
    Listener consists of 1D cnn and some specified layers of lstms
    """

    def __init__(self, input_size, initial_hidden_size, reduce_factor):
        super().__init__()
        # The first layer doesn't reduce the number of times steps/
        # The rest each reduces the number of time steps by a factor
        self.cnn = torch.nn.Sequential(
            ResidualBlock1D(input_size, initial_hidden_size, kernel_size=3),
            ResidualBlock1D(initial_hidden_size, initial_hidden_size, kernel_size=3)
        )
        self.pyramid_encoder = PyLSTMEncoder(initial_hidden_size, initial_hidden_size, reduce_factor)

    def forward(self, x, seq_lenghts):
       embeded_seq, embeded_seq_length = self.pyramid_encoder.forward(
           self.cnn.forward(x.transpose(1, 2)).transpose(1, 2), seq_lenghts
       )
       return embeded_seq, embeded_seq_length


class Speller(nn.Module):
    def __init__(self, seq_embedding_size, char_embedding_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size
        self.seq_embedding_size = seq_embedding_size
        self.attend_layer = Attention(hidden_size, hidden_size)
        self.char_embedding = nn.Embedding(output_size, char_embedding_size)
        self.decoder = nn.LSTMCell(char_embedding_size, hidden_size)
        self.cdn = nn.Linear(hidden_size + seq_embedding_size, output_size)

    def spell_step(self, batch_prev_y, hx, prev_context):
        # TODO: use 2 LSTMCell as per the paper
        y_embeddings = self.char_embedding.forward(batch_prev_y)
        lstm_inputs = torch.concat([y_embeddings, prev_context], dim=1)
        lstm_out, hx = self.decoder.forward(lstm_inputs, hx)
        return hx

    def forward(self, seq_embeddings, seq_embedding_lengths):
        if self.training:
            raise ValueError("This class is used only under evaluation mode")
        batch_size = seq_embeddings.shape[0]
        max_decode_length = 600
        prev_y = torch.zeros(batch_size, self.hidden_size, dtype=seq_embeddings.dtype, device=seq_embeddings.device)
        hx = None
        prev_context = self.attend_layer.forward(
            torch.zeros(batch_size, self.hidden_size),
            seq_embeddings,
            seq_embedding_lengths
        )
        packed_seq_embeddings = pack_padded_sequence(
            seq_embeddings,
            seq_embedding_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        for i in range(max_decode_length):
            query_i = self.spell_step(prev_y, hx, prev_context)
            prev_context = self.attend_layer.forward(query_i, seq_embeddings, seq_embedding_lengths)



    @staticmethod
    def random_decode(seq_distribution):
        pass


