import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import softmax


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

    def __init__(self, lstm_layers, input_size, initial_hidden_size, reduce=None):
        super().__init__()
        # The first layer doesn't reduce the number of times steps/
        # The rest each reduces the number of time steps by a factor
