import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import softmax
from phonetics import SOS_TOKEN


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
                    hidden_size=p_hidden_size // 2,
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
        boolean_mask = torch.tile(
            torch.arange(0, max_length, device=query.device)[None, :],
            (batch_size, 1)
        )[:, :, None] # (batch_size, max_length, 1). Expand to allow broadcasting on the last dim
        boolean_mask = boolean_mask.lt(torch.tensor(batch_seq_lengths).to(query.device)[:, None, None])
        print(boolean_mask.shape)
        masked_embedding = embedding_seq * boolean_mask

        # For each query in the batch, compute
        # its context.
        keys = self._key_mlp.forward(masked_embedding)  # (batch, max_length, key_dim)
        values = self._value_mlp.forward(masked_embedding)  # (batch, max_length, val_dim)
        weights = softmax(
            (keys *  query[:, None, :]).sum(dim=1) / torch.sqrt(torch.tensor(hidden_size)).to(query.device),
            dim=1
        )
        print(weights.shape)
        print(values.shape)
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
        self.output_size = output_size
        self.char_embedding_size = char_embedding_size
        self.seq_embedding_size = seq_embedding_size
        self.attend_layer = Attention(hidden_size, hidden_size)
        self.char_embedding = nn.Embedding(output_size, char_embedding_size)
        self.decoder = nn.LSTMCell(hidden_size + seq_embedding_size, hidden_size)
        self.cdn = nn.Linear(hidden_size + seq_embedding_size, output_size)

    def spell_step(self, batch_prev_y, hx, prev_context):
        # TODO: use 2 LSTMCell as per the paper
        y_embeddings = self.char_embedding.forward(batch_prev_y)
        lstm_inputs = torch.concat([y_embeddings, prev_context], dim=1)
        hx = self.decoder.forward(lstm_inputs, hx)
        return hx

    def forward(self, seq_embeddings, seq_embedding_lengths, batch_y=None, tf_rate=1.0):
        batch_size = seq_embeddings.shape[0]
        max_decode_length = 600 if batch_y is None else batch_y.shape[1]
        prev_y = torch.zeros(batch_size, dtype=torch.long, device=seq_embeddings.device)
        prev_y[:] = SOS_TOKEN
        hx = None
        prev_context, _ = self.attend_layer.forward(
            torch.zeros(
                batch_size,
                self.hidden_size,
                device=seq_embeddings.device,
                dtype=seq_embeddings.dtype
            ),
            seq_embeddings,
            seq_embedding_lengths
        )
        output_symbols = torch.zeros(batch_size, max_decode_length, dtype=torch.long, device=seq_embeddings.device)
        output_logits = torch.zeros(
            batch_size,
            max_decode_length,
            self.output_size,
            dtype=seq_embeddings.dtype,
            device=seq_embeddings.device
        )
        for i in range(1, max_decode_length):
            hx = self.spell_step(prev_y, hx, prev_context)
            current_context, _ = self.attend_layer.forward(hx[0], seq_embeddings, seq_embedding_lengths)
            cdn_inputs = torch.concat([hx[0], current_context], dim=1)
            cdn_out_i = self.cdn.forward(cdn_inputs)  # (batch, output_size)
            output_logits[:, i, :] = cdn_out_i
            y_i = self.random_decode(cdn_out_i)
            output_symbols[:, i] = y_i
            prev_context = current_context
            if random.random() < tf_rate and batch_y is not None:
                prev_y = batch_y[:, i]
            else:
                prev_y = y_i

        return output_logits, output_symbols

    @staticmethod
    def random_decode(cdn_out):
        probs = torch.softmax(cdn_out, dim=-1)
        samples = torch.multinomial(probs, 1).squeeze(-1)
        return samples


class LAS(nn.Module):
    def __init__(self, char_embedding_size, seq_embedding_size, output_size, plstm_layers, teacher_force_rate):
        """
        A composite model that consists of a listner and a speller.

        Note: Each layer of plstm reduces the number of input time-step by
        a factor of and increase final sequence embedding size by a factor of 2.
        So the sequence embedding size output by the lister is `seq_embedding_size` * (2 ** plstm_layers)

        :param char_embedding_size:  Size for each input character embedding
        :param seq_embedding_size: Size for each sequence embedding.
        :param output_size: Size of the vocabulary bag.
        :param plstm_layers: Number of pyramid lstm layers.
        :param teacher_force_rate: Initial teacher force rate during training.
        """
        super().__init__()
        self.listener = Listener(15, seq_embedding_size, plstm_layers)
        self.speller = Speller(
            seq_embedding_size * (2 ** plstm_layers),
            char_embedding_size,
            char_embedding_size,
            output_size
        )
        self.tf_rate = teacher_force_rate

    def forward(self, seq_x, seq_lengths, seq_y=None):
        seq_embeddings, seq_embeddings_lengths = self.listener.forward(seq_x, seq_lengths)
        logits, symbols = self.speller.forward(seq_embeddings, seq_embeddings_lengths, seq_y, self.tf_rate)
        return logits, symbols
