import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch.nn.functional import softmax, gumbel_softmax
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


class LockedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self._p = p

    def forward(self, batch_x):
        if not self.training:
            return batch_x

        batch_size, max_length, hidden_size = batch_x.shape
        mask = batch_x.new_empty(batch_size, 1, hidden_size, requires_grad=False).bernoulli(1 - self._p)
        batch_x = mask * batch_x
        batch_x = batch_x / (1 - self._p)
        return batch_x


class PyramidLSTM(nn.Module):
    """
    The pyramid-lstm block reduce input's time-step by half and feed into a
    bidirectional lstm layer
    """
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.dropout_layer = nn.Sequential()
        if dropout > 0:
            self.dropout_layer.append(LockedDropout(dropout))

    def forward(self, batch_x, seq_lengths):
        # Reduce the sequence length by 2
        batch_x_resized, seq_lengths_resized = self.reduce_length(batch_x, seq_lengths)
        packed_data = pack_padded_sequence(batch_x_resized, seq_lengths_resized, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.layer.forward(packed_data)
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        padded_out = self.dropout_layer.forward(padded_out)
        return padded_out, seq_lengths_resized

    @staticmethod
    def reduce_length(batch_x: torch.tensor, seq_lengths) -> (torch.Tensor, torch.Tensor):
        batch_size, max_length, hidden_size = batch_x.shape
        # With num of time steps reduced by a number of 2
        # and hidden sizes increased by a number of 2.
        max_length = max_length - (max_length % 2)
        reduced_batch_x = batch_x[:, : max_length, :].contiguous().reshape(batch_size, max_length // 2, hidden_size * 2)
        reduced_seq_lengths = seq_lengths // 2

        return reduced_batch_x, reduced_seq_lengths


class PyLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        """
        The final output size is (hidden_size * (2 ** layers ))
        :param input_size:
        :param hidden_size:
        :param layers:
        """
        super().__init__()
        self.b_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True, batch_first=True)
        # Since each p-lstm is also bidirectional, the output size
        # is x4 the specified hidden size given that adjacent time-steps are concatenated.
        # the final output is the size of hidden_size
        self.p_lstms = nn.ModuleList()
        for i in range(layers):
            self.p_lstms.append(
                PyramidLSTM(
                    input_size=hidden_size * 2,
                    hidden_size=hidden_size // 2,
                    dropout=dropout
                )
            )
        self.locked_dropout = nn.Sequential()
        if dropout > 0:
            self.locked_dropout.append(LockedDropout(dropout))

    def forward(self, batch_x, seq_lengths):
        packed_x = pack_padded_sequence(batch_x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_b_out, _ = self.b_lstm.forward(packed_x)
        padded_b_out, _ = pad_packed_sequence(packed_b_out, batch_first=True)
        padded_b_out = self.locked_dropout.forward(padded_b_out)
        p_input, p_size = padded_b_out, seq_lengths
        for p_lstm in self.p_lstms:
            p_input, p_size = p_lstm.forward(p_input, p_size)
        return p_input, p_size


class Attention(nn.Module):
    def __init__(self, hidden_dim, key_dim, val_dim):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._key_dim = key_dim
        self._val_dim = val_dim
        self._key_mlp = nn.Linear(hidden_dim, key_dim)
        self._value_mlp = nn.Linear(hidden_dim, val_dim)
        self._query_mlp = nn.Linear(hidden_dim, key_dim)
        self._mask = None

    def set_mask(self, embedding_seq, embedding_seq_lengths):
        """
        Since each embedding seq is used in a loop
        when decoding, so is the mask.

        :return:
        """
        batch_size, max_length, hidden_size = embedding_seq.shape

        mask = torch.arange(0, max_length, device=embedding_seq.device)[None, :]  # (1, max_length)
        mask = mask >= embedding_seq_lengths[:, None].to(embedding_seq.device)
        self._mask = mask

    def _get_weights(self, query, embedding_seq, batch_seq_lengths):
        # For each query in the batch, compute
        # its context.
        _, hidden_size = query.shape
        energy = torch.bmm(
            self._key_mlp.forward(embedding_seq),
            self._query_mlp.forward(query)[:, :, None]
        )  # (B, T, 1)

        energy = energy.squeeze(2)  # (B, T)
        filling_value = -1e+30 if energy.dtype == torch.float32 else -1e+4
        energy = energy.masked_fill(self._mask, filling_value)
        weights = softmax(energy / torch.sqrt(torch.tensor(hidden_size)).to(query.device), dim=1)

        return weights

    def forward(self, query: torch.Tensor, embedding_seq: torch.Tensor, batch_seq_lengths: list):
        """
        :param query: tensor of size (batch, hidden)
        :param embedding_seq: (batch, max_length, hidden_dim)
        :param batch_seq_lengths: the length of each sequence in the batch
        :return:
        """

        weights = self._get_weights(query, embedding_seq, batch_seq_lengths)
        context = torch.bmm(weights[:, None, :], self._value_mlp.forward(embedding_seq)).squeeze(1)

        return context, weights


class ConvTransposed1DBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=3, stride=2),
            ResidualBlock1D(output_channels, output_channels, kernel_size=3),
            ResidualBlock1D(output_channels, output_channels, kernel_size=3)
        )

    def forward(self, batch_x):
        return self.layer.forward(batch_x)


class SelfDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, increasing_factor):
        super().__init__()
        self.layer = nn.Sequential()
        for _ in range(increasing_factor):
            self.layer.append(ConvTransposed1DBlock(input_channels, input_channels // 2))
            input_channels = input_channels // 2
        self.layer.append(ResidualBlock1D(input_channels, output_channels, kernel_size=3))
        self.linear = nn.Linear(output_channels, output_channels)

    def forward(self, batch_x):
        raw = self.layer.forward(batch_x)
        return self.linear.forward(raw.transpose(1, 2)).transpose(1, 2)


class Listener(nn.Module):
    """
    Listener consists of 1D cnn and some specified layers of lstms
    """

    def __init__(self, input_size, initial_hidden_size, reduce_factor, dropout):
        super().__init__()
        # The first layer doesn't reduce the number of times steps/
        # The rest each reduces the number of time steps by a factor
        self.cnn = torch.nn.Sequential(
            ResidualBlock1D(input_size, initial_hidden_size, kernel_size=3),
            ResidualBlock1D(initial_hidden_size, initial_hidden_size, kernel_size=3)
        )
        self.pyramid_encoder = PyLSTMEncoder(initial_hidden_size, initial_hidden_size, reduce_factor, dropout)

    def forward(self, x, seq_lenghts):
        embeded_seq, embeded_seq_length = self.pyramid_encoder.forward(
            self.cnn.forward(x.transpose(1, 2)).transpose(1, 2), seq_lenghts
        )
        return embeded_seq, embeded_seq_length


class Speller(nn.Module):
    def __init__(self, embedding_size, context_size, output_size, dropout):
        super().__init__()

        self.hidden_size = embedding_size
        self.output_size = output_size
        self.context_size = context_size
        self.attend_layer = Attention(embedding_size, embedding_size, self.context_size)
        self.char_embedding = nn.Embedding(output_size, embedding_size)
        self.decoder = nn.LSTM(
            input_size=self.context_size + embedding_size,
            hidden_size=embedding_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.transformation = nn.Sequential(
            nn.Linear(self.context_size + embedding_size, embedding_size),
            nn.ReLU()
        )
        self.cdn = nn.Linear(embedding_size, output_size)

        # Weight tying
        self.cdn.weight = self.char_embedding.weight

    def spell_step(self, batch_prev_y, hx, prev_context, gumble=False):
        if gumble:
            # During training, the previous y may be sampled
            # after gumble-softmax
            y_embeddings = batch_prev_y @ self.char_embedding.weight
        else:
            y_embeddings = self.char_embedding.forward(batch_prev_y)

        lstm_inputs = torch.concat([y_embeddings, prev_context], dim=1)
        output, hx = self.decoder.forward(lstm_inputs[:, None, :])
        return output.squeeze(1), hx

    def forward(self, seq_embeddings, seq_embedding_lengths, batch_y=None, tf_rate=1.0):
        batch_size = seq_embeddings.shape[0]
        max_decode_length = 600 if batch_y is None else batch_y.shape[1]
        prev_y = torch.zeros(batch_size, dtype=torch.long, device=seq_embeddings.device)
        prev_y[:] = SOS_TOKEN
        hx = None
        self.attend_layer.set_mask(seq_embeddings, seq_embedding_lengths)

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

        output_logits_seq = []
        gumble = False
        output_char_seq = None if self.training else []  # Output character-sequence when in validation mode.
        for i in range(1, max_decode_length):
            torch.cuda.empty_cache()
            spell_out, hx = self.spell_step(prev_y, hx, prev_context, gumble=gumble)
            current_context, _ = self.attend_layer.forward(spell_out, seq_embeddings, seq_embedding_lengths)
            cdn_inputs = torch.concat([spell_out, current_context], dim=1)
            cdn_out_i = self.cdn.forward(self.transformation.forward(cdn_inputs))  # (batch, output_size)
            output_logits_seq.append(cdn_out_i)
            y_i_gumble = gumbel_softmax(cdn_out_i, dim=1)
            prev_context = current_context

            if self.training:
                if random.random() < tf_rate and batch_y is not None:
                    gumble = False
                    prev_y = batch_y[:, i]
                else:
                    gumble = True
                    prev_y = y_i_gumble
            # Validation
            else:
                gumble = False
                prev_y = self.random_decode(cdn_out_i)
                output_char_seq.append(prev_y)

        if not self.training:
            output_char_seq = torch.stack(output_char_seq, dim=1)

        return torch.stack(output_logits_seq, dim=1), output_char_seq

    @staticmethod
    def random_decode(cdn_out):
        probs = torch.softmax(cdn_out, dim=-1)
        samples = torch.multinomial(probs, 1).squeeze(-1)
        return samples

    @staticmethod
    def greedy_decode(cdn_out):
        return torch.argmax(cdn_out, dim=1)


class LAS(nn.Module):
    def __init__(
            self,
            embedding_size,
            context_size,
            output_size,
            plstm_layers,
            teacher_force_rate,
            encoder_dropout,
            decoder_dropout,
            freq_mask,
            time_mask
    ):
        """
        A composite model that consists of a listner and a speller.

        Note: Each layer of plstm reduces the number of input time-step by
        a factor of and increase final sequence embedding size by a factor of 2.
        So the sequence embedding size output by the lister is `seq_embedding_size` * (2 ** plstm_layers)

        :param output_size: Size of the vocabulary bag.
        :param plstm_layers: Number of pyramid lstm layers.
        :param teacher_force_rate: Initial teacher force rate during training.
        :param encoder_dropout: dropout rate for the encoder
        """
        super().__init__()
        self.listener = Listener(15, embedding_size, plstm_layers, encoder_dropout)
        self.speller = Speller(
            embedding_size,
            context_size,
            output_size,
            decoder_dropout
        )

        self.mask = nn.Sequential()
        if freq_mask > 0:
            self.mask.append(FrequencyMasking(freq_mask, iid_masks=True))
        if time_mask > 0:
            self.mask.append(TimeMasking(time_mask, iid_masks=True))
        self.tf_rate = teacher_force_rate

        # From the LAS paper. Weights are initialized by Uniform(-0.1, 0.1)
        for param in self.listener.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

        for param in self.speller.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, seq_x, seq_lengths, seq_y=None):
        if self.training:
            seq_x = self.mask.forward(seq_x.transpose(1, 2)).transpose(1, 2)
        seq_embeddings, seq_embeddings_lengths = self.listener.forward(seq_x, seq_lengths.cpu())
        logits, chars = self.speller.forward(seq_embeddings, seq_embeddings_lengths, seq_y, self.tf_rate)
        return logits, chars
