import random

import torch.nn
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import SOS_TOKEN, DEVICE


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
        # print(x.shape)
        out = self.conv_layer(x)
        residual = self.shortcut(x)
        # print(out.shape)
        return torch.nn.functional.relu(out + residual)


class DownSampleBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self._input_size = input_size

        self._layer = nn.Sequential(
            ResidualBlock1D(self._input_size, self._input_size, kernel_size=3),
            ResidualBlock1D(self._input_size, self._input_size, kernel_size=3),
            ResidualBlock1D(self._input_size, self._input_size, kernel_size=3),
            ResidualBlock1D(self._input_size, self._input_size * 2, kernel_size=3, stride=2),
        )

    def forward(self, inputs):
        return self._layer.forward(inputs)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, projection_size, max_seq_len=176):
        super().__init__()
        # Read the Attention Is All You Need paper to learn how to code code the positional encoding
        self._position_encodings = torch.zeros(size=(max_seq_len, projection_size), dtype=torch.float)
        positions = torch.arange(max_seq_len)
        dimensions = torch.arange(projection_size // 2)
        xs = positions[:, None] / (10000 ** (2 * dimensions[None, :] / projection_size))
        self._position_encodings[:, ::2] = torch.sin(xs)
        self._position_encodings[:, 1::2] = torch.cos(xs)
        self._position_encodings = nn.Parameter(self._position_encodings, requires_grad=False)

    def forward(self, x):
        _, x_max_length, _ = x.shape
        return x + self._position_encodings[:x_max_length, :][None, ...]


class TransformerEncoder(torch.nn.Module):
    def __init__(self, projection_size, num_heads):
        super().__init__()

        # create the key, query and value weights
        # self.KW         = # TODO
        # self.VW         = # TODO
        # self.QW         = # TODO
        self._kw = nn.Linear(projection_size, projection_size, bias=False)
        self._vw = nn.Linear(projection_size, projection_size, bias=False)
        self._qw = nn.Linear(projection_size, projection_size, bias=False)

        # self.permute    = PermuteBlock()

        # Compute multihead attention. You are free to use the version provided by pytorch
        # self.attention = nn.MultiheadAttention(projection_size, num_heads=num_heads, batch_first=True)
        self.attention = MultiHeadAttention(projection_size, num_heads)
        self.bn1 = nn.BatchNorm1d(projection_size)

        self.bn2 = nn.BatchNorm1d(projection_size)

        # Feed forward neural network
        self.mlp = nn.Sequential(
            nn.Linear(projection_size, projection_size * 4),
            nn.ReLU(),
            nn.Linear(projection_size * 4, projection_size),
        )

    def forward(self, x, lx):
        # compute the key, query and value
        key = self._kw.forward(x)
        value = self._vw.forward(x)
        query = self._qw.forward(x)

        # compute the output of the attention module
        max_length = lx.max()
        key_padding_mask = ~(torch.arange(max_length)[None, :] < lx[:, None]).to(DEVICE)
        out1, _ = self.attention.forward(key=key, value=value, query=query, key_padding_mask=key_padding_mask)
        # Create a residual connection between the input and the output of the attention module
        out1 = out1 + x
        # Apply batch norm to out1
        out1 = self.bn1.forward(out1.transpose(1, 2)).transpose(1, 2)
        # Apply the output of the feed forward network
        out2 = self.mlp(out1)
        # Apply a residual connection between the input and output of the  FFN
        out2 = out2 + out1
        # Apply batch norm to the output
        out2 = self.bn2(out2.transpose(1, 2)).transpose(1, 2)

        return out2


class TransformerListener(torch.nn.Module):

    def __init__(self,
                 input_size,
                 base_lstm_layers=1,
                 seq_embedding_layers=1,
                 pblstm_layers=1,
                 listener_hidden_size=256,
                 n_heads=8,
                 tf_blocks=1):
        super().__init__()

        lstm_output_size = listener_hidden_size // (2 * (2 ** seq_embedding_layers))
        # create an lstm layer
        self.base_lstm = nn.LSTM(
            input_size,
            lstm_output_size,
            batch_first=True,
            bidirectional=True,
        )

        self._downsample_factor = (2 ** seq_embedding_layers)
        # create a sequence of Conv1d layers
        embed_layers = []
        embed_input_size = lstm_output_size * 2  # Count for bidirectional
        for i in range(seq_embedding_layers):
            embed_layers.append(DownSampleBlock(embed_input_size * (2 ** i)))
        self.embedding = nn.Sequential(
            *embed_layers
        )

        # compute the postion encoding
        self.positional_encoding = PositionalEncoding(listener_hidden_size, max_seq_len=2048)

        # create a sequence of transformer blocks
        self.transformer_encoder = torch.nn.ModuleList()
        for i in range(tf_blocks):
            self.transformer_encoder.append(
                TransformerEncoder(listener_hidden_size, num_heads=n_heads),
            )

    def forward(self, x, x_len):
        # pack the inputs before passing them to the LSTm
        x_packed = pack_padded_sequence(x, lengths=x_len, batch_first=True, enforce_sorted=False)
        # Pass the packed sequence through the lstm
        lstm_out, _ = self.base_lstm.forward(x_packed)
        # Unpack the output of the lstm
        output, output_lengths = pad_packed_sequence(
            lstm_out,
            batch_first=True,
        )

        # Pass the output through the embedding

        output = self.embedding.forward(output.transpose(1, 2))
        output = output.transpose(1, 2)   # (B, L, P)
        # calculate the new output length
        factor = 1
        while factor < self._downsample_factor:
            output_lengths = ((output_lengths - 1) // 2) + 1
            factor *= 2

        # calculate the position encoding
        output = self.positional_encoding.forward(output)
        # Pass the output of the positional encoding through the transformer encoder
        for m in self.transformer_encoder:
            output = m.forward(output, output_lengths)
        return output, output_lengths


class MultiHeadAttention(nn.Module):
    def __init__(self, projection_size, num_heads):
        super().__init__()
        self._kw = nn.Linear(projection_size, projection_size)
        self._vw = nn.Linear(projection_size, projection_size)
        self._qw = nn.Linear(projection_size, projection_size)
        self._num_heads = num_heads

    def forward(self, key, value, query, key_padding_mask):
        """
        keys: (B, L, P)
        values: (B, L, P)
        queries: (B, P)

        returns: (B, P)
        """
        batch_size, key_length, projection_size = key.shape
        _, query_length, _ = query.shape
        key_heads = self._kw.forward(
            key
        ).reshape(batch_size, key_length, self._num_heads, -1) / (projection_size ** 0.5)

        value_heads = self._vw.forward(
            value
        ).reshape(batch_size, query_length, self._num_heads, -1)  # (B, KL, H, P / H)

        query_heads = self._qw.forward(
            query
        ).reshape(batch_size, query_length, self._num_heads, -1)  # (B, QL, H, P / H)
        fill_value = torch.finfo(query_heads.dtype).min
        with torch.cuda.amp.autocast(enabled=False):    # Prevent from fp16 overflow.
            weights = torch.softmax(
                torch.masked_fill(
                    torch.einsum("bqhp, bkhp -> bhqk", query_heads.float(), key_heads.float()),
                    mask=key_padding_mask[:, None, None, :],
                    value=fill_value
                ),
                dim=-1
            )
        # if weights.isnan().any():
        #     print(torch.norm(query_heads, dim=-1))
        #     print(weights)

        result = torch.matmul(weights, value_heads.transpose(1, 2))\
            .transpose(1, 2)\
            .reshape(batch_size, key_length, -1)  # (B, KL,  H, P/H)
        return result, weights


class Attention(nn.Module):
    """
    Attention is calculated using the key, value (from encoder embeddings) and query from decoder.

    After obtaining the raw weights, compute and return attention weights and context as follows.:

    attention_weights   = softmax(raw_weights)
    attention_context   = einsum("thinkwhatwouldbetheequationhere",attention, value) #take hint from raw_weights calculation

    At the end, you can pass context through a linear layer too.
    """

    def __init__(
            self,
            listener_hidden_size,
            speller_hidden_size,
            projection_size,
    ):
        super().__init__()
        self._vw = nn.Linear(listener_hidden_size, projection_size)
        self._kw = nn.Linear(listener_hidden_size, projection_size)
        self._qw = nn.Linear(speller_hidden_size, projection_size)
        self._projection_size = projection_size
        self._key = None
        self._value = None
        self._key_mask = None

    def set_key_value(self, encoder_outputs, output_lengths):
        """
        In this function we take the encoder embeddings and make key and values from it.
        key.shape   = (batch_size, timesteps, projection_size)
        value.shape = (batch_size, timesteps, projection_size)
        """
        self._key = self._kw.forward(encoder_outputs)
        self._value = self._vw.forward(encoder_outputs)
        _, max_length, _ = encoder_outputs.shape
        self._key_mask = (output_lengths[:, None] < torch.arange(max_length)[None, :]).to(DEVICE)

    def compute_context(self, decoder_context):
        """
        In this function from decoder context, we make the query, and then we
         multiply the queries with the keys to find the attention logits,
         finally we take a softmax to calculate attention energy which gets
         multiplied to the generted values and then gets summed.

        key.shape   = (batch_size, timesteps, projection_size)
        value.shape = (batch_size, timesteps, projection_size)
        query.shape = (batch_size, projection_size)

        You are also recomended to check out Abu's Lecture 19 to understand Attention better.
        """
        # query = QW(decoder_context) #(batch_size, projection_size)
        query = self._qw.forward(decoder_context)[:, None, :].transpose(1, 2)
        raw_weights = torch.matmul(self._key, query) / torch.sqrt(torch.tensor(self._projection_size, device=DEVICE))  # (B, L, 1)
        # raw_weights = #using bmm or einsum. We need to perform batch matrix multiplication. It is important you do this step correctly.
        # #What will be the shape of raw_weights?

        # attention_weights = #What makes raw_weights -> attention_weights
        fill_value = torch.finfo(raw_weights.dtype).min
        with torch.cuda.amp.autocast(enabled=False):    # Prevent from fp16 overflow.
            attention_weights = torch.softmax(
                torch.masked_fill(
                    raw_weights,
                    mask=self._key_mask[..., None],
                    value=fill_value
                ),
                dim=1
            )

        attention_context = (attention_weights * self._value).sum(dim=1)

        return attention_context, attention_weights


class Speller(torch.nn.Module):

    # Refer to your HW4P1 implementation for help with setting up the language model.
    # The only thing you need to implement on top of your HW4P1 model is the attention module and teacher forcing.

    def __init__(self, attender: Attention, embedding_size, voc_size, n_lstm_layers):
        super().__init__()

        self.embedding_size = embedding_size
        self.attend = attender  # Attention object in speller
        self.max_timesteps = 600

        self.embedding = nn.Embedding(voc_size, embedding_size)

        self.lstm_projection = nn.Linear(embedding_size * 2, embedding_size)
        self.lstm_cells = nn.ModuleList()
        for i in range(n_lstm_layers):
            self.lstm_cells.append(nn.LSTMCell(embedding_size, embedding_size))

        # For CDN (Feel free to change)
        self.output_to_char = nn.Linear(2 * embedding_size,
                                        embedding_size)  # Linear module to convert outputs to correct hidden size (Optional: TO make dimensions match)
        self.activation = nn.ReLU()  # Check which activation is suggested
        self.char_prob = nn.Linear(embedding_size,
                                   voc_size)  # Linear layer to convert hidden space back to logits for token classification
        self.char_prob.weight = self.embedding.weight  # Weight tying (From embedding layer)

    def lstm_step(self, input_stats, hidden_state_list):
        if len(hidden_state_list) == 0:
            hidden_state_list = [[None] * len(self.lstm_cells)]
        hidden_state_t = []
        input_stats = self.lstm_projection.forward(input_stats)
        # Compute the outputs of a stacked lstm cells, each cell's hidden state is the input
        # to the next cell.
        for i in range(len(self.lstm_cells)):
            input_stats, hidden_state = self.lstm_cells[i].forward(input_stats, hidden_state_list[-1][i])
            hidden_state_t.append((input_stats, hidden_state))
        return hidden_state_t

    def cdn(self, inputs):
        # Make the CDN here, you can add the output-to-char
        inputs = self.output_to_char(inputs)
        inputs = self.activation(inputs)
        inputs = self.char_prob(inputs)
        return inputs

    def forward(self, batch_size, y=None, teacher_forcing_ratio=1, gumble=False, hard_gumble=False):

        raw_outputs = []
        attention_plot = []

        if y is None:
            timesteps = self.max_timesteps
            teacher_forcing_ratio = 0  # Why does it become zero?

        else:
            _, timesteps = y.shape  # How many timesteps are we predicting for?

        attn_context = torch.zeros(size=(batch_size, self.embedding_size)).to(
            DEVICE)  # initial context tensor for time t = 0
        char_embed = self.embedding.forward(
            torch.ones(size=(batch_size,), dtype=torch.long).to(DEVICE) * SOS_TOKEN
        )  # Initial inputs
        hidden_states_list = []  # Initialize your hidden_states list here similar to HW4P1
        for t in range(timesteps):
            p = random.random()  # generate a probability p between 0 and 1

            if p < teacher_forcing_ratio and t > 0:  # Why do we consider cases only when t > 0? What is considered when t == 0? Think.
                output_symbol = y[:, t - 1]  # Take from y, else draw from probability distribution
                char_embed = self.embedding.forward(output_symbol)  # Embed the character symbol

            # Concatenate the character embedding and context from attention, as shown in the diagram
            lstm_input = torch.concat([char_embed, attn_context], dim=1)

            hidden_states_t = self.lstm_step(lstm_input,
                                             hidden_states_list)  # Feed the input through LSTM Cells and attention.
            # What should we retrieve from forward_step to prepare for the next timestep?
            hidden_states_list.append(hidden_states_t)

            # Take the output of the final lstm cell.
            attn_context, attn_weights = self.attend.compute_context(
                hidden_states_t[-1][0])  # Feed the resulting hidden state into attention

            cdn_input = torch.concat([hidden_states_t[-1][0],
                                      attn_context], dim=1)

            raw_pred = self.cdn(cdn_input)  # call CDN with cdn_input

            # Generate next output-embedding with gumble-softmax trick
            if gumble:
                char_embed = torch.nn.functional.gumbel_softmax(raw_pred, tau=0.1, dim=1, hard=hard_gumble).matmul(self.embedding.weight)
            else:
                char_embed = self.embedding.forward(raw_pred.argmax(dim=1))

            raw_outputs.append(raw_pred)  # for loss calculation
            attention_plot.append(attn_weights)  # for plotting attention plot

        attention_plot = torch.stack(attention_plot, dim=1)
        raw_outputs = torch.stack(raw_outputs, dim=1)

        return raw_outputs, attention_plot


class ASRModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, voc_size, seq_embedding_layers):  # add parameters
        super().__init__()

        # Pass the right parameters here
        self.listener = TransformerListener(
            input_size=input_size,
            listener_hidden_size=hidden_size,
            seq_embedding_layers=seq_embedding_layers
        )
        self.attend = Attention(hidden_size, hidden_size, projection_size=hidden_size)
        self.speller = Speller(self.attend, embedding_size=hidden_size, voc_size=voc_size, n_lstm_layers=3)

    def forward(self, x, lx, y=None, tf_rate=1, gumble=False, hard_gumble=False):
        # Encode speech features
        encoder_outputs, output_lengths = self.listener(x, lx)

        # We want to compute keys and values ahead of the decoding step, as they are constant for all timesteps
        # Set keys and values using the encoder outputs
        self.attend.set_key_value(encoder_outputs, output_lengths)

        # Decode text with the speller using context from the attention
        raw_outputs, attention_plots = self.speller(batch_size=x.shape[0], y=y, teacher_forcing_ratio=tf_rate)

        return raw_outputs, attention_plots
