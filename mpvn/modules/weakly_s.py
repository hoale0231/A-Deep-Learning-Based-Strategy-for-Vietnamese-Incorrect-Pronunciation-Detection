import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from mpvn.modules.attention import LocationAwareAttention
from mpvn.modules.modules import View
import random


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        kernel: int = 5,
        padding: int = 2,
        stride: int = 1,
        dropout: float = 0.045
    ):
        super(CNNEncoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=dropout)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)

class GRUEncoder(nn.Module):
    def __init__(
        self,
        units: int = 128,
        dropout: float = 0.045
    ):
        super(GRUEncoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.GRU(input_size=units),
            nn.Dropout(p=dropout)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)

class RCNNMelEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        channels: int = 16,
        kernel: int = 5,
        padding: int = 2,
        stride: int = 1,
        dropout_cnn: float = 0.045,
        units: int = 128,
        dropout_gru: float = 0.045
    ):
        super(RCNNMelEncoder, self).__init__()
        self.sequential = nn.Sequential(
            CNNEncoder(in_channels=input_dim, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(in_channels=channels, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(in_channels=channels, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            GRUEncoder(units=units, dropout=dropout_gru)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)

class RCNNPhonemeEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int = 40,
        kernel: int = 5,
        padding: int = 2,
        stride: int = 1,
        dropout_cnn: float = 0.2,
        units: int = 128,
        dropout_gru: float = 0.2
    ):
        super(RCNNPhonemeEncoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Embedding(num_classes, channels),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            GRUEncoder(units=units, dropout=dropout_gru)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)
    
class DecoderARNN(nn.Module):
    """
    Converts higher level features (from encoder) into output utterances
    by specifying a probability distribution over sequences of characters.
    Args:
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the decoder hidden state `h`
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        num_heads (int, optional): number of attention heads. (default: 4)
        num_layers (int, optional): number of recurrent layers (default: 2)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability of decoder (default: 0.2)
    """

    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            max_length: int = 128,
            mel_dim: int = 128,
            phone_dim: int = 64,
            hidden_state_dim: int = 1024,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'gru',
            dropout_p: float = 0.3,
            use_tpu: bool = False,
    ) -> None:
        super(DecoderARNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length

        self.use_tpu = use_tpu
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=hidden_state_dim * 2,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.attention = LocationAwareAttention(mel_dim=mel_dim, phone_dim=phone_dim, hidden_dim=hidden_state_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
            self,
            input: Tensor,
            hidden_states: Optional[Tensor],
            encoder_outputs: Tensor,
            last_attn: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, output_lengths = input.size(0), input.size(1)


        if torch.cuda.is_available():
            input = input.cuda()

        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        context, attn = self.attention(embedded, encoder_outputs, last_attn)
        input = torch.cat((embedded, context), dim=2)
        
        outputs, hidden_states = self.rnn(input, hidden_states)
        outputs = torch.cat((outputs, context), dim=2)
        
        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
            self,
            targets: Optional[Tensor] = None,
            encoder_outputs: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.
        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            teacher_forcing_ratio (float): ratio of teacher forcing
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        hidden_states, attn = None, None
        predicted_log_probs, attns = list(), list()

        targets, batch_size, max_length = self._validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        targets = targets[targets != self.eos_id].view(batch_size, -1)
        
        if use_teacher_forcing:
            for di in range(max_length):
                input_rnn = targets[:, di].unsqueeze(1)
                step_outputs, hidden_states, attn = self.forward_step(input_rnn, hidden_states, encoder_outputs, attn)
                predicted_log_probs.append(step_outputs)
                attns.append(attn)
        else:
            input_rnn = targets[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(input_rnn, hidden_states, encoder_outputs, attn)
                predicted_log_probs.append(step_outputs)
                input_rnn = predicted_log_probs[-1].topk(1)[1]

        predicted_log_probs = torch.stack(predicted_log_probs, dim=1)
        attns = torch.stack(attns, dim=1).permute(0,2,1,3).squeeze()
        return predicted_log_probs, attns

    def _validate_args(
            self,
            targets: Optional[Tensor] = None,
            encoder_outputs: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[Tensor, int, int]:
        """ Validate arguments """
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:  # inference
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                targets = targets.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1  # minus the start of sequence symbol

        return targets, batch_size, max_length

class WordDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            num_words: int,
            hidden_state_dim: int = 1024,
            num_heads: int = 4,
            dropout_p: float = 0.3
    ) -> None:
        super(WordDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.embedding = nn.Embedding(num_words, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        self.ff = nn.Linear(hidden_state_dim * 2, hidden_state_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim * 2, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
        )
    def forward(
            self,
            words: Optional[Tensor] = None,
            encoder_outputs: Tensor = None
    ) -> Tensor:
        batch_size, output_lengths = words.size(0), words.size(1)
        embedded = self.embedding(words)
        embedded = self.input_dropout(embedded)
        encoder_outputs = self.ff(encoder_outputs)
        context, attn = self.attention(embedded, encoder_outputs, encoder_outputs)

        outputs = torch.cat((embedded, context), dim=2)

        outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)             
        outputs = outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return outputs, attn
    