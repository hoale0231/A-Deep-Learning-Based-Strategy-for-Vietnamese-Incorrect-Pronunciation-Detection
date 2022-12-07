import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import random
from mpvn.model.attention import MultiHeadAttention
from mpvn.model.modules import View

class RNNDecoder(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int = 1024,
            eos_id: int = 2,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3
    ) -> None:
        super(RNNDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim * 2, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
        )

    def forward(
            self,
            targets: Optional[Tensor] = None,
            encoder_outputs: Tensor = None
    ) -> Tensor:
        batch_size = targets.size(0)
        input_rnn = targets[targets != self.eos_id].view(batch_size, -1)
        output_lengths = input_rnn.size(1)

        if torch.cuda.is_available():
            input_rnn = input_rnn.cuda()

        embedded = self.embedding(input_rnn)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, _ = self.rnn(embedded)
        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)
        
        # Create phoneme-level mispronunciation features 
        # by concat cannonical phonemes and context vector,
        # but with shift and remove <sos>, <eos> items
        
        # mispronunciation_phone_features = torch.cat((embedded[1:], context[:-1]), dim=2)
        mispronunciation_phone_features = torch.cat((outputs, context), dim=2)
        outputs = torch.cat((outputs, context), dim=2)
        

        outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)             
        outputs = outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return outputs, attn, mispronunciation_phone_features

class WordDecoder(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int = 1024,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3,
    ) -> None:
        super(WordDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
            self,
            input_var: Tensor,
            hidden_states: Optional[Tensor],
            encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)


        if torch.cuda.is_available():
            input_var = input_var.cuda()

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)
        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn, context

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
        predicted_log_probs = list()
        

        targets, batch_size, max_length = self._validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        targets = targets[targets != self.eos_id].view(batch_size, -1)
        
        if use_teacher_forcing:
            step_outputs, hidden_states, attn, context = self.forward_step(targets, hidden_states, encoder_outputs)

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                predicted_log_probs.append(step_output)

        else:
            input_var = targets[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs)
                predicted_log_probs.append(step_outputs)
                input_var = predicted_log_probs[-1].topk(1)[1]

        predicted_log_probs = torch.stack(predicted_log_probs, dim=1)

        return predicted_log_probs, attn, context

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



