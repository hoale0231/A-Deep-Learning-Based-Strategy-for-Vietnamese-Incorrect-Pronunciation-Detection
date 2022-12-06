import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

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
        outputs = torch.cat((outputs, context), dim=2)

        outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)             
        outputs = outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return outputs, attn



