import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import random
from mpvn.model.attention import MultiHeadAttention, MultiHeadedSelfAttentionModule
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
            space_id: int = 1,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3
    ) -> None:
        super(RNNDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.eos_id = eos_id
        self.space_id = space_id
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
        
    def _split_output_to_word(self, input: Tensor, output: Tensor):
        word_list = []
        max_len = 0
        for b in range(len(input)):
            word = []
            for i, o in zip(input[b], output[b]):
                if i == self.space_id or i == self.eos_id:
                    word_list.append(torch.stack(word))
                    max_len = max(len(word), max_len)
                    word = []
                else:
                    word.append(o)
                if i == self.eos_id:
                    break
            if word:
                max_len = max(len(word), max_len)
                word_list.append(torch.stack(word))
        words = torch.zeros(len(word_list), max_len, output.shape[-1])
        if torch.cuda.is_available():
            words = words.cuda()
        for word_tensor, word in zip(words, word_list):
            word_tensor[:len(word)] = word
        return words
        

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
        mispronunciation_phone_features = torch.cat((embedded[:,1:], context[:,:-1]), dim=2)
        mispronunciation_phone_features = self._split_output_to_word(input_rnn[:,1:], mispronunciation_phone_features)
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
            num_layers: int = 1,
            dropout_p: float = 0.3,
            rnn_type: str = 'gru'
    ) -> None:
        super(WordDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.input_dropout = nn.Dropout(dropout_p)
        self.self_attention = MultiHeadedSelfAttentionModule(hidden_state_dim, num_heads=num_heads)
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.ff = nn.Linear(hidden_state_dim * 2, hidden_state_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_state_dim // 2, num_classes),
        )

    def forward(
            self,
            inputs: Tensor = None
    ) -> Tensor:
        inputs = self.ff(inputs)

        output = self.self_attention(inputs)
        output, _ = self.rnn(output)
        output = self.fc(output[:,-1,:]).log_softmax(dim=-1)
        return output



