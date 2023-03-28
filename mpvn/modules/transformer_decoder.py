import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
from torch import Tensor
from typing import Optional, Any, Tuple

from mpvn.modules.attention import *
from mpvn.modules.embedding import Embedding, PositionalEncoding
from mpvn.modules.mask import get_attn_subsequent_mask, get_attn_pad_mask
from mpvn.modules.modules import Linear, AddNorm, ResidualConnectionModule, View, LayerNorm
from mpvn.modules.feed_forward import FeedForwardModule

class DecoderTransformer(nn.Module):
    def __init__(self,
        num_classes: int,
        hidden_state_dim: int = 512,
        num_heads: int = 4,
        pad_id: int = 0,
        space_id: int = 1,
        eos_id: int = 3,
        sos_id: int = 2,
        dropout_p: float = 0.3,
        attention_dropout_p = 0.1,
        feed_forward_expansion_factor: int = 4,
        feed_forward_dropout_p: float = 0.1,
        half_step_residual: bool = True
    ) -> None:
        super(DecoderTransformer, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1
        
        self.embedding = Embedding(num_classes, pad_id, hidden_state_dim)
        self.positional_encoding = PositionalEncoding(hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.pre_feed_forward = ResidualConnectionModule(
                module=FeedForwardModule(
                    d_model=hidden_state_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )
        self.self_attention = AddNorm(
            MultiHeadedSelfAttentionMaskedQueryModule(
                hidden_state_dim, 
                num_heads, 
                attention_dropout_p
                ), d_model=hidden_state_dim
            )
        self.attention = AddNorm(
            MultiHeadedAttentionModule(
                hidden_state_dim, 
                num_heads, 
                attention_dropout_p
                ), d_model=hidden_state_dim
            )
        self.post_feed_forward = ResidualConnectionModule(
                module=FeedForwardModule(
                    d_model=hidden_state_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
        )
        self.space_id = space_id
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.hidden_state_dim = hidden_state_dim

    def _split_output_to_word(self, input: Tensor, output: Tensor):
        word_list = []
        max_len = 0

        for b in range(len(input)):
            word = []
            for i, o in zip(input[b], output[b]):
                if i == self.space_id or i == self.pad_id:
                    word_list.append(torch.stack(word))
                    max_len = max(len(word), max_len)
                    word = []
                else:
                    word.append(o)
                if i == self.pad_id:
                    break
            if word:
                max_len = max(len(word), max_len)
                word_list.append(torch.stack(word))

        return pad_sequence(word_list, batch_first=True)
    
    def get_mask(
        self, 
        decoder_inputs: Tensor, 
        decoder_inputs_lengths, 
        encoder_outputs: Tensor, 
        encoder_output_lengths: Tensor
    ):
        dec_self_attn_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_inputs_lengths, decoder_inputs.size(1)
        )
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(
            encoder_outputs, encoder_output_lengths, decoder_inputs.size(1)
        )
        return self_attn_mask, encoder_attn_mask
        
    
    def forward(
            self, 
            decoder_inputs: Optional[torch.LongTensor],
            encoder_outputs: Tensor,
            targets_lengths: Tensor,
            encoder_output_lengths: Tensor,
            get_mispronunciation_phone_features: bool = True
    ):
        batch_size = decoder_inputs.size(0)

        output_lengths = decoder_inputs.size(1)
        
        self_attn_mask, encoder_attn_mask = self.get_mask(decoder_inputs, targets_lengths, encoder_outputs, encoder_output_lengths)
        
        outputs = self.embedding(decoder_inputs) + self.positional_encoding(output_lengths)
        outputs = self.input_dropout(outputs)
        outputs = self.pre_feed_forward(outputs)
        
        outputs_res, outputs = self.self_attention(outputs, self_attn_mask)
        context, _, attn = self.attention(outputs, encoder_outputs, encoder_outputs, encoder_attn_mask)
        outputs = self.post_feed_forward(context)

        if get_mispronunciation_phone_features:
            mispronunciation_phone_features = torch.cat((outputs_res, context), dim=2)
            mispronunciation_phone_features = self._split_output_to_word(decoder_inputs, mispronunciation_phone_features)
        else:
            mispronunciation_phone_features = None
            
        outputs = self.fc(outputs.view(-1, self.hidden_state_dim)).log_softmax(dim=-1)
        outputs = outputs.view(batch_size, output_lengths, -1).squeeze(1)
        return outputs, attn, mispronunciation_phone_features