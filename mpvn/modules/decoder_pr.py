import torch
import torch.nn as nn
import random
from torch import Tensor
from typing import Optional, Any, Tuple

from mpvn.modules.attention import *
from mpvn.modules.embedding import Embedding, PositionalEncoding
from mpvn.modules.mask import get_attn_pad_mask, get_attn_subsequent_mask
from mpvn.modules.modules import Linear, AddNorm, PositionWiseFeedForwardNet, View, LayerNorm

class SpeechTransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            ffnet_style: str = 'ff'         # style of feed forward network
    ) -> None:
        super(SpeechTransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.encoder_decoder_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Optional[Any] = None,
            encoder_attn_mask: Optional[Any] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        output, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output, ecoder_decoder_attn = self.encoder_decoder_attention(output, encoder_outputs, encoder_outputs, encoder_attn_mask)
        output = self.feed_forward(output)
        return output, self_attn, ecoder_decoder_attn


class SpeechTransformerDecoder(nn.Module):
    r"""
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        num_classes: umber of classes
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of decoder layers
        num_heads: number of attention heads
        ffnet_style: style of feed forward network
        dropout_p: probability of dropout
        pad_id: identification of pad token
        eos_id: identification of end of sentence token
    """

    def __init__(
            self,
            num_classes: int,
            d_model: int = 512,
            d_ff: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            ffnet_style: str = 'ff',
            dropout_p: float = 0.3,
            pad_id: int = 0,
            eos_id: int = 3,
    ) -> None:
        super(SpeechTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            SpeechTransformerDecoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)
        ])
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, num_classes, bias=False),
        )

    def forward_step(
            self,
            decoder_inputs,
            decoder_input_lengths,
            encoder_outputs,
            encoder_output_lengths,
            positional_encoding_length,
    ) -> Tensor:
        dec_self_attn_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_input_lengths, decoder_inputs.size(1)
        )
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(
            encoder_outputs, encoder_output_lengths, decoder_inputs.size(1)
        )

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, encoder_decoder_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )

        return outputs, encoder_decoder_attn

    def forward(
            self,
            encoder_outputs: Tensor,
            targets: Optional[torch.LongTensor] = None,
            encoder_output_lengths: Tensor = None,
            target_lengths: Tensor = None
    ) -> Tensor:
        r"""
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size
                ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths (torch.LongTensor): The length of encoders outputs. ``(batch)``

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        batch_size = encoder_outputs.size(0)

        targets = targets[targets != self.eos_id].view(batch_size, -1)
        target_length = targets.size(1)

        outputs, encoder_decoder_attn = self.forward_step(
            decoder_inputs=targets,
            decoder_input_lengths=target_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=target_length,
        )
        
        return self.fc(outputs).log_softmax(dim=-1), encoder_decoder_attn


class DecoderRNN(nn.Module):
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
            hidden_state_dim: int = 1024,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3,
            use_tpu: bool = False,
    ) -> None:
        super(DecoderRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.use_tpu = use_tpu
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


class DecoderTransformer(nn.Module):
    def __init__(self,
        num_classes: int,
        hidden_state_dim: int = 512,
        num_heads: int = 4,
        pad_id: int = 0,
        eos_id: int = 3,
        dropout_p: float = 0.3,
        attention_dropout_p = 0.1

    ) -> None:
        super(DecoderTransformer, self).__init__()
        self.embedding = Embedding(num_classes, pad_id, hidden_state_dim)
        self.positional_encoding = PositionalEncoding(hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        
        self.self_attention = AddNorm(MultiHeadedSelfAttentionModule(hidden_state_dim, num_heads, attention_dropout_p), hidden_state_dim)
        self.attention = RelativeMultiHeadAttention(hidden_state_dim, num_heads=num_heads, dropout_p=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, hidden_state_dim), contiguous=True),
            nn.Linear(hidden_state_dim, num_classes),
            LayerNorm(hidden_state_dim)
        )
        
        self.eos_id = eos_id
        self.hidden_state_dim = hidden_state_dim

        
    def forward(
            self, 
            encoder_outputs: Tensor,
            targets: Optional[torch.LongTensor] = None,
    ):
        batch_size = targets.size(0)
        decoder_inputs = targets[targets != self.eos_id].view(batch_size, -1)
        
        output_lengths = decoder_inputs.size(1)
        positional_encoding_length = decoder_inputs.size(1)
      
        self_attn_mask = torch.gt(get_attn_subsequent_mask(decoder_inputs), 0)

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)
        
        outputs = self.self_attention(outputs, self_attn_mask)
        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)
        
        outputs = torch.cat((outputs, context), dim=2)
        
        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, attn
        

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
            hidden_state_dim: int = 1024,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.3,
            use_tpu: bool = False,
    ) -> None:
        super(DecoderARNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
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
        self.attention = MultiHeadLocationAwareAttention(hidden_state_dim, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
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

