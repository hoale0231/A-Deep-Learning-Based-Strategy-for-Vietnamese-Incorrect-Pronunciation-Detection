import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple

# Encoder

class CNNEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 16,
        kernel: int = 5,
        padding: int = 2,
        stride: int = 1,
        dropout: float = 0.045
    ):
        super(CNNEncoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(),
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
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            GRUEncoder(units=units, dropout=dropout_gru)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)
    
class RCNNPhonemeEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 15,
        channels: int = 40,
        kernel: int = 5,
        padding: int = 2,
        stride: int = 1,
        dropout_cnn: float = 0.2,
        units: int = 128,
        dropout_gru: float = 0.2
    ):
        super(RCNNPhonemeEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Embedding(embedding_dim=dim),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            GRUEncoder(units=units, dropout=dropout_gru)
        )
        
    def forward(self, inputs: Tensor):
        return self.sequential(inputs)

# Decoder

class LocationSensitiveAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.
    Args:
        hidden_dim (int): dimesion of hidden state vector
        smoothing (bool): flag indication whether to use smoothing or not.
    Inputs: query, value, last_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        super(LocationSensitiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD

        return context, attn
    
class GRUDecoder(nn.Module):
    def __init__(
        self,
        units: int = 64,
        dropout: float = 0.2
    ):
        super(GRUDecoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.GRU(input_size=units),
            nn.Dropout(p=dropout)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)
    
class ARNNPhonemeDecoder(nn.Module):
    def __init__(
        self,
        units_lsa: int = 10,
        units_gru: int = 64,
        dropout: float = 0.2
    ):
        super(ARNNPhonemeDecoder, self).__init__()
        self.sequential = nn.Sequential(
            LocationSensitiveAttention(hidden_dim=units_lsa),
            GRUDecoder(units=units_gru, dropout=dropout)
        )
  
    def forward(self, inputs: Tensor):
        return self.sequential(inputs)
    
class ARNNWordDecoder(nn.Module):
    def __init__(
        self,
        units_lsa: int = 10,
        units_gru: int = 64,
        dropout: float = 0.2
    ):
        super(ARNNWordDecoder, self).__init__()
        self.sequential = nn.Sequential(
            LocationSensitiveAttention(hidden_dim=units_lsa),
            GRUDecoder(units=units_gru, dropout=dropout)
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)