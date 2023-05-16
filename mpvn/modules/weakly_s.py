import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from mpvn.modules.attention import LocationAwareAttention
from mpvn.modules.modules import View, Transpose, Linear
import torch.nn.functional as F

class ARNNDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, encoder_size, attention_dim,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', 
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0):
        super(ARNNDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size * 2 if self.bidirectional_encoder else encoder_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        self.rnn = self.rnn_cell(self.hidden_size + self.encoder_output_size, self.hidden_size, self.n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional_decoder)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_p)
        
        self.attention = LocationAwareAttention(dec_dim=self.hidden_size, enc_dim=self.encoder_output_size, attn_dim=attention_dim)
        self.fc = nn.Linear(self.hidden_size + self.encoder_output_size, self.vocab_size)

    def forward(self, inputs: Tensor = None, encoder_outputs: Tensor = None, train_md: bool = False):
        """
        param:inputs: Decoder inputs sequence, Shape=(B, dec_T)
        param:encoder_outputs: Encoder outputs, Shape=(B,enc_T,enc_D)
        """
        batch_size = encoder_outputs.size(0)

        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2)) # (B, D)
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1)) # (B, T)
        hidden = None
        
        # if not train_md:
        decoder_input = inputs[inputs != self.eos_id].view(batch_size, -1)
        
        inputs = self.embedding(decoder_input) # (B, dec_T, voc_D) -> (B, dec_T, dec_D)
        inputs = self.input_dropout(inputs)

        y_all = []
        attn_w_all = []
        output_all = []
        for i in range(inputs.size(1)):
            embedded_inputs = inputs[:, i, :] # (B, dec_D)
            
            rnn_input = torch.cat([embedded_inputs, context], dim=1) # (B, dec_D + enc_D)
            rnn_input = rnn_input.unsqueeze(1) 
            output, hidden = self.rnn(rnn_input, hidden) # (B, 1, dec_D)

            context, attn_w = self.attention(output, encoder_outputs, attn_w) # (B, 1, enc_D), (B, enc_T)
            attn_w_all.append(attn_w)
            
            context = context.squeeze(1)
            output = output.squeeze(1) # (B, 1, dec_D) -> (B, dec_D)
            context = self.input_dropout(context)
            output = self.input_dropout(output)
            output = torch.cat((output, context), dim=1) # (B, dec_D + enc_D)
            output_all.append(torch.clone(output))

            pred = F.log_softmax(self.fc(output), dim=-1)
            y_all.append(pred)

        y_all = torch.stack(y_all, dim=1) # (B, dec_T, out_D)
        attn_w_all = torch.stack(attn_w_all, dim=1) # (B, dec_T, enc_T)
        output_all = torch.stack(output_all, dim=1) # (B, dec_T, dec_D + enc_D)
        
        return y_all, attn_w_all, output_all


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
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, inputs: Tensor):
        return self.sequential(inputs)


class RCNNMelEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 80,
        channels: int = 16,
        kernel: int = 5,
        padding: int = 2,
        stride: int = 1,
        dropout_cnn: float = 0.045,
        units: int = 128,
        dropout_gru: float = 0.045,
        joint_ctc_attention: bool = True
    ):
        super(RCNNMelEncoder, self).__init__()
        self.cnns = nn.Sequential(
            CNNEncoder(in_channels=1, out_channels=channels, kernel=kernel, padding=padding, stride=2, dropout=dropout_cnn),
            CNNEncoder(in_channels=channels, out_channels=channels, kernel=kernel, padding=padding, stride=2, dropout=dropout_cnn),
            CNNEncoder(in_channels=channels, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn)
        )
        self.input_projection = nn.Linear((input_dim >> 2)  * channels, units)
        self.gru = nn.GRU(
            input_size=units,
            hidden_size=units,
            bias=True,
            batch_first=True,
            dropout=dropout_gru,
            bidirectional=False,       
        )
        self.joint_ctc_attention = joint_ctc_attention
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(units),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_gru),
                Linear(units, num_classes, bias=False),
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        outputs = self.cnns(inputs.unsqueeze(1))
        batch_size, channels, seq_lengths, seq_dim = outputs.size()
        outputs = outputs.transpose(1, 2)
        outputs = outputs.contiguous().view(batch_size, seq_lengths, channels * seq_dim)
        outputs = self.input_projection(outputs)
        outputs, hidden = self.gru(outputs)
        if self.joint_ctc_attention:
            encoder_log_probs = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)
        output_lengths = input_lengths >> 2
        return encoder_log_probs, outputs, output_lengths

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
        embed_dim: int = 256,
        dropout_gru: float = 0.2,
        eos_id: float = 3
    ):
        super(RCNNPhonemeEncoder, self).__init__()
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, channels)
        self.cnns = nn.Sequential(
            CNNEncoder(in_channels=1, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(in_channels=channels, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn),
            CNNEncoder(in_channels=channels, out_channels=channels, kernel=kernel, padding=padding, stride=stride, dropout=dropout_cnn)
        )
        self.input_projection = nn.Linear(channels * channels, units)
        self.gru = nn.GRU(
            input_size=units,
            hidden_size=units,
            bias=True,
            batch_first=True,
            dropout=dropout_gru,
            bidirectional=False,       
        )
        self.output_projection = nn.Linear(units, embed_dim)

    def forward(self, inputs: Tensor):
        batch_size = inputs.size(0)
        inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
        outputs = self.embedding(inputs)
        outputs = self.cnns(outputs.unsqueeze(1))
        batch_size, channels, seq_lengths, seq_dim = outputs.size()
        outputs = outputs.transpose(1, 2)
        outputs = outputs.contiguous().view(batch_size, seq_lengths, channels * seq_dim)
        outputs = self.input_projection(outputs)
        outputs = self.gru(outputs)[0]
        outputs = self.output_projection(outputs)
        return outputs
    

class WordDecoderARNN(nn.Module):
    def __init__(self, output_size, hidden_size, encoder_size, attention_dim,
                 n_layers=1, rnn_cell='gru',
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0):
        super(WordDecoderARNN, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size * 2 if self.bidirectional_encoder else encoder_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        self.rnn = self.rnn_cell(self.encoder_output_size, self.hidden_size, self.n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional_decoder)

        self.input_dropout = nn.Dropout(self.dropout_p)
        
        self.attention = LocationAwareAttention(dec_dim=self.hidden_size, enc_dim=self.encoder_output_size, attn_dim=attention_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size + self.encoder_output_size, (self.hidden_size + self.encoder_output_size) // 2),
            nn.Tanh(),
            nn.Linear((self.hidden_size + self.encoder_output_size) // 2, self.output_size),
        )
        

    def forward(self, encoder_outputs: Tensor = None, max_length: int = None):
        """
        param:inputs: Decoder inputs sequence, Shape=(B, dec_T)
        param:encoder_hidden: Encoder last hidden states, Default : None
        param:encoder_outputs: Encoder outputs, Shape=(B,enc_T,enc_D)
        """
        batch_size = encoder_outputs.size(0)

        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2)) # (B, D)
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1)) # (B, T)

        hidden = None
        
        y_all = []
        attn_w_all = []
        
        for _ in range(max_length):
            rnn_input = context.unsqueeze(1) 
            output, hidden = self.rnn(rnn_input, hidden) # (B, 1, dec_D)

            context, attn_w = self.attention(output, encoder_outputs, attn_w) # (B, 1, enc_D), (B, enc_T)
            attn_w_all.append(attn_w)
            
            context = context.squeeze(1)
            output = output.squeeze(1) # (B, 1, dec_D) -> (B, dec_D)
            context = self.input_dropout(context)
            output = self.input_dropout(output)
            output = torch.cat((output, context), dim=1) # (B, dec_D + enc_D)

            pred = F.log_softmax(self.fc(output), dim=-1)
            y_all.append(pred)
            
        y_all = torch.stack(y_all, dim=1) # (B, dec_T, out_D)
        attn_w_all = torch.stack(attn_w_all, dim=1) # (B, dec_T, enc_T)

        return y_all, attn_w_all

