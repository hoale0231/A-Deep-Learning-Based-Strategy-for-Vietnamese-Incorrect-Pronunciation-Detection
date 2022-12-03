from torch import nn
from mpvn.model.modules import LayerNorm, ResidualConnectionModule
from mpvn.model.feed_forward import FeedForwardModule
from mpvn.model.attention import MultiHeadAttention, MultiHeadedSelfAttentionModule

class TransformerDecoderBlock(nn.Module):

    def __init__(self, d_model, feed_forward_expansion_factor, n_head, drop_prob):
        super(TransformerDecoderBlock, self).__init__()
            
        self.self_attention = MultiHeadedSelfAttentionModule(d_model=d_model, num_heads=n_head, dropout_p=drop_prob)
        self.norm1 = LayerNorm(dim=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(dim=d_model, num_heads=n_head)
        self.norm2 = LayerNorm(dim=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = FeedForwardModule(d_model=d_model, expansion_factor=feed_forward_expansion_factor, dropout_p=drop_prob)
        self.norm3 = LayerNorm(dim=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, decoder_input, encoder_output, t_mask, s_mask):
        # 1. compute self attention
        _x = decoder_input
        x = self.self_attention(inputs=_x, mask=t_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if encoder_output is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x, attn = self.enc_dec_attention(query=x, key=encoder_output, value=encoder_output, mask=s_mask)
            
            # 4. add and norm
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.norm3(x + _x)
        x = self.dropout3(x)
        return x, attn

class TransformerDecoder(nn.Module):
    def __init__(self, num_classes, d_model, feed_forward_expansion_factor, n_head, n_layers, drop_prob):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_model)

        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model=d_model,
                                                  feed_forward_expansion_factor=feed_forward_expansion_factor,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, target, encode_output, trg_mask, src_mask):
        target = self.emb(target)

        for layer in self.layers:
            target, attn = layer(target, encode_output, trg_mask, src_mask)

        output = self.linear(target)
        print(output)
        output = output.log_softmax(dim=-1)
        print(output)
        return output, attn

