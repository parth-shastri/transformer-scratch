import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        # make the pe matrix to store the encoding shape (seq_len, d_model)
        pe = torch.zeros((seq_len, d_model))
        # create a vector with the seq_len indices shape (seq_len)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # create a vector of frequencies for each dimension of the model shape (d_model)
        freqs = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(1000) / d_model)  # (d_model / 2)
        # fill sin of the freq at each position of the seq_len at the even indices of d_model
        pe[:, 0::2] = torch.sin(positions * freqs)
        # fill sin of the freq at each position of the seq_len at the odd indices of d_model
        pe[:, 1::2] = torch.cos(positions * freqs)
        # add batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the pe as a buffer to save the state of the pe
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (B, seq_len, d_model)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter
    
    def forward(self, x):

        # x: (B, seq_len, d_model)
        # keep the last dimension for broadcasting
        mean = x.mean(dim=-1, keep_dim=True)  # (B, seq_len, 1)
        # Keep the last dimension for broadcasting
        std = x.std(dim=-1, keep_dim=True) # (B, seq_len, 1)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # x: (B, seq_len)
        # Multiply by sqrt(d_model) according to Attention is all you need
        return self.embedding(x) * math.sqrt(self.d_model)  # (B, seq_len, d_model)
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.seq_len = seq_len
        # make sure the d_model is divisible by h
        assert d_model % h == 0, 'd_model is not divisible by h'
        # initilalize the projection layers to generate the q, k & v

        self.register_buffer("masked_bias", torch.tril(torch.ones([self.seq_len, self.seq_len])).unsqueeze(0).unsqueeze(0))
        
        self.d_k = self.d_model // h # the per head feature dim
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, dropout: nn.Dropout):
        _, _, T, d_k = q.shape
        # formula from the paper attention is all you need
        # (B, h, seq_len, d_k) ---> (b, h, seq_len, seq_len)
        # attention_scores = (q @ k.transpose(-2, -1) / math.sqrt(d_k))
        attention_scores = einsum(q, k, "b h i d, b h j d -> b h i j",) / math.sqrt(d_k)
        if self.masked_bias is not None:
            # using the masked_fill_ method of the tensor
            # write very small values where mask is zero to denote -inf ==> -1e9
            attention_scores.masked_fill_(self.masked_bias[:, :, :T, :T] == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (B, h, seq_len, seq_len) ---> (B, h, seq_len, d_k)
        # return the attention scores so that they can be visualized
        return (attention_scores @ v), attention_scores

    def forward(self, q, k, v):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_q(v)

        # (B, seq_len, d_model) --> (B, seq_len, h, d_k) --> (B, h, seq_len, d_k)
        query = rearrange(query.reshape(query.shape[0], query.shape[1], self.h, self.d_k), 'b s h d -> b h s d')
        key = rearrange(key.reshape(key.shape[0], key.shape[1], self.h, self.d_k), 'b s h d -> b h s d')
        value = rearrange(value.reshape(value.shape[0], value.shape[1], self.h, self.d_k), 'b s h d -> b h s d')

        # calculate the attention
        x, self.attention_scores = self.attention(query, key, value, self.dropout)

        # combine all the heads
        #  (B, h, seq_len, d_k) --> (B, seq_len, h, d_k) --> (B, seq_len, d_model)
        x = rearrange(x, 'b h s d -> b s (h d)')

        # multiply by Wo
        # (B, seq_len, d_model) --> (B, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float, prenorm: bool=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)
        self.prenorm = prenorm # to use the normalization layer before the sublayer

    def forward(self, x, sublayer):
        if self.prenorm:
            return x + self.dropout(self.norm(sublayer(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x)))


class FeedForwardBlock(nn.Module):
    def __init__(self, d_ff: int, d_model: int, dropout: float):
        super().__init__()
        self.ff_1 = nn.Linear(d_model, d_ff) # the up projection layer
        self.dropout = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(d_ff, d_model) # the down projection layer

    def forward(self, x):
        # (B, seq_len, d_model) --> (B, seq_len, d_model)
        return self.ff_2(self.dropout(self.ff_1(x)))
    
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, h: int, d_ff:int, mha_dropout: float, ff_dropout: float, res_dropout: float, prenorm=True):
        super().__init__()
        # init the multihead attention and the feedforward block
        self.mha = MultiHeadAttentionBlock(d_model, seq_len, h, mha_dropout)
        # init the residual connections
        self.res_1 = ResidualConnection(d_model, res_dropout, prenorm=True)
        self.res_2 = ResidualConnection(d_model, res_dropout, prenorm=True)

        # init the feedforward connection
        self.ff = FeedForwardBlock(d_ff, d_model, ff_dropout)

    def forward(self, x):
        # (B, seq_len, d_model)
        x = self.res_1(x, lambda x: self.mha(x, x, x))
        x = self.res_2(x, self.ff)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int, seq_len: int, vocab_size: int,  d_model: int, h: int, d_ff:int, mha_dropout: float, ff_dropout: float, res_dropout: float, prenorm=True):
        super().__init__()
        # init the input embeddings layer
        self.emb = InputEmbeddings(d_model, vocab_size)
        # init the positional encoding layer
        self.pos_enc = PositionalEncoding(d_model, seq_len, res_dropout)
        # init the layers with the decoder block
        decoder_blocks = [DecoderBlock(d_model, seq_len, h, d_ff, mha_dropout, ff_dropout, res_dropout, prenorm) for _ in range(n_layers)]

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
    
    def forward(self, x):
        # (B, seq_len) --> (B, seq_len, d_model)
        x = self.emb(x)
        x = self.pos_enc(x)

        for sublayer in self.decoder_blocks:
            x = sublayer(x)
        
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (B, seq_len, d_model) --> (B, seq_len, vocab_size)
        return self.proj(x)