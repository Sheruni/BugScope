import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CustomCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomCrossAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # Apply positional encoding
        query = self.positional_encoding(query)
        key = self.positional_encoding(key)
        value = self.positional_encoding(value)

        # Apply layer normalization
        query = self.layer_norm1(query)
        key = self.layer_norm1(key)
        value = self.layer_norm1(value)

        # Perform multi-head attention
        attn_output, attn_weights = self.multihead_attention(query, key, value)

        # Apply layer normalization to the output
        attn_output = self.layer_norm2(attn_output)

        return attn_output, attn_weights

