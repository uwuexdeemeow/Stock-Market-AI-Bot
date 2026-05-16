"""
Shared neural model definitions for training and inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, H = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        w = self.dropout(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1))
        o = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(o).mean(dim=1)


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_heads, n_ret_bins):
        super().__init__()
        self.model_type = "lstm_attention"
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.trunk = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, 64), nn.ReLU(),
        )
        self.dir_head = nn.Linear(64, 2)
        self.ret_head = nn.Linear(64, n_ret_bins)

    def forward(self, x):
        out, _ = self.lstm(x)
        shared = self.trunk(self.layer_norm(self.attention(out)))
        return self.dir_head(shared), self.ret_head(shared)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_layers, dim_ff, dropout, n_ret_bins, max_len):
        super().__init__()
        self.model_type = "transformer"
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_dropout = nn.Dropout(dropout * 0.5)
        pe = torch.zeros(max_len, hidden_size)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))
        self.recency_weight = nn.Parameter(torch.linspace(0.5, 1.5, max_len))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.trunk = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, 64), nn.ReLU(),
        )
        self.dir_head = nn.Linear(64, 2)
        self.ret_head = nn.Linear(64, n_ret_bins)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_dropout(x + self.pos_encoding[:, :x.size(1), :])
        x = x * self.recency_weight[:x.size(1)].unsqueeze(0).unsqueeze(-1)
        shared = self.trunk(self.layer_norm(self.transformer(x).mean(dim=1)))
        return self.dir_head(shared), self.ret_head(shared)


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, cnn_filters, kernel, n_ret_bins):
        super().__init__()
        self.model_type = "cnn_lstm"
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel, padding=kernel // 2),
            nn.BatchNorm1d(cnn_filters), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel, padding=kernel // 2),
            nn.BatchNorm1d(cnn_filters * 2), nn.GELU(),
        )
        self.lstm = nn.LSTM(
            cnn_filters * 2, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, 64), nn.GELU()
        )
        self.dir_head = nn.Linear(64, 2)
        self.ret_head = nn.Linear(64, n_ret_bins)

    def forward(self, x):
        x = self.conv2(self.conv1(x.permute(0, 2, 1))).permute(0, 2, 1)
        out, _ = self.lstm(x)
        shared = self.trunk(out[:, -1, :])
        return self.dir_head(shared), self.ret_head(shared)


class PlainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, n_ret_bins):
        super().__init__()
        self.model_type = "lstm_plain"
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, 64), nn.ReLU(),
        )
        self.dir_head = nn.Linear(64, 2)
        self.ret_head = nn.Linear(64, n_ret_bins)

    def forward(self, x):
        out, _ = self.lstm(x)
        shared = self.trunk(out[:, -1, :])
        return self.dir_head(shared), self.ret_head(shared)


def build_model(model_type: str, n_features: int, saved: dict):
    h = saved["hidden_size"]
    l = saved["num_layers"]
    d = saved["dropout"]
    ah = saved["attn_heads"]
    cf = saved["cnn_filters"]
    ck = saved["cnn_kernel"]
    th = saved["transformer_heads"]
    tdff = saved["transformer_dff"]
    tl = saved["transformer_layers"]
    nrb = saved["n_return_bins"]
    ml = saved["transformer_max_len"]

    builders = {
        "lstm_attention": lambda: LSTMWithAttention(n_features, h, l, d, ah, nrb),
        "transformer": lambda: TransformerEncoder(n_features, h, th, tl, tdff, d, nrb, ml),
        "cnn_lstm": lambda: CNNLSTM(n_features, h, l, d, cf, ck, nrb),
        "lstm_plain": lambda: PlainLSTM(n_features, h, l, d, nrb),
    }
    return builders[model_type]()


def build_neural_model(
    model_type: str,
    input_size: int,
    *,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    attn_heads: int,
    cnn_filters: int,
    cnn_kernel: int,
    transformer_heads: int,
    transformer_dff: int,
    transformer_layers: int,
    n_return_bins: int,
    window_size: int,
) -> nn.Module:
    factories = {
        "lstm_attention": lambda: LSTMWithAttention(input_size, hidden_size, num_layers, dropout, attn_heads, n_return_bins),
        "transformer": lambda: TransformerEncoder(input_size, hidden_size, transformer_heads, transformer_layers, transformer_dff, dropout, n_return_bins, window_size),
        "cnn_lstm": lambda: CNNLSTM(input_size, hidden_size, num_layers, dropout, cnn_filters, cnn_kernel, n_return_bins),
        "lstm_plain": lambda: PlainLSTM(input_size, hidden_size, num_layers, dropout, n_return_bins),
    }
    if model_type not in factories:
        raise ValueError(f"Unknown model: {model_type}")
    return factories[model_type]()
