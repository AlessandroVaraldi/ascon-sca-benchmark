#!/usr/bin/env python3
"""
TinyTransformer: small Transformer with convolutional stem for SCA traces.

Designed to work with the Ascon pipeline:
- Input: traces [B, T] or [B, 1, T]
- Output: logits [B, num_classes]
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class ConvStem(nn.Module):
    def __init__(self, in_ch: int, d_model: int, k: int = 5):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        return self.act(self.bn(self.conv(x)))


class PositionalMixing(nn.Module):
    """Depthwise 1D conv as positional mixing; cheap and locality-aware."""

    def __init__(self, d_model: int, k: int = 3, dilation: int = 1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=k,
            padding=pad,
            dilation=dilation,
            groups=d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,d]
        z = x.transpose(1, 2)        # [B,d,T]
        z = self.dw(z)               # [B,d,T]
        return x + z.transpose(1, 2) # residual


class MultiheadSelfAttention(nn.Module):
    """Explicit MHA with pre-LN; global attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        B, T, D = x.shape
        h, dh = self.nhead, self.head_dim

        x_n = self.ln(x)
        qkv = self.qkv(x_n)                      # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)           # each [B,T,D]

        q = q.view(B, T, h, dh).transpose(1, 2)  # [B,h,T,dh]
        k = k.view(B, T, h, dh).transpose(1, 2)
        v = v.view(B, T, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,h,T,T]
        # numerically stable softmax
        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn = torch.exp(scores)
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn = attn / attn_sum
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, v)              # [B,h,T,dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)

        out = self.proj(ctx)
        out = self.proj_drop(out)
        return out


class TransformerMLP(nn.Module):
    """FFN with pre-LN: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int, hidden_mult: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln(x)
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop1(z)
        z = self.fc2(z)
        z = self.drop2(z)
        return z


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: MHA + residual, then MLP + residual."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_mult: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = MultiheadSelfAttention(
            d_model,
            nhead,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )
        self.mlp = TransformerMLP(d_model, hidden_mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(x)
        x = x + self.mlp(x)
        return x


class LoRAAdapter(nn.Module):
    """LoRA-style residual adapter: y = x + (alpha/r)*(B(A(x)))."""

    def __init__(self, d_model: int, r: int = 0, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        if r <= 0:
            self.A = None
            self.B = None
            self.drop = nn.Identity()
        else:
            self.A = nn.Linear(d_model, r, bias=False)
            self.B = nn.Linear(r, d_model, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0:
            return x
        return x + (self.alpha / self.r) * self.B(self.drop(self.A(x)))


class TinyTransformer(nn.Module):
    """
    Tiny Transformer for SCA (global attention + attention pooling).

    Input:  x [B, T] or [B, 1, T]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        in_ch: int = 1,
        d_model: int = 96,
        nhead: int = 4,
        depth: int = 3,
        ffn_mult: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        lora_r: int = 0,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = ConvStem(in_ch, d_model, k=5)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.adapters = nn.ModuleList(
            [
                LoRAAdapter(d_model, lora_r, lora_alpha, lora_dropout)
                for _ in range(depth)
            ]
        )

        self.pos = PositionalMixing(d_model, k=3, dilation=1)
        self.final_norm = nn.LayerNorm(d_model)

        # Attention pooling over time
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T] or [B,1,T]
        if x.dim() == 2:
            x = x.unsqueeze(1)   # [B,1,T]

        z = self.stem(x)        # [B,d,T]
        z = z.transpose(1, 2)   # [B,T,d]
        z = self.pos(z)         # [B,T,d]

        for i, blk in enumerate(self.blocks):
            z = blk(z)
            z = self.adapters[i](z)

        z = self.final_norm(z)  # [B,T,d]

        # Attention pooling along T
        w = self.attn_pool(z).squeeze(-1)  # [B,T]
        w = torch.softmax(w, dim=1)
        feat = (z * w.unsqueeze(-1)).sum(dim=1)  # [B,d]

        logits = self.classifier(feat)    # [B,num_classes]
        return logits
