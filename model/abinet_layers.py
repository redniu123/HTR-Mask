"""
ABINet Language Model Components
Ported from: https://github.com/FangShancheng/ABINet
Adapted for HTR-VT integration

核心组件:
- PositionalEncoding: 位置编码
- TransformerDecoderLayer: Transformer 解码层
- PositionAttention: 位置注意力 (Visual -> Fixed Length)
- BCNLanguage: 双向完形填空网络 (语言模型)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# Helper Functions
# ============================================================

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    Args:
        d_model: embedding dimension
        dropout: dropout rate
        max_len: maximum sequence length
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, N, E) tensor
        Returns:
            (T, N, E) tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ============================================================
# Transformer Decoder Components
# ============================================================

class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention (simplified version)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
        """
        Args:
            query: (T, N, E)
            key: (S, N, E)
            value: (S, N, E)
        Returns:
            output: (T, N, E)
            attn_weights: (N, T, S) if need_weights else None
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Project Q, K, V
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling
        
        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # (bsz * num_heads, tgt_len, src_len)
        
        # Apply masks
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights + attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)  # (bsz * num_heads, tgt_len, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_weights.sum(dim=1) / self.num_heads
        return attn_output, None


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer for Language Model (no self-attention by default)
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", 
                 self_attn: bool = False):
        super().__init__()
        self.has_self_attn = self_attn
        
        if self.has_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (T, N, E)
            memory: (S, N, E)
        Returns:
            (T, N, E)
        """
        if self.has_self_attn:
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                     key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Module):
    """Stack of Transformer Decoder Layers"""
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        return output


# ============================================================
# Position Attention (Visual Feature -> Fixed Length Queries)
# ============================================================

def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode == 'nearest' else True
    return nn.Sequential(
        nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class PositionAttention(nn.Module):
    """
    Position Attention Module
    将变长的视觉特征转换为固定长度的字符查询
    
    ABINet 原版输入是 (N, E, H, W) 的特征图
    HTR-VT 输入是 (N, L, E) 的序列，需要适配
    """
    def __init__(self, max_length: int, in_channels: int = 512, 
                 num_channels: int = 64, mode: str = 'nearest'):
        super().__init__()
        self.max_length = max_length
        self.in_channels = in_channels
        
        # 对于 HTR-VT: 输入是 (B, 128, 768)，需要 reshape 成 (B, 768, 1, 128)
        # 然后通过 encoder-decoder 处理
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),  # -> (N, 64, 1, 64)
            encoder_layer(num_channels, num_channels, s=(1, 2)),  # -> (N, 64, 1, 32)
            encoder_layer(num_channels, num_channels, s=(1, 2)),  # -> (N, 64, 1, 16)
            encoder_layer(num_channels, num_channels, s=(1, 2))   # -> (N, 64, 1, 8)
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=(1, 2), mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=(1, 2), mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=(1, 2), mode=mode),
            decoder_layer(num_channels, in_channels, size=(1, 128), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (N, L, E) - ViT encoder output, L=128, E=768
            
        Returns:
            attn_vecs: (N, T, E) - T=max_length 固定长度输出
            attn_scores: (N, T, L) - 注意力权重
        """
        N, L, E = x.shape  # (B, 128, 768)
        
        # Reshape to (N, E, 1, L) for conv processing
        x_2d = x.permute(0, 2, 1).unsqueeze(2)  # (N, 768, 1, 128)
        
        k, v = x_2d, x_2d
        
        # Calculate key vector through encoder-decoder
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)  # (N, E, 1, L)
        
        # Calculate query vector using positional encoding
        zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        
        # Calculate attention: q @ k^T
        k_flat = k.flatten(2, 3)  # (N, E, L)
        attn_scores = torch.bmm(q, k_flat)  # (N, T, L)
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        v_flat = v.flatten(2, 3).permute(0, 2, 1)  # (N, L, E)
        attn_vecs = torch.bmm(attn_scores, v_flat)  # (N, T, E)
        
        return attn_vecs, attn_scores


# ============================================================
# BCN Language Model (Bidirectional Cloze Network)
# ============================================================

class BCNLanguage(nn.Module):
    """
    Bidirectional Cloze Network - 语言模型
    
    接收视觉分支的 softmax 输出，通过 Transformer 进行语言建模
    
    Args:
        num_classes: 字符类别数 (包含 blank/padding)
        max_length: 最大序列长度
        d_model: Transformer 维度
        nhead: 注意力头数
        d_inner: FFN 内部维度
        dropout: dropout 率
        num_layers: Transformer 层数
        detach: 是否 detach 输入梯度
    """
    def __init__(self, num_classes: int, max_length: int,
                 d_model: int = 512, nhead: int = 8, d_inner: int = 2048,
                 dropout: float = 0.1, num_layers: int = 4,
                 detach: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.num_classes = num_classes
        self.detach = detach
        
        # Project from class probabilities to embedding space
        self.proj = nn.Linear(num_classes, d_model, bias=False)
        
        # Positional encodings
        self.token_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=max_length)
        
        # Transformer decoder (no self-attention for BCN)
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, d_inner, dropout, 
            activation='relu', self_attn=False
        )
        self.model = TransformerDecoder(decoder_layer, num_layers)
        
        # Output classifier
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            tokens: (N, T, C) - softmax probabilities from visual branch
            lengths: (N,) - 序列长度 (可选，用于 padding mask)
            
        Returns:
            dict with:
                - 'feature': (N, T, E) - 语言特征
                - 'logits': (N, T, C) - 分类 logits
        """
        if self.detach:
            tokens = tokens.detach()
        
        N, T, C = tokens.shape
        
        # Project to embedding space
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        
        # Create padding mask if lengths provided
        padding_mask = None
        if lengths is not None:
            padding_mask = self._get_padding_mask(lengths, T)
        
        # Create location mask (diagonal mask for cloze)
        location_mask = self._get_location_mask(T, tokens.device)
        
        # Query using positional encoding
        zeros = embed.new_zeros(*embed.shape)
        query = self.pos_encoder(zeros)
        
        # Transformer forward
        output = self.model(
            query, embed,
            tgt_key_padding_mask=padding_mask,
            memory_mask=location_mask,
            memory_key_padding_mask=padding_mask
        )  # (T, N, E)
        
        output = output.permute(1, 0, 2)  # (N, T, E)
        logits = self.cls(output)  # (N, T, C)
        
        return {'feature': output, 'logits': logits}
    
    @staticmethod
    def _get_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        """Generate padding mask"""
        lengths = lengths.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=lengths.device).unsqueeze(0)
        return grid >= lengths  # True = masked
    
    @staticmethod
    def _get_location_mask(sz: int, device) -> torch.Tensor:
        """Generate location mask for cloze (mask diagonal)"""
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask


# ============================================================
# Fusion Module
# ============================================================

class GatedFusion(nn.Module):
    """
    Gated Fusion for combining visual and language features
    
    Args:
        d_model: feature dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, v_feat: torch.Tensor, l_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_feat: (N, T, E) visual features
            l_feat: (N, T, E) language features
        Returns:
            (N, T, E) fused features
        """
        concat = torch.cat([v_feat, l_feat], dim=-1)  # (N, T, 2E)
        gate = self.gate(concat)  # (N, T, E)
        fused = gate * v_feat + (1 - gate) * l_feat
        return fused

