import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from model.abinet_layers import PositionAttention, BCNLanguage, GatedFusion
from functools import partial


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        self.bias = torch.ones(1, 1, self.num_patches, self.num_patches)
        self.back_bias = torch.triu(self.bias)
        self.forward_bias = torch.tril(self.bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            num_patches,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.0,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)


class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    
    扩展版本: 支持 CTC + ABINet 语言模型双分支
    """

    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 32],
                 patch_size=[8, 32],
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 max_length=26,
                 use_language_model=True):
        super().__init__()

        self.nb_cls = nb_cls
        self.max_length = max_length
        self.use_language_model = use_language_model
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        
        # 根据论文规范：ResNet输出为 (B, C, 1, L)，其中 L=128
        # grid_size 应该反映实际的特征图形状，而不是 img_size // patch_size
        # 对于 H=64, W=512 的输入，ResNet 输出 H'=1, W'=128
        # 所以 grid_size = [1, 128] 用于正确的 2D Sin-Cos 位置编码
        self.grid_size = [1, img_size[1] // patch_size[0]]  # [1, 128] for H=64, W=512, patch_w=4
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # = 128
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        
        # --------------------------------------------------------------------------
        # Branch 1: CTC Head
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        
        # --------------------------------------------------------------------------
        # Branch 2: ABINet Language Branch (Optional)
        if self.use_language_model:
            # Position Attention: 将 ViT 特征 (B, 128, 768) 转换为固定长度 (B, max_length, 768)
            self.pos_attn = PositionAttention(
                max_length=max_length,
                in_channels=embed_dim,
                num_channels=64
            )
            # Visual head for position attention output
            self.vis_cls = nn.Linear(embed_dim, nb_cls)
            
            # BCN Language Model
            self.language_model = BCNLanguage(
                num_classes=nb_cls,
                max_length=max_length,
                d_model=512,  # Language model dimension
                nhead=8,
                d_inner=2048,
                dropout=0.1,
                num_layers=4,
                detach=True  # Detach visual input for stable training
            )
            
            # Gated Fusion (optional, can use simple residual)
            self.use_gated_fusion = False
            if self.use_gated_fusion:
                self.fusion = GatedFusion(d_model=nb_cls)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # pos_embed = get_2d_sincos_pos_embed(self.embed_dim, [1, self.nb_query])
        # self.qry_tokens.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        """
        生成 Span Mask，确保不重叠以达到目标 mask 比例
        
        Args:
            x: 输入 tensor (N, L, D)
            mask_ratio: 目标 mask 比例，例如 0.4
            max_span_length: 每个 span 的长度，例如 8
            
        Returns:
            mask: (N, L, 1) tensor，1 表示保留，0 表示 mask
        """
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        
        # 计算需要 mask 的 token 数量
        num_mask = int(L * mask_ratio)
        # 计算需要的 span 数量
        num_spans = num_mask // max_span_length
        
        if num_spans == 0:
            return mask
            
        # 为每个 batch 独立生成不重叠的 span 位置
        for b in range(N):
            # 将序列划分为多个可能的 span 起始位置区间，避免重叠
            available_positions = list(range(0, L - max_span_length + 1))
            selected_positions = []
            
            for _ in range(num_spans):
                if not available_positions:
                    break
                # 随机选择一个位置
                idx = torch.randint(len(available_positions), (1,)).item()
                pos = available_positions[idx]
                selected_positions.append(pos)
                
                # 移除会导致重叠的位置
                # 新 span 的范围是 [pos, pos + max_span_length)
                # 需要移除 [pos - max_span_length + 1, pos + max_span_length - 1] 范围内的起始位置
                remove_start = max(0, pos - max_span_length + 1)
                remove_end = min(L - max_span_length, pos + max_span_length - 1)
                available_positions = [p for p in available_positions 
                                       if p < remove_start or p > remove_end]
            
            # 应用 mask
            for pos in selected_positions:
                mask[b, pos:pos + max_span_length, :] = 0
                
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        """
        Forward pass with optional language model branch
        
        Args:
            x: Input image (B, 1, H, W)
            mask_ratio: Mask ratio for span masking
            max_span_length: Maximum span length for masking
            use_masking: Whether to apply masking (training only)
            
        Returns:
            dict with:
                - 'ctc': (B, L, C) CTC logits, L=128
                - 'attn': (B, T, C) Attention logits, T=max_length (if use_language_model)
                - 'vis_logits': (B, T, C) Visual logits before language model
                - 'lang_logits': (B, T, C) Language model logits
        """
        # embed patches
        x = self.layer_norm(x)
        x = self.patch_embed(x)  # (B, C, H', W') = (B, embed_dim, 1, 128)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, L, C) where L = H' * W' = 128
        
        # masking: length -> length * mask_ratio
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        x = x + self.pos_embed
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        visual_feat = self.norm(x)  # (B, 128, 768) - ViT encoder output
        
        # --------------------------------------------------------------------------
        # Branch 1: CTC
        ctc_logits = self.head(visual_feat)  # (B, 128, nb_cls)
        ctc_logits = self.layer_norm(ctc_logits)
        
        # --------------------------------------------------------------------------
        # Branch 2: ABINet Language Branch
        if self.use_language_model:
            # Position Attention: (B, 128, 768) -> (B, max_length, 768)
            attn_vecs, attn_scores = self.pos_attn(visual_feat)  # (B, T, E)
            
            # Visual classification
            vis_logits = self.vis_cls(attn_vecs)  # (B, T, nb_cls)
            
            # Language Model: refine using BCN
            vis_probs = torch.softmax(vis_logits, dim=-1)  # (B, T, nb_cls)
            lang_output = self.language_model(vis_probs)  # dict with 'logits'
            lang_logits = lang_output['logits']  # (B, T, nb_cls)
            
            # Fusion: simple residual or gated
            if self.use_gated_fusion:
                fused_logits = self.fusion(vis_logits, lang_logits)
            else:
                # Simple residual fusion
                fused_logits = vis_logits + lang_logits
            
            return {
                'ctc': ctc_logits,           # (B, 128, C) for CTC loss
                'attn': fused_logits,        # (B, T, C) for CE loss (final output)
                'vis_logits': vis_logits,    # (B, T, C) visual logits
                'lang_logits': lang_logits,  # (B, T, C) language logits
            }
        else:
            # CTC only mode (backward compatible)
            return ctc_logits


def create_model(nb_cls, img_size, max_length=26, use_language_model=True, **kwargs):
    """
    创建 HTR-VT 模型 (支持 CTC + ABINet 语言模型双分支)
    
    根据论文 Li et al., 2025 的规范:
    - embed_dim = 768
    - depth (layers) = 4  
    - num_heads = 6
    - mlp_ratio = 4 (Hidden dim = 3072)
    - 2D Sin-Cos positional embedding (Fixed)
    
    新增 ABINet 语言分支:
    - PositionAttention: 将变长视觉特征转为固定长度
    - BCNLanguage: 双向完形填空网络进行语言建模
    
    Args:
        nb_cls: 类别数量 (字符表大小 + 1 for CTC blank)
        img_size: [H, W] 图像尺寸，默认 [64, 512]
        max_length: 最大序列长度 (ABINet 分支输出长度)
        use_language_model: 是否启用语言模型分支
        
    Returns:
        model: MaskedAutoencoderViT instance
        
    Notes:
        - patch_size=(4, 64) 表示 W 方向下采样 4 倍，H 方向压缩到 1
        - ResNet 输出 L=128 个 tokens (512/4=128)
        - 当 use_language_model=True 时，forward 返回 dict
        - 当 use_language_model=False 时，forward 返回 tensor (向后兼容)
    """
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),  # W方向stride=4, H方向全压缩
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 max_length=max_length,
                                 use_language_model=use_language_model,
                                 **kwargs)
    return model

