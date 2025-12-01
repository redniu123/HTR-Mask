# Implementation Spec 01: HTR-VT Baseline Reproduction

## 1. 架构概述 (Architecture Overview)

复现 `Li et al., 2025` 提出的 HTR-VT 模型。核心是将图像切片序列化后，通过 ViT Encoder 提取上下文特征，最后通过 CTC 解码。

### Data Flow

$$Image (H, W) \xrightarrow{CNN} Features (C, H', W') \xrightarrow{Flatten} Tokens (L, C) \xrightarrow{ViT + Mask} Context (L, D) \xrightarrow{Linear} Logits (L, \text{vocab}) \xrightarrow{CTC} Text$$

## 2. 核心组件参数 (Key Components & Shapes)

### A. CNN Feature Extractor (Modified ResNet-18)

- **Source**: `model/resnet18.py` (Need verification)
- [cite_start]**Modification**: [cite: 656] 论文提到移除了最后一个 Residual Block，并调整了 Stride。
- [cite_start]**Input**: `(B, 1 or 3, 64, 512)` (Fixed Height: 64, Width: 512 [cite: 685])
- **Output**: Feature Map `(B, C, 1, W_feat)` -> Flatten to `(B, W_feat, C)`
  - [cite_start]_Check_: 论文中 $L=128$ [cite: 861]。这意味着下采样倍率为 $512 / 128 = 4$。
  - **Constraint**: 确保 ResNet 的总 Stride 在 W 维度是 4，在 H 维度是 64 (压缩到 1)。

### B. ViT Encoder

- **Source**: `model/HTR_VT.py`
- [cite_start]**Config**[cite: 676]:
  - `depth (layers)` = 4
  - `embed_dim` = 768
  - `num_heads` = 6
  - `mlp_ratio` = 4 (Hidden dim = 3072)
- [cite_start]**Positional Embedding**: 2D Sin-Cos embedding (Fixed)[cite: 642].

### C. Masking Strategy (Span Mask)

- **Source**: `model/HTR_VT.py` -> `generate_span_mask`
- **Logic**: 随机 Mask 掉连续的 Token 片段。
- [cite_start]**Params**[cite: 679]:
  - `mask_ratio` = 0.4
  - `span_length` = 8
  - _Note_: 在 Training 时开启，Testing 时关闭。

### D. Optimization (SAM)

- [cite_start]**Algorithm**: Sharpness-Aware Minimization[cite: 666].
- **Requires**: 需要在训练循环中实现两次 Backward pass (一次基于原始 Loss，一次基于 Perturbed Loss)。
- **Params**: `rho` (neighborhood size) 需要确认默认值 (通常 0.05)。

## 3. 待办检查项 (Action Items for Engineer)

1.  **检查 `resnet18.py`**: 确认其 `forward` 输出的 Tensor 形状是否严格满足 `(B, C, 1, 128)`。如果不满足，需要调整 Pooling 或 Stride。
2.  **数据加载器 (Dataset)**:
    - 实现 `data/dataset.py`。
    - [cite_start]确保 Image Resize 操作是 `(512, 64)` 且保持长宽比 (Pad 还是 Stretch? 论文说是 "fix the input image resolution to 512x64" [cite: 685]，通常 implies stretching or padding. 需检查代码实现)。
3.  **SAM 集成**: 标准 PyTorch 优化器不支持 SAM，需要手动实现训练步逻辑：

    ```python
    # Pseudo-code for SAM step
    loss = criterion(model(images), targets)
    loss.backward()
    optimizer.first_step(zero_grad=True) # Perturb weights

    criterion(model(images), targets).backward() # Second pass
    optimizer.second_step(zero_grad=True) # Update weights
    ```

## 4. 成功标准 (Success Metrics)

运行 `iam.sh` 后，Validation Set 达到：

- **CER**: < 4.0%
- **WER**: < 12.0%
