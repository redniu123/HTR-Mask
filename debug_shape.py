import torch
from model.HTR_VT import create_model


def test_shape_alignment():
    print(">>> 🚀 Starting Shape Dry Run...")

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ℹ️ Using device: {device}")

    # 1. 配置参数 (模拟 Phase 2 配置)
    nb_cls = 80  # 字符集大小
    img_size = [512, 64]  # [W, H] - 与 option.py 中的格式一致
    max_len = 26  # ABINet 分支的最大预测长度
    bs = 2  # 模拟 Batch Size

    # 2. 初始化模型 (启用 Language Model 分支)
    # 注意: create_model 期望 img_size 为 [H, W] 格式，所以需要反转
    # 这与 train.py 中的 img_size[::-1] 一致
    try:
        model = create_model(
            nb_cls=nb_cls,
            img_size=img_size[::-1],  # [W, H] -> [H, W] = [64, 512]
            use_language_model=True,  # <--- 关键开关
            max_length=max_len,
        ).to(device)
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. 构造伪造数据
    # 输入: (B, C, H, W) -> HTR-VT 接受 grayscale (1 channel) 或 RGB (3)
    # 注意: dataset.py 里通常是 (1, 64, 512)
    dummy_input = torch.randn(bs, 1, 64, 512).to(device)
    print(f"ℹ️ Input Shape: {dummy_input.shape}")

    # 4. 前向传播测试
    try:
        # 先测试各个子模块的输出形状
        print("\n--- Debug: Testing submodules ---")

        with torch.no_grad():
            # Test ResNet + ViT encoder
            x = model.layer_norm(dummy_input)
            x = model.patch_embed(x)
            print(f"After patch_embed (ResNet): {x.shape}")

            b, c, h, w = x.shape
            x = x.view(b, c, -1).permute(0, 2, 1)
            print(f"After reshape to sequence: {x.shape}")

            x = x + model.pos_embed
            for i, blk in enumerate(model.blocks):
                x = blk(x)
            x = model.norm(x)
            print(f"After ViT blocks (visual_feat): {x.shape}")

            # Test CTC head
            ctc_logits = model.head(x)
            ctc_logits = model.layer_norm(ctc_logits)
            print(f"CTC logits: {ctc_logits.shape}")

            # Test Position Attention
            print("\n--- Debug: Testing PositionAttention ---")
            attn_vecs, attn_scores = model.pos_attn(x)
            print(f"PositionAttention output: {attn_vecs.shape}")

            # Test visual classifier
            vis_logits = model.vis_cls(attn_vecs)
            print(f"Visual logits: {vis_logits.shape}")

            # Test Language Model
            print("\n--- Debug: Testing BCNLanguage ---")
            vis_probs = torch.softmax(vis_logits, dim=-1)
            lang_output = model.language_model(vis_probs)
            lang_logits = lang_output["logits"]
            print(f"Language logits: {lang_logits.shape}")

        print("\n--- Full Forward Pass ---")
        output = model(dummy_input)

        # 5. 检查输出结构
        if isinstance(output, dict):
            ctc_out = output.get("ctc")
            attn_out = output.get("attn")
            print("✅ Forward pass returned a dictionary.")
        else:
            print(f"❌ Forward pass returned {type(output)}, expected dict.")
            return

        # 6. 检查 CTC 分支维度
        # 预期: (B, 128, nb_cls)
        if ctc_out is not None:
            print(
                f"✅ CTC Output Shape: {ctc_out.shape} (Expected: [{bs}, 128, {nb_cls}])"
            )
        else:
            print("❌ CTC Output is None!")

        # 7. 检查 ABINet 分支维度 (最关键!)
        # 预期: (B, max_len, nb_cls) -> (2, 26, 80)
        if attn_out is not None:
            if attn_out.shape == (bs, max_len, nb_cls):
                print(
                    f"✅ Attention Output Shape: {attn_out.shape} MATCHES Expected: [{bs}, {max_len}, {nb_cls}]"
                )
            else:
                print(
                    f"❌ Attention Output Shape Mismatch! Got {attn_out.shape}, Expected [{bs}, {max_len}, {nb_cls}]"
                )
        else:
            print("❌ Attention Output is None! Check 'use_language_model' flag.")

    except RuntimeError as e:
        print(f"❌ Runtime Error during forward: {e}")
        import traceback

        traceback.print_exc()
        # 常见错误提示: "mat1 and mat2 shapes cannot be multiplied" 通常意味着 Linear 层输入不对


if __name__ == "__main__":
    test_shape_alignment()
