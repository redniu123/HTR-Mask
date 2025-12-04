import torch
from model.HTR_VT import create_model


def test_shape_alignment():
    print(">>> 🚀 Starting Shape Dry Run...")

    # 1. 配置参数 (模拟 Phase 2 配置)
    nb_cls = 80  # 字符集大小
    img_size = [512, 64]  # [W, H]
    max_len = 26  # ABINet 分支的最大预测长度
    bs = 2  # 模拟 Batch Size

    # 2. 初始化模型 (启用 Language Model 分支)
    try:
        model = create_model(
            nb_cls=nb_cls,
            img_size=img_size,
            use_language_model=True,  # <--- 关键开关
            max_length=max_len,
        ).cuda()
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        return

    # 3. 构造伪造数据
    # 输入: (B, C, H, W) -> HTR-VT 接受 grayscale (1 channel) 或 RGB (3)
    # 注意: dataset.py 里通常是 (1, 64, 512)
    dummy_input = torch.randn(bs, 1, 64, 512).cuda()
    print(f"ℹ️ Input Shape: {dummy_input.shape}")

    # 4. 前向传播测试
    try:
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
        # 常见错误提示: "mat1 and mat2 shapes cannot be multiplied" 通常意味着 Linear 层输入不对


if __name__ == "__main__":
    test_shape_alignment()
