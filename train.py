import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial

# Wandb for experiment tracking
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# ============================================================
# Loss Computation Functions
# ============================================================


def compute_ctc_loss(preds, criterion, text, length, batch_size):
    """计算 CTC Loss"""
    preds = preds.float()  # CTC Loss 要求 float32
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    preds = preds.permute(1, 0, 2).log_softmax(2)

    torch.backends.cudnn.enabled = False
    loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True
    return loss


def compute_attn_loss(logits, targets, ignore_index=0):
    """
    计算 Attention 分支的 CrossEntropy Loss

    Args:
        logits: (B, T, C) - 模型输出
        targets: (B, T) - 目标标签 (PAD_IDX=0 will be ignored)
        ignore_index: 忽略的标签索引 (PAD)
    """
    B, T, C = logits.shape
    logits_flat = logits.view(-1, C)  # (B*T, C)
    targets_flat = targets.view(-1)  # (B*T,)
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
    return loss


def compute_hybrid_loss(
    args,
    model,
    image,
    batch_size,
    ctc_criterion,
    ctc_text,
    ctc_length,
    attn_targets=None,
    use_language_model=True,
    lambda_ctc=1.0,
    lambda_attn=1.0,
    lambda_lang=1.0,
):
    """
    计算混合损失 (CTC + Attention)

    Loss = λ_ctc * L_CTC + λ_attn * L_CE(vis_logits) + λ_lang * L_CE(fused_logits)

    Args:
        args: 训练参数
        model: HTR-VT 模型
        image: 输入图像 (B, 1, H, W)
        batch_size: 批次大小
        ctc_criterion: CTC Loss 函数
        ctc_text: CTC 编码的文本
        ctc_length: CTC 文本长度
        attn_targets: Attention 分支的目标 (B, T)
        use_language_model: 是否使用语言模型
        lambda_ctc: CTC 损失权重
        lambda_attn: Visual Attention 损失权重
        lambda_lang: Language Model 损失权重

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    # Forward pass
    outputs = model(image, args.mask_ratio, args.max_span_length, use_masking=True)

    if use_language_model and isinstance(outputs, dict):
        # Hybrid mode: CTC + Attention
        ctc_logits = outputs["ctc"]  # (B, 128, C)
        attn_logits = outputs["attn"]  # (B, T, C) - fused output
        vis_logits = outputs["vis_logits"]  # (B, T, C)
        lang_logits = outputs["lang_logits"]  # (B, T, C)

        # CTC Loss
        loss_ctc = compute_ctc_loss(
            ctc_logits, ctc_criterion, ctc_text, ctc_length, batch_size
        )

        # Attention Loss (on fused output)
        loss_attn = compute_attn_loss(attn_logits, attn_targets, ignore_index=0)

        # Visual Attention Loss (auxiliary)
        loss_vis = compute_attn_loss(vis_logits, attn_targets, ignore_index=0)

        # Total loss
        total_loss = (
            lambda_ctc * loss_ctc + lambda_attn * loss_vis + lambda_lang * loss_attn
        )

        loss_dict = {
            "ctc": loss_ctc.item(),
            "attn": loss_attn.item(),
            "vis": loss_vis.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
    else:
        # CTC only mode (backward compatible)
        preds = outputs if not isinstance(outputs, dict) else outputs["ctc"]
        loss = compute_ctc_loss(preds, ctc_criterion, ctc_text, ctc_length, batch_size)
        return loss, {"ctc": loss.item(), "total": loss.item()}


def main():
    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Initialize wandb if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="HTR-Mask",
            name=args.exp_name,
            config=vars(args),
            dir=args.save_dir,
        )
        logger.info("Wandb initialized successfully!")
    elif args.use_wandb and not WANDB_AVAILABLE:
        logger.warning("Wandb requested but not available. Using tensorboard only.")

    writer = SummaryWriter(args.save_dir)

    # ============================================================
    # Model Configuration
    # ============================================================
    max_length = getattr(args, "max_length", 26)  # 默认最大序列长度
    use_language_model = getattr(args, "use_language_model", True)  # 是否使用语言模型

    model = HTR_VT.create_model(
        nb_cls=args.nb_cls,
        img_size=args.img_size[::-1],
        max_length=max_length,
        use_language_model=use_language_model,
    )

    total_param = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_param:,}")
    logger.info(f"Language Model: {'Enabled' if use_language_model else 'Disabled'}")
    logger.info(f"Max sequence length: {max_length}")

    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    logger.info("Loading train loader...")
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=partial(dataset.SameTrCollate, args=args),
    )
    train_iter = dataset.cycle_data(train_loader)

    logger.info("Loading val loader...")
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    optimizer = sam.SAM(
        model.parameters(),
        torch.optim.AdamW,
        lr=1e-7,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    # ============================================================
    # Loss Functions & Converters
    # ============================================================
    # CTC Loss & Converter
    ctc_criterion = torch.nn.CTCLoss(reduction="none", zero_infinity=True)
    ctc_converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    # Attention Loss & Converter (for language model branch)
    attn_converter = None
    if use_language_model:
        attn_converter = utils.AttnLabelConverter(
            train_dataset.ralph.values(), max_length=max_length
        )
        logger.info(
            f"AttnLabelConverter initialized with {attn_converter.num_classes} classes"
        )

    best_cer, best_wer = 1e6, 1e6
    train_loss = 0.0
    train_loss_ctc = 0.0
    train_loss_attn = 0.0

    # Loss weights (可以通过 args 配置)
    lambda_ctc = getattr(args, "lambda_ctc", 1.0)
    lambda_attn = getattr(args, "lambda_attn", 1.0)
    lambda_lang = getattr(args, "lambda_lang", 1.0)

    # ---- Resume from checkpoint if exists ----
    checkpoint_path = os.path.join(args.save_dir, "best_CER.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # Load model weights (strict=False 以支持新旧模型兼容)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
            logger.info("Loaded model weights from checkpoint (strict)")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            model.load_state_dict(checkpoint["model"], strict=False)
            logger.info("Loaded model weights from checkpoint (non-strict)")

        # Load EMA weights
        if "state_dict_ema" in checkpoint:
            try:
                model_ema.ema.load_state_dict(checkpoint["state_dict_ema"], strict=True)
                logger.info("Loaded EMA weights from checkpoint (strict)")
            except RuntimeError:
                model_ema.ema.load_state_dict(
                    checkpoint["state_dict_ema"], strict=False
                )
                logger.info("Loaded EMA weights from checkpoint (non-strict)")

        # NOTE: We intentionally do NOT load optimizer state to reset momentum
        logger.info("Optimizer state NOT loaded (momentum reset for fresh start)")

        del checkpoint  # Free memory
        torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint found, starting training from scratch")

    #### ---- train & eval ---- ####

    # Create progress bar
    pbar = tqdm(range(1, args.total_iter), desc="Training", unit="iter")

    for nb_iter in pbar:
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer
        )

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        batch_size = image.size(0)

        # Encode labels for both branches
        ctc_text, ctc_length = ctc_converter.encode(batch[1])
        attn_targets = None
        if use_language_model and attn_converter is not None:
            attn_targets, attn_lengths = attn_converter.encode(batch[1])

        # ============================================================
        # First forward-backward pass (SAM Step 1)
        # ============================================================
        loss, loss_dict = compute_hybrid_loss(
            args,
            model,
            image,
            batch_size,
            ctc_criterion,
            ctc_text,
            ctc_length,
            attn_targets=attn_targets,
            use_language_model=use_language_model,
            lambda_ctc=lambda_ctc,
            lambda_attn=lambda_attn,
            lambda_lang=lambda_lang,
        )

        # === [Safety] NaN/Inf Loss Protection ===
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"⚠️ Warning: NaN/Inf loss detected at Iter {nb_iter}. Skipping batch to prevent crash."
            )
            optimizer.zero_grad()
            continue
        # ========================================

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.first_step(zero_grad=True)

        # ============================================================
        # Second forward-backward pass (SAM Step 2)
        # ============================================================
        loss2, _ = compute_hybrid_loss(
            args,
            model,
            image,
            batch_size,
            ctc_criterion,
            ctc_text,
            ctc_length,
            attn_targets=attn_targets,
            use_language_model=use_language_model,
            lambda_ctc=lambda_ctc,
            lambda_attn=lambda_attn,
            lambda_lang=lambda_lang,
        )
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.second_step(zero_grad=True)

        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)

        # Accumulate losses
        train_loss += loss_dict["total"]
        train_loss_ctc += loss_dict.get("ctc", 0)
        train_loss_attn += loss_dict.get("attn", 0)

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss_dict['total']:.4f}",
                "ctc": f"{loss_dict.get('ctc', 0):.3f}",
                "attn": f"{loss_dict.get('attn', 0):.3f}",
                "lr": f"{current_lr:.6f}",
                "best_cer": f"{best_cer:.4f}" if best_cer < 1e6 else "N/A",
            }
        )

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter
            train_loss_ctc_avg = train_loss_ctc / args.print_iter
            train_loss_attn_avg = train_loss_attn / args.print_iter

            logger.info(
                f"Iter : {nb_iter} \t LR : {current_lr:0.5f} \t "
                f"Total : {train_loss_avg:0.4f} \t CTC : {train_loss_ctc_avg:0.4f} \t "
                f"Attn : {train_loss_attn_avg:0.4f}"
            )

            writer.add_scalar("./Train/lr", current_lr, nb_iter)
            writer.add_scalar("./Train/train_loss", train_loss_avg, nb_iter)
            writer.add_scalar("./Train/loss_ctc", train_loss_ctc_avg, nb_iter)
            writer.add_scalar("./Train/loss_attn", train_loss_attn_avg, nb_iter)

            # Log to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/loss": train_loss_avg,
                        "train/loss_ctc": train_loss_ctc_avg,
                        "train/loss_attn": train_loss_attn_avg,
                        "train/lr": current_lr,
                        "iter": nb_iter,
                    }
                )

            train_loss = 0.0
            train_loss_ctc = 0.0
            train_loss_attn = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(
                    model_ema.ema, ctc_criterion, val_loader, ctc_converter
                )

                if val_cer < best_cer:
                    logger.info(f"CER improved from {best_cer:.4f} to {val_cer:.4f}!!!")
                    best_cer = val_cer
                    checkpoint = {
                        "model": model.state_dict(),
                        "state_dict_ema": model_ema.ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, "best_CER.pth"))

                if val_wer < best_wer:
                    logger.info(f"WER improved from {best_wer:.4f} to {val_wer:.4f}!!!")
                    best_wer = val_wer
                    checkpoint = {
                        "model": model.state_dict(),
                        "state_dict_ema": model_ema.ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, "best_WER.pth"))

                logger.info(
                    f"Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f}"
                )

                writer.add_scalar("./VAL/CER", val_cer, nb_iter)
                writer.add_scalar("./VAL/WER", val_wer, nb_iter)
                writer.add_scalar("./VAL/bestCER", best_cer, nb_iter)
                writer.add_scalar("./VAL/bestWER", best_wer, nb_iter)
                writer.add_scalar("./VAL/val_loss", val_loss, nb_iter)

                # Log validation metrics to wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/cer": val_cer,
                            "val/wer": val_wer,
                            "val/best_cer": best_cer,
                            "val/best_wer": best_wer,
                            "iter": nb_iter,
                        }
                    )

                model.train()

    # Close progress bar
    pbar.close()

    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        logger.info("Wandb run finished.")


if __name__ == "__main__":
    main()
