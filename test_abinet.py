"""
test_abinet.py - Evaluate the Attention/Language Branch of HTR-VT

This script evaluates the ABINet language branch output (not CTC).
Uses greedy decoding on attention logits instead of CTC decoding.

Usage:
    python test_abinet.py --exp-name iam IAM
"""

import torch
import torch.utils.data

import os
import re
import json
import editdistance
from collections import OrderedDict

from utils import utils
from utils import option
from data import dataset
from model import HTR_VT


def greedy_decode_attn(
    attn_logits: torch.Tensor, converter: utils.AttnLabelConverter
) -> list:
    """
    Greedy decoding for Attention logits

    Args:
        attn_logits: (B, T, C) tensor of logits
        converter: AttnLabelConverter instance

    Returns:
        list of decoded strings
    """
    # Greedy: take argmax at each position
    preds = attn_logits.argmax(dim=-1)  # (B, T)

    # Use converter's decode method
    pred_strs = converter.decode(preds)

    return pred_strs


def evaluate_attention_branch(model, test_loader, attn_converter, logger):
    """
    Evaluate the Attention/Language branch of the model

    Args:
        model: HTR-VT model with use_language_model=True
        test_loader: DataLoader for test set
        attn_converter: AttnLabelConverter instance
        logger: Logger instance

    Returns:
        cer: Character Error Rate
        wer: Word Error Rate
        all_preds: List of predicted strings
        all_labels: List of ground truth strings
    """
    model.eval()

    tot_ED = 0  # Total edit distance (characters)
    tot_ED_wer = 0  # Total edit distance (words)
    length_of_gt = 0
    length_of_gt_wer = 0

    all_preds = []
    all_labels = []

    logger.info("Evaluating Attention Branch...")

    with torch.no_grad():
        for batch_idx, (image_tensors, labels) in enumerate(test_loader):
            batch_size = image_tensors.size(0)
            images = image_tensors.cuda()

            # Forward pass
            outputs = model(images)

            # Check output format
            if not isinstance(outputs, dict):
                raise RuntimeError(
                    "Model output is not a dict. "
                    "Make sure model was created with use_language_model=True"
                )

            # Extract attention logits
            attn_logits = outputs["attn"]  # (B, T, C)

            # Greedy decode
            preds_str = greedy_decode_attn(attn_logits, attn_converter)

            all_preds.extend(preds_str)
            all_labels.extend(labels)

            # Calculate CER (Character Error Rate)
            for pred, gt in zip(preds_str, labels):
                ed = editdistance.eval(pred, gt)
                tot_ED += ed
                length_of_gt += len(gt) if len(gt) > 0 else 1

            # Calculate WER (Word Error Rate)
            for pred, gt in zip(preds_str, labels):
                pred_words = utils.format_string_for_wer(pred).split(" ")
                gt_words = utils.format_string_for_wer(gt).split(" ")
                ed_wer = editdistance.eval(pred_words, gt_words)
                tot_ED_wer += ed_wer
                length_of_gt_wer += len(gt_words) if len(gt_words) > 0 else 1

            # Progress logging
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Processed {batch_idx + 1}/{len(test_loader)} batches..."
                )

    # Calculate final metrics
    cer = tot_ED / float(length_of_gt) if length_of_gt > 0 else 0
    wer = tot_ED_wer / float(length_of_gt_wer) if length_of_gt_wer > 0 else 0

    return cer, wer, all_preds, all_labels


def main():
    args = option.get_args_parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info("=" * 60)
    logger.info("ABINet Attention Branch Evaluation")
    logger.info("=" * 60)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ============================================================
    # Load Dataset (needed for vocabulary)
    # ============================================================
    logger.info("Loading datasets...")
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size
    )
    test_dataset = dataset.myLoadDS(
        args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    logger.info(f"Test set size: {len(test_dataset)}")

    # ============================================================
    # Initialize Converters & Determine Vocabulary Size
    # ============================================================
    max_length = getattr(args, "max_length", 26)

    # Initialize AttnLabelConverter
    attn_converter = utils.AttnLabelConverter(
        train_dataset.ralph.values(), max_length=max_length
    )
    logger.info(
        f"AttnLabelConverter: {attn_converter.num_classes} classes, max_length={max_length}"
    )

    # Dynamic class adjustment (same as train.py)
    real_nb_cls = attn_converter.num_classes
    if real_nb_cls > args.nb_cls:
        logger.info(f"⚠️ Auto-adjusting nb_cls from {args.nb_cls} to {real_nb_cls}")
        args.nb_cls = real_nb_cls

    # ============================================================
    # Create Model with Language Branch
    # ============================================================
    logger.info("Creating model with Language Branch enabled...")
    model = HTR_VT.create_model(
        nb_cls=args.nb_cls,
        img_size=args.img_size[::-1],
        max_length=max_length,
        use_language_model=True,  # Must be True for attention branch
    )

    # ============================================================
    # Load Checkpoint
    # ============================================================
    # Try latest.pth first, then best_CER.pth
    latest_path = os.path.join(args.save_dir, "latest.pth")
    best_cer_path = os.path.join(args.save_dir, "best_CER.pth")

    if os.path.exists(latest_path):
        pth_path = latest_path
    elif os.path.exists(best_cer_path):
        pth_path = best_cer_path
    else:
        logger.error(f"No checkpoint found in {args.save_dir}")
        return

    logger.info(f"Loading checkpoint from: {pth_path}")

    ckpt = torch.load(pth_path, map_location="cpu")
    model_dict = OrderedDict()
    pattern = re.compile("module.")

    # Handle 'state_dict_ema' or 'model' key
    state_dict_key = "state_dict_ema" if "state_dict_ema" in ckpt else "model"

    for k, v in ckpt[state_dict_key].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict[k] = v

    # Load with strict=False to allow for architecture differences
    try:
        model.load_state_dict(model_dict, strict=True)
        logger.info("Loaded checkpoint (strict=True)")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        model.load_state_dict(model_dict, strict=False)
        logger.info("Loaded checkpoint (strict=False)")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # ============================================================
    # Evaluate Attention Branch
    # ============================================================
    cer, wer, preds, labels = evaluate_attention_branch(
        model, test_loader, attn_converter, logger
    )

    # ============================================================
    # Print Results
    # ============================================================
    logger.info("=" * 60)
    logger.info("📊 Attention Branch Results")
    logger.info("=" * 60)
    logger.info(f"  CER: {cer:.4f} ({cer * 100:.2f}%)")
    logger.info(f"  WER: {wer:.4f} ({wer * 100:.2f}%)")
    logger.info("=" * 60)

    # Print some sample predictions
    logger.info("\n📝 Sample Predictions (first 10):")
    for i in range(min(10, len(preds))):
        logger.info(f"  [{i + 1}] GT:   '{labels[i]}'")
        logger.info(f"       Pred: '{preds[i]}'")
        logger.info("")

    # Also evaluate CTC branch for comparison
    logger.info("\n--- For Comparison: CTC Branch ---")
    ctc_converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction="none", zero_infinity=True).to(device)

    # Quick CTC evaluation (reusing valid.validation)
    import valid

    with torch.no_grad():
        ctc_loss, ctc_cer, ctc_wer, _, _ = valid.validation(
            model, criterion, test_loader, ctc_converter
        )

    logger.info(f"  CTC Branch CER: {ctc_cer:.4f} ({ctc_cer * 100:.2f}%)")
    logger.info(f"  CTC Branch WER: {ctc_wer:.4f} ({ctc_wer * 100:.2f}%)")
    logger.info("=" * 60)

    # Summary comparison
    logger.info("\n📈 Summary Comparison:")
    logger.info(f"  {'Branch':<15} {'CER':>10} {'WER':>10}")
    logger.info(f"  {'-' * 35}")
    logger.info(f"  {'Attention':<15} {cer * 100:>9.2f}% {wer * 100:>9.2f}%")
    logger.info(f"  {'CTC':<15} {ctc_cer * 100:>9.2f}% {ctc_wer * 100:>9.2f}%")


if __name__ == "__main__":
    main()
