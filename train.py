import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
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


def compute_loss(args, model, image, batch_size, criterion, text, length, use_amp=True):
    with autocast(device_type='cuda', enabled=use_amp):
        preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
        preds = preds.permute(1, 0, 2).log_softmax(2)

        # CTC loss needs float32, so we cast here
        torch.backends.cudnn.enabled = False
        loss = criterion(preds.float(), text.cuda(), preds_size, length.cuda()).mean()
        torch.backends.cudnn.enabled = True
    return loss


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

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    total_param = sum(p.numel() for p in model.parameters())
    logger.info("total_param is {}".format(total_param))

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
    criterion = torch.nn.CTCLoss(reduction="none", zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    # Mixed Precision Training (AMP)
    # 注意：SAM 优化器需要两个 scaler，因为它有两步更新
    scaler1 = GradScaler('cuda', enabled=args.use_amp)
    scaler2 = GradScaler('cuda', enabled=args.use_amp)
    if args.use_amp:
        logger.info("Mixed Precision Training (AMP) enabled - utilizing Tensor Cores!")
    else:
        logger.info("Mixed Precision Training (AMP) disabled")

    best_cer, best_wer = 1e6, 1e6
    train_loss = 0.0

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
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)

        # First forward-backward pass with AMP
        loss = compute_loss(
            args,
            model,
            image,
            batch_size,
            criterion,
            text,
            length,
            use_amp=args.use_amp,
        )
        scaler1.scale(loss).backward()

        # SAM first step - unscale gradients and take first step
        scaler1.unscale_(optimizer)
        optimizer.first_step(zero_grad=True)
        scaler1.update()

        # Second forward-backward pass with AMP
        loss2 = compute_loss(
            args,
            model,
            image,
            batch_size,
            criterion,
            text,
            length,
            use_amp=args.use_amp,
        )
        scaler2.scale(loss2).backward()

        # SAM second step with scaler
        scaler2.unscale_(optimizer)
        optimizer.second_step(zero_grad=True)
        scaler2.update()

        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.6f}",
                "best_cer": f"{best_cer:.4f}" if best_cer < 1e6 else "N/A",
            }
        )

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter

            logger.info(
                f"Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f}"
            )

            writer.add_scalar("./Train/lr", current_lr, nb_iter)
            writer.add_scalar("./Train/train_loss", train_loss_avg, nb_iter)

            # Log to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/loss": train_loss_avg,
                        "train/lr": current_lr,
                        "iter": nb_iter,
                    }
                )

            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(
                    model_ema.ema, criterion, val_loader, converter
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
