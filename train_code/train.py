import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import (
    AverageMeter,
    initialize_logger,
    save_checkpoint,
    record_loss,
    time2file_name,
    Loss_MRAE,
    Loss_MRAE_custom,
    Loss_RMSE,
    Loss_PSNR,
    Loss_SAM,
)
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy

# loss function
from losses import SSIMLoss, SIDLoss, ERGASLoss, DeltaE00Loss


class EMA:
    """Exponential Moving Average for model parameters.

    This helps stabilize training by maintaining a smoothed version of model weights.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--method", type=str, default="mst_plus_plus")
parser.add_argument(
    "--pretrained_model_path",
    type=str,
    default=None,
    help="Path to pretrained model for fine-tuning (only loads model weights)",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint for resuming training (loads model, optimizer, scheduler, EMA states)",
)
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument(
    "--outf", type=str, default="./exp/mst_plus_plus/", help="path log files"
)
parser.add_argument("--data_root", type=str, default="../dataset/")
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default="0", help="path log files")
parser.add_argument(
    "--upscale_factor",
    type=int,
    default=1,
    help="upscale factor for super-resolution (1=no upscaling, 2=2x SR)",
)
parser.add_argument("--lambda_l1", type=float, default=1.0, help="weight for L1 loss")
parser.add_argument("--lambda_sam", type=float, default=1.0, help="weight for SAM loss")
parser.add_argument(
    "--lambda_ssim", type=float, default=0.4, help="weight for SSIM loss"
)
parser.add_argument("--lambda_sid", type=float, default=0.6, help="weight for SID loss")
parser.add_argument(
    "--lambda_ergas", type=float, default=0.01, help="weight for ERGAS loss"
)
parser.add_argument(
    "--lambda_delta_e00", type=float, default=0.05, help="weight for DeltaE00 loss"
)
parser.add_argument(
    "--ema_decay", type=float, default=0.999, help="EMA decay rate for model weights"
)
parser.add_argument(
    "--use_ema",
    action="store_true",
    default=True,
    help="Use Exponential Moving Average",
)
parser.add_argument(
    "--lr_restart_epochs",
    type=int,
    default=10,
    help="Number of epochs for first restart cycle in CosineAnnealingWarmRestarts",
)
parser.add_argument(
    "--lr_restart_mult",
    type=int,
    default=2,
    help="Multiplier for restart cycle length (T_mult in CosineAnnealingWarmRestarts)",
)
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(
    data_root=opt.data_root,
    crop_size=opt.patch_size,
    bgr2rgb=True,
    arg=True,
    stride=opt.stride,
    upscale_factor=opt.upscale_factor,
    isTest=False,
)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(
    data_root=opt.data_root,
    bgr2rgb=True,
    upscale_factor=opt.upscale_factor,
    isTest=False,
)
print("Validation set samples: ", len(val_data))


criterion_l1 = (
    nn.L1Loss()
)  # Use L1 loss for training stability (MRAE unstable with values near zero)
criterion_mrae = Loss_MRAE_custom()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_ssim = SSIMLoss()
criterion_sid = SIDLoss()
criterion_ergas = ERGASLoss(upscale_factor=opt.upscale_factor)
criterion_delta_e00 = DeltaE00Loss()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(
    method, pretrained_model_path, upscale_factor=opt.upscale_factor
).cuda()
print("Parameters number is ", sum(param.numel() for param in model.parameters()))

# output path
# If resuming, use the original checkpoint's folder; otherwise create new folder with timestamp
if opt.resume is not None and os.path.isfile(opt.resume):
    # Extract the directory from the resume checkpoint path
    opt.outf = os.path.dirname(opt.resume)
    print(f"Resuming: using existing output folder '{opt.outf}'")
else:
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    opt.outf = opt.outf + date_time
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_l1.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_sam.cuda()
    criterion_ssim.cuda()
    criterion_sid.cuda()
    criterion_ergas.cuda()
    criterion_delta_e00.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# logging
log_dir = os.path.join(opt.outf, "train.log")
logger = initialize_logger(log_dir)


def main():
    cudnn.benchmark = False
    best_psnr = 0
    writer = SummaryWriter(log_dir=os.path.join(opt.outf, "tensorboard"))

    # Create data loaders once outside epoch loop
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), capturable=True
    )
    # CosineAnnealingWarmRestarts: periodic restarts to escape local minima
    # T_0: steps in first cycle, T_mult: multiplier for subsequent cycles
    T_0_steps = opt.lr_restart_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0_steps, T_mult=opt.lr_restart_mult, eta_min=1e-6
    )
    # Store scheduler config for saving and resume validation
    scheduler_config = {
        "lr_restart_epochs": opt.lr_restart_epochs,
        "lr_restart_mult": opt.lr_restart_mult,
        "eta_min": 1e-6,
    }
    logger.info(
        f"Using CosineAnnealingWarmRestarts: T_0={opt.lr_restart_epochs} epochs ({T_0_steps} steps), T_mult={opt.lr_restart_mult}"
    )

    # Load pretrained model (fine-tuning, only model weights)
    if opt.pretrained_model_path is not None:
        if os.path.isfile(opt.pretrained_model_path):
            checkpoint = torch.load(opt.pretrained_model_path)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(
                f"=> Loaded pretrained model from '{opt.pretrained_model_path}'"
            )

    # Resume training (loads full state: model, optimizer, scheduler, EMA)
    last_epoch = 0
    ema_shadow_loaded = None
    if opt.resume is not None:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_epoch = checkpoint["epoch"]

            # Restore scheduler state if available
            if "scheduler" in checkpoint:
                # Check if scheduler config matches
                if "scheduler_config" in checkpoint:
                    saved_config = checkpoint["scheduler_config"]
                    if saved_config["lr_restart_epochs"] != opt.lr_restart_epochs:
                        logger.warning(
                            f"lr_restart_epochs mismatch: checkpoint={saved_config['lr_restart_epochs']}, "
                            f"current={opt.lr_restart_epochs}. Using checkpoint value."
                        )
                        opt.lr_restart_epochs = saved_config["lr_restart_epochs"]
                        T_0_steps = opt.lr_restart_epochs * len(train_loader)
                    if saved_config["lr_restart_mult"] != opt.lr_restart_mult:
                        logger.warning(
                            f"lr_restart_mult mismatch: checkpoint={saved_config['lr_restart_mult']}, "
                            f"current={opt.lr_restart_mult}. Using checkpoint value."
                        )
                        opt.lr_restart_mult = saved_config["lr_restart_mult"]
                    # Update scheduler_config to match checkpoint
                    scheduler_config = saved_config
                scheduler.load_state_dict(checkpoint["scheduler"])
                logger.info("Scheduler state restored from checkpoint")
            else:
                # If no scheduler state, recreate scheduler with correct last_epoch
                resumed_step = last_epoch * len(train_loader)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=T_0_steps,
                    T_mult=opt.lr_restart_mult,
                    eta_min=1e-6,
                    last_epoch=resumed_step - 1,
                )
                logger.info(f"Scheduler recreated at step {resumed_step}")

            # Load EMA shadow weights if available (will be applied after EMA init)
            if "ema_shadow" in checkpoint:
                ema_shadow_loaded = checkpoint["ema_shadow"]
                logger.info("EMA shadow weights loaded from checkpoint")

            logger.info(
                f"=> Resuming training from '{opt.resume}' at epoch {checkpoint['epoch']}"
            )

    # Initialize EMA for model weight smoothing (reduces validation metric oscillation)
    ema = None
    if opt.use_ema:
        ema = EMA(model, decay=opt.ema_decay)
        # Restore EMA shadow weights if resuming
        if ema_shadow_loaded is not None:
            ema.shadow = ema_shadow_loaded
            logger.info(f"EMA shadow weights restored")
        logger.info(f"EMA enabled with decay={opt.ema_decay}")

    # Standard epoch-based training loop
    for epoch in range(opt.end_epoch):
        if epoch < last_epoch:
            continue
        model.train()
        losses_total = AverageMeter()
        losses_l1 = AverageMeter()
        losses_sam = AverageMeter()
        losses_ssim = AverageMeter()
        losses_sid = AverageMeter()
        losses_ergas = AverageMeter()
        losses_delta_e00 = AverageMeter()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{opt.end_epoch}",
        )
        for step, (images, labels) in pbar:
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()
            output = model(images)

            # Combined loss
            loss = 0
            loss_l1, loss_sam, loss_ssim, loss_sid, loss_ergas, loss_delta_e00 = (
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(0),
            )

            # Check for NaN in model output first
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning(
                    f"NaN/Inf detected in model output at epoch {epoch + 1}, step {step}. Skipping batch."
                )
                optimizer.zero_grad()
                scheduler.step()
                continue

            skip_batch = False
            if opt.lambda_l1 > 0:
                loss_l1 = criterion_l1(output, labels)
                if torch.isnan(loss_l1) or torch.isinf(loss_l1):
                    logger.warning(
                        f"L1 loss is NaN/Inf at epoch {epoch + 1}, step {step}. Skipping batch."
                    )
                    skip_batch = True
                else:
                    loss += opt.lambda_l1 * loss_l1
                    losses_l1.update(loss_l1.item())

            if not skip_batch and opt.lambda_sam > 0:
                loss_sam = criterion_sam(output, labels)
                if torch.isnan(loss_sam) or torch.isinf(loss_sam):
                    logger.warning(
                        f"SAM loss is NaN/Inf at epoch {epoch + 1}, step {step}. Skipping batch."
                    )
                    skip_batch = True
                else:
                    loss += opt.lambda_sam * loss_sam
                    losses_sam.update(loss_sam.item())

            if not skip_batch and opt.lambda_ssim > 0:
                loss_ssim = criterion_ssim(output, labels)
                if torch.isnan(loss_ssim) or torch.isinf(loss_ssim):
                    logger.warning(
                        f"SSIM loss is NaN/Inf at epoch {epoch + 1}, step {step}. Skipping batch."
                    )
                    skip_batch = True
                else:
                    loss += opt.lambda_ssim * loss_ssim
                    losses_ssim.update(loss_ssim.item())

            if not skip_batch and opt.lambda_sid > 0:
                loss_sid = criterion_sid(output, labels)
                if torch.isnan(loss_sid) or torch.isinf(loss_sid):
                    logger.warning(
                        f"SID loss is NaN/Inf at epoch {epoch + 1}, step {step}. Skipping batch."
                    )
                    skip_batch = True
                else:
                    loss += opt.lambda_sid * loss_sid
                    losses_sid.update(loss_sid.item())

            if not skip_batch and opt.lambda_ergas > 0:
                loss_ergas = criterion_ergas(output, labels)
                if torch.isnan(loss_ergas) or torch.isinf(loss_ergas):
                    logger.warning(
                        f"ERGAS loss is NaN/Inf at epoch {epoch + 1}, step {step}. Skipping batch."
                    )
                    skip_batch = True
                else:
                    loss += opt.lambda_ergas * loss_ergas
                    losses_ergas.update(loss_ergas.item())

            if not skip_batch and opt.lambda_delta_e00 > 0:
                loss_delta_e00 = criterion_delta_e00(output, labels)
                if torch.isnan(loss_delta_e00) or torch.isinf(loss_delta_e00):
                    logger.warning(
                        f"DeltaE00 loss is NaN/Inf at epoch {epoch + 1}, step {step}. Skipping batch."
                    )
                    skip_batch = True
                else:
                    loss += opt.lambda_delta_e00 * loss_delta_e00
                    losses_delta_e00.update(loss_delta_e00.item())

            # Skip batch if any NaN detected
            if skip_batch or torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"Skipping batch at epoch {epoch + 1}, step {step} due to NaN/Inf loss."
                )
                optimizer.zero_grad()
                scheduler.step()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()

            # Update EMA weights after each optimization step
            if ema is not None:
                ema.update()

            losses_total.update(loss.item())

            pbar.set_postfix(
                {
                    "Total_Loss": f"{losses_total.avg:.6f}",
                    "L1": f"{loss_l1.item():.6f}",
                    "SAM": f"{loss_sam.item():.6f}",
                    "SSIM": f"{loss_ssim.item():.6f}",
                    "SID": f"{loss_sid.item():.6f}",
                    "ERGAS": f"{loss_ergas.item():.6f}",
                    "dE00": f"{loss_delta_e00.item():.6f}",
                    "LR": f"{lr:.6f}",
                }
            )
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
            if opt.lambda_l1 > 0:
                writer.add_scalar("Train/L1_Loss", loss_l1.item(), global_step)
            if opt.lambda_sam > 0:
                writer.add_scalar("Train/SAM_Loss", loss_sam.item(), global_step)
            if opt.lambda_ssim > 0:
                writer.add_scalar("Train/SSIM_Loss", loss_ssim.item(), global_step)
            if opt.lambda_sid > 0:
                writer.add_scalar("Train/SID_Loss", loss_sid.item(), global_step)
            if opt.lambda_ergas > 0:
                writer.add_scalar("Train/ERGAS_Loss", loss_ergas.item(), global_step)
            if opt.lambda_delta_e00 > 0:
                writer.add_scalar(
                    "Train/DeltaE00_Loss", loss_delta_e00.item(), global_step
                )

        # Apply EMA weights for validation (smoother, more stable metrics)
        if ema is not None:
            ema.apply_shadow()

        (
            mrae_loss,
            rmse_loss,
            psnr_loss,
            sam_loss,
            sid_loss,
            ergas_loss,
            ssim_loss,
            delta_e00_loss,
        ) = validate(val_loader, model)

        # Restore original weights after validation
        if ema is not None:
            ema.restore()

        # Save model
        if psnr_loss > best_psnr:
            best_psnr = psnr_loss
            logger.info(f"Saving to {opt.outf}")
            save_checkpoint(
                opt.outf,
                epoch + 1,
                step,
                model,
                optimizer,
                psnr_loss,
                scheduler,
                ema,
                scheduler_config,
            )

        # Log detailed metrics
        log_msg = f"Epoch[{epoch + 1}] ===> Avg. Loss: {losses_total.avg:.6f}, -> Val PSNR: {psnr_loss:.6f} <-, "
        log_msg += f"Val MRAE: {mrae_loss:.6f}, Val RMSE: {rmse_loss:.6f}, "
        log_msg += f"Val SAM: {sam_loss:.6f}, Val SID: {sid_loss:.6f}, Val SSIM: {ssim_loss:.6f} "
        log_msg += f"Val ERGAS: {ergas_loss:.6f}, Val dE00: {delta_e00_loss:.6f}"
        logger.info(log_msg)

        writer.add_scalar("Val/MRAE", mrae_loss, epoch)
        writer.add_scalar("Val/RMSE", rmse_loss, epoch)
        writer.add_scalar("Val/PSNR", psnr_loss, epoch)
        writer.add_scalar("Val/SAM", sam_loss, epoch)
        writer.add_scalar("Val/SID", sid_loss, epoch)
        writer.add_scalar("Val/SSIM", ssim_loss, epoch)
        writer.add_scalar("Val/ERGAS", ergas_loss, epoch)
        writer.add_scalar("Val/DeltaE00", delta_e00_loss, epoch)

    writer.close()
    return 0


# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_sid = AverageMeter()
    losses_ergas = AverageMeter()
    losses_ssim = AverageMeter()
    losses_delta_e00 = AverageMeter()

    # Calculate border crop based on upscale factor
    border_crop = 128 * opt.upscale_factor
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            # Crop borders to avoid edge effects
            if (
                border_crop > 0
                and output.shape[2] > 2 * border_crop
                and output.shape[3] > 2 * border_crop
            ):
                output_cropped = output[
                    :, :, border_crop:-border_crop, border_crop:-border_crop
                ]
                target_cropped = target[
                    :, :, border_crop:-border_crop, border_crop:-border_crop
                ]
            else:
                output_cropped = output
                target_cropped = target

            loss_mrae = criterion_mrae(output_cropped, target_cropped)
            loss_rmse = criterion_rmse(output_cropped, target_cropped)
            loss_psnr = criterion_psnr(output_cropped, target_cropped)
            loss_sam = criterion_sam(output_cropped, target_cropped)
            loss_sid = criterion_sid(output_cropped, target_cropped)
            ssim_value = 1 - criterion_ssim(output_cropped, target_cropped)
            loss_ergas = criterion_ergas(output_cropped, target_cropped)
            loss_delta_e00 = criterion_delta_e00(output_cropped, target_cropped)

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)
        losses_sid.update(loss_sid.data)
        losses_ssim.update(ssim_value.data)
        losses_ergas.update(loss_ergas.data)
        losses_delta_e00.update(loss_delta_e00.data)

    return (
        losses_mrae.avg,
        losses_rmse.avg,
        losses_psnr.avg,
        losses_sam.avg,
        losses_sid.avg,
        losses_ergas.avg,
        losses_ssim.avg,
        losses_delta_e00.avg,
    )


if __name__ == "__main__":
    main()
    # print(torch.__version__)
