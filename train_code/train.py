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

# loss function
from losses import SSIMLoss, SIDLoss, ERGASLoss, DeltaE00Loss

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--method", type=str, default="mst_plus_plus")
parser.add_argument("--pretrained_model_path", type=str, default=None)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader), eta_min=1e-6
    )

    # Resume
    last_epoch = 0
    resume_file = opt.pretrained_model_path
    if resume_file is not None:
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_epoch = checkpoint["epoch"]
            logger.info(
                f"=> loading checkpoint '{resume_file}' from epoch {checkpoint['epoch']}"
            )

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

            if opt.lambda_l1 > 0:
                loss_l1 = criterion_l1(output, labels)
                if torch.isnan(loss_l1):
                    raise ValueError(
                        f"L1 loss is NaN at epoch {epoch + 1}, step {step}"
                    )
                loss += opt.lambda_l1 * loss_l1
                losses_l1.update(loss_l1.item())

            if opt.lambda_sam > 0:
                loss_sam = criterion_sam(output, labels)
                if torch.isnan(loss_sam):
                    raise ValueError(
                        f"SAM loss is NaN at epoch {epoch + 1}, step {step}"
                    )
                loss += opt.lambda_sam * loss_sam
                losses_sam.update(loss_sam.item())

            if opt.lambda_ssim > 0:
                loss_ssim = criterion_ssim(output, labels)
                if torch.isnan(loss_ssim):
                    raise ValueError(
                        f"SSIM loss is NaN at epoch {epoch + 1}, step {step}"
                    )
                loss += opt.lambda_ssim * loss_ssim
                losses_ssim.update(loss_ssim.item())

            if opt.lambda_sid > 0:
                loss_sid = criterion_sid(output, labels)
                if torch.isnan(loss_sid):
                    raise ValueError(
                        f"SID loss is NaN at epoch {epoch + 1}, step {step}"
                    )
                loss += opt.lambda_sid * loss_sid
                losses_sid.update(loss_sid.item())

            if opt.lambda_ergas > 0:
                loss_ergas = criterion_ergas(output, labels)
                if torch.isnan(loss_ergas):
                    raise ValueError(
                        f"ERGAS loss is NaN at epoch {epoch + 1}, step {step}"
                    )
                loss += opt.lambda_ergas * loss_ergas
                losses_ergas.update(loss_ergas.item())

            if opt.lambda_delta_e00 > 0:
                loss_delta_e00 = criterion_delta_e00(output, labels)
                if torch.isnan(loss_delta_e00):
                    raise ValueError(
                        f"DeltaE00 loss is NaN at epoch {epoch + 1}, step {step}"
                    )
                loss += opt.lambda_delta_e00 * loss_delta_e00
                losses_delta_e00.update(loss_delta_e00.item())

            if torch.isnan(loss):
                raise ValueError(f"Total loss is NaN at epoch {epoch + 1}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
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

        # Save model
        if psnr_loss > best_psnr:
            best_psnr = psnr_loss
            logger.info(f"Saving to {opt.outf}")
            save_checkpoint(opt.outf, epoch + 1, step, model, optimizer, psnr_loss)

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
