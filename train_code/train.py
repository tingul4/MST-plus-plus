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
)
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--method", type=str, default="mst_plus_plus")
parser.add_argument("--pretrained_model_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
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
)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(
    data_root=opt.data_root, bgr2rgb=True, upscale_factor=opt.upscale_factor
)
print("Validation set samples: ", len(val_data))

# loss function
# Use L1 loss for training stability (MRAE unstable with values near zero)
criterion_train = nn.L1Loss()
criterion_mrae = Loss_MRAE_custom()  # Only for validation
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

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
    criterion_train.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# logging
log_dir = os.path.join(opt.outf, "train.log")
logger = initialize_logger(log_dir)


def main():
    cudnn.benchmark = True
    best_psnr = 0
    writer = SummaryWriter(log_dir=os.path.join(opt.outf, "tensorboard"))

    # Create data loaders once outside epoch loop
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
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
        losses = AverageMeter()

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
            loss = criterion_mrae(output, labels)  # Use MRAE for training
            # loss = criterion_train(output, labels)  # Use L1 loss for training
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)

            pbar.set_postfix(
                {
                    "Step": f"{step + 1}/{len(train_loader)}",
                    "Loss": f"{losses.avg:.6f}",
                    "LR": f"{lr:.6f}",
                }
            )
            writer.add_scalar("Train/MRAE", loss, step)
            writer.add_scalar("Train/Loss", losses.avg, step)

        mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)

        # Save model
        if psnr_loss > best_psnr:
            logger.info(f"Saving to {opt.outf}")
            save_checkpoint(opt.outf, epoch + 1, step, model, optimizer, psnr_loss)

        # Log detailed metrics
        logger.info(
            f"Epoch[{epoch + 1}] =======> Avg. Loss: {losses.avg:.6f}, Val MRAE: {mrae_loss:.6f}, Val RMSE: {rmse_loss:.6f}, Val PSNR: {psnr_loss:.6f}"
        )
        writer.add_scalar("Val/RMSE", rmse_loss, epoch)
        writer.add_scalar("Val/PSNR", psnr_loss, epoch)

    writer.close()
    return 0


# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
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
                loss_mrae = criterion_mrae(
                    output[:, :, border_crop:-border_crop, border_crop:-border_crop],
                    target[:, :, border_crop:-border_crop, border_crop:-border_crop],
                )
                loss_rmse = criterion_rmse(
                    output[:, :, border_crop:-border_crop, border_crop:-border_crop],
                    target[:, :, border_crop:-border_crop, border_crop:-border_crop],
                )
                loss_psnr = criterion_psnr(
                    output[:, :, border_crop:-border_crop, border_crop:-border_crop],
                    target[:, :, border_crop:-border_crop, border_crop:-border_crop],
                )
            else:
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


if __name__ == "__main__":
    main()
    # print(torch.__version__)
