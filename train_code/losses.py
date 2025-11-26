import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class SSIMLoss(nn.Module):
    """
    Implementation of SSIM Loss.
    Details can be found in:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.
    IEEE transactions on image processing, 13(4), 600-612.
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # The loss is 1 - SSIM
        return 1 - self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


class SIDLoss(nn.Module):
    """
    Spectral Information Divergence Loss
    """

    def __init__(self, eps=1e-6):
        super(SIDLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: (batch, channels, height, width)
        # Add a clamp to ensure non-negative values and avoid extreme values
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        target = torch.clamp(target, min=self.eps, max=1.0)

        # Normalize to get probability distributions for each pixel's spectrum
        pred_sum = torch.sum(pred, dim=1, keepdim=True)
        target_sum = torch.sum(target, dim=1, keepdim=True)

        pred_norm = pred / pred_sum
        target_norm = target / target_sum

        # Compute KL divergence
        # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P) - P * log(Q))

        log_pred = torch.log(pred_norm)
        log_target = torch.log(target_norm)

        kl_pred_target = torch.sum(pred_norm * (log_pred - log_target), dim=1)
        kl_target_pred = torch.sum(target_norm * (log_target - log_pred), dim=1)

        sid = kl_pred_target + kl_target_pred

        return torch.mean(sid)


class ERGASLoss(nn.Module):
    """
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) Loss
    """

    def __init__(self, upscale_factor=1, eps=1e-4):
        super(ERGASLoss, self).__init__()
        self.upscale_factor = upscale_factor
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: (batch, channels, height, width)
        pred = torch.clamp(pred, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)

        batch_size, channels, _, _ = pred.shape

        # Reshape for easier calculation per band
        pred_reshaped = pred.reshape(batch_size, channels, -1)
        target_reshaped = target.reshape(batch_size, channels, -1)

        # MSE per band
        mse_per_band = torch.mean(
            (pred_reshaped - target_reshaped).pow(2), dim=2
        )  # shape: (batch, channels)

        # Mean of target per band
        mean_target_per_band = torch.mean(
            target_reshaped, dim=2
        )  # shape: (batch, channels)

        # Ratio of MSE to squared mean
        # Clamp denominator to avoid gradient explosion when target is close to 0
        denom = mean_target_per_band.pow(2)
        denom = torch.clamp(denom, min=self.eps)

        ratio = mse_per_band / denom

        # Mean over bands
        mean_ratio = torch.mean(ratio, dim=1)  # shape: (batch)

        # ERGAS
        ergas = 100 * self.upscale_factor * torch.sqrt(mean_ratio + 1e-8)

        return torch.mean(ergas)


class DeltaE00Loss(nn.Module):
    """
    CIEDE2000 Delta E Loss function.
    This loss module encapsulates the entire differentiable pipeline from
    Hyperspectral Imaging (HSI) to CIELAB color space and then calculates the
    CIEDE2000 color difference.
    It assumes the input HSI data has 61 channels from 400nm to 1000nm with 10nm steps.
    """

    def __init__(self, eps=1e-8):
        super(DeltaE00Loss, self).__init__()
        self.eps = eps

        # CIE 1931 2-degree CMFs for 61 bands (400-1000nm, 10nm step)
        # Corrected values sourced from CVRL/Standard Tables
        cmfs_data = [
            [0.01431, 0.000396, 0.06785],  # 400nm
            [0.04351, 0.00121, 0.20740],  # 410nm
            [0.13438, 0.00400, 0.64560],  # 420nm
            [0.28390, 0.01160, 1.38560],  # 430nm
            [0.34828, 0.02300, 1.74706],  # 440nm
            [0.33620, 0.03800, 1.77211],  # 450nm
            [0.29080, 0.06000, 1.66920],  # 460nm
            [0.19536, 0.09098, 1.28764],  # 470nm
            [0.09564, 0.13902, 0.81295],  # 480nm
            [0.03201, 0.20802, 0.46518],  # 490nm
            [0.00490, 0.32300, 0.27200],  # 500nm
            [0.00930, 0.50300, 0.15820],  # 510nm
            [0.06327, 0.71000, 0.07825],  # 520nm
            [0.16550, 0.86200, 0.04216],  # 530nm
            [0.29040, 0.95400, 0.02030],  # 540nm
            [0.43345, 0.99495, 0.00875],  # 550nm
            [0.59450, 0.99500, 0.00390],  # 560nm
            [0.76210, 0.95200, 0.00210],  # 570nm
            [0.91630, 0.87000, 0.00165],  # 580nm
            [1.02630, 0.75700, 0.00110],  # 590nm
            [1.06220, 0.63100, 0.00080],  # 600nm
            [1.00260, 0.50300, 0.00034],  # 610nm
            [0.85445, 0.38100, 0.00019],  # 620nm
            [0.64240, 0.26500, 0.00005],  # 630nm
            [0.44790, 0.17500, 0.00002],  # 640nm
            [0.28350, 0.10700, 0.00000],  # 650nm
            [0.16490, 0.06100, 0.00000],  # 660nm
            [0.08740, 0.03200, 0.00000],  # 670nm
            [0.04677, 0.01700, 0.00000],  # 680nm
            [0.02270, 0.00821, 0.00000],  # 690nm
            [0.01136, 0.00410, 0.00000],  # 700nm
        ]
        # Pad with 30 rows of zeros for 710nm to 1000nm
        for _ in range(30):
            cmfs_data.append([0.0, 0.0, 0.0])

        # CIE Standard Illuminant D65 relative spectral power distribution (400-700nm, 10nm steps)
        d65_data = [
            82.75,
            91.49,
            93.43,
            86.68,
            104.86,
            117.01,
            117.81,
            114.86,
            115.92,
            108.81,
            109.35,
            107.80,
            104.79,
            107.69,
            104.41,
            104.05,
            100.00,
            96.33,
            95.79,
            88.69,
            90.01,
            89.60,
            87.70,
            83.29,
            83.70,
            80.03,
            80.21,
            82.28,
            78.28,
            69.72,
            71.61,
        ]
        # Pad D65 with zeros for 710-1000nm to match CMFs length (61)
        d65_data += [0.0] * 30

        cmfs = torch.tensor(cmfs_data, dtype=torch.float32)
        d65 = torch.tensor(d65_data, dtype=torch.float32).unsqueeze(1)  # Shape: (61, 1)

        # Integrate: XYZ_white = Sum(D65 * CMF)
        # We want Y_white = 100 for perfect white reflectance (1.0)
        base_xyz = torch.sum(cmfs * d65, dim=0)  # (3,)
        base_y = base_xyz[1]

        # Normalization factor k
        k = 100.0 / (base_y + self.eps)

        # Pre-compute weighted CMFs: k * D65 * CMF
        weighted_cmfs = k * d65 * cmfs

        self.register_buffer("cmfs_const", weighted_cmfs)

        # D65 standard illuminant, for XYZ to Lab conversion reference white
        # Matches the target white point of the integration
        ref_white = torch.tensor([95.047, 100.000, 108.883], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.register_buffer("ref_white_const", ref_white)

        # Conversion matrices
        xyz_to_srgb_mat = torch.tensor(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("xyz_to_srgb_mat_const", xyz_to_srgb_mat)

    def _hsi_to_xyz(self, hsi):
        # hsi shape: (B, C, H, W), where C=61
        # cmfs shape: (C, 3)
        xyz = torch.einsum("bchw,cd->bdhw", hsi, self.cmfs_const)
        return xyz

    def _xyz_to_srgb(self, xyz):
        # xyz shape: (B, 3, H, W)
        # Apply matrix transformation
        srgb_linear = torch.einsum("bdhw,de->behw", xyz, self.xyz_to_srgb_mat_const)

        # Normalize to [0, 1] by dividing by 100 (since XYZ is scaled to Y=100)
        srgb_linear = srgb_linear / 100.0

        # Apply gamma correction
        srgb = torch.where(
            srgb_linear <= 0.0031308,
            12.92 * srgb_linear,
            1.055 * torch.pow(srgb_linear.clamp(min=self.eps), 1 / 2.4) - 0.055,
        )
        return srgb.clamp(0.0, 1.0)

    def _xyz_to_lab(self, xyz):
        # xyz shape: (B, 3, H, W)
        xyz_ref = xyz / self.ref_white_const

        # Non-linear transformation
        f_xyz = torch.where(
            xyz_ref > 0.008856,
            torch.pow(xyz_ref.clamp(min=self.eps), 1 / 3),
            7.787 * xyz_ref + 16 / 116,
        )

        fx, fy, fz = f_xyz[:, 0, :, :], f_xyz[:, 1, :, :], f_xyz[:, 2, :, :]

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return torch.stack([L, a, b], dim=1)

    def _delta_e_00(self, lab1, lab2):
        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

        C1 = torch.sqrt(a1**2 + b1**2 + 1e-8)
        C2 = torch.sqrt(a2**2 + b2**2 + 1e-8)
        C_bar = (C1 + C2) / 2

        G = 0.5 * (1 - torch.sqrt(C_bar**7 / (C_bar**7 + 25**7 + 1e-8)))

        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2

        C1_prime = torch.sqrt(a1_prime**2 + b1**2 + 1e-8)
        C2_prime = torch.sqrt(a2_prime**2 + b2**2 + 1e-8)

        h1_prime = torch.atan2(b1, a1_prime)
        h2_prime = torch.atan2(b2, a2_prime)

        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        delta_h_prime = h2_prime - h1_prime
        # Avoid in-place operation for gradient safety
        mask_delta = delta_h_prime.abs() > torch.pi
        adjustment_delta = torch.sign(delta_h_prime) * 2 * torch.pi
        delta_h_prime = torch.where(
            mask_delta, delta_h_prime - adjustment_delta, delta_h_prime
        )

        delta_H_prime = (
            2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(delta_h_prime / 2)
        )

        L_bar_prime = (L1 + L2) / 2
        C_bar_prime = (C1_prime + C2_prime) / 2

        h_bar_prime = (h1_prime + h2_prime) / 2
        # Avoid in-place operation
        mask_bar = (h1_prime.abs() > torch.pi) & (h2_prime.abs() > torch.pi)
        h_bar_prime = torch.where(mask_bar, h_bar_prime + torch.pi, h_bar_prime)

        T = (
            1
            - 0.17 * torch.cos(h_bar_prime - torch.deg2rad(torch.tensor(30.0)))
            + 0.24 * torch.cos(2 * h_bar_prime)
            + 0.32 * torch.cos(3 * h_bar_prime + torch.deg2rad(torch.tensor(6.0)))
            - 0.20 * torch.cos(4 * h_bar_prime - torch.deg2rad(torch.tensor(63.0)))
        )

        S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / torch.sqrt(
            20 + (L_bar_prime - 50) ** 2
        )
        S_C = 1 + 0.045 * C_bar_prime
        S_H = 1 + 0.015 * C_bar_prime * T

        k_L, k_C, k_H = 1, 1, 1

        delta_E = torch.sqrt(
            (delta_L_prime / (k_L * S_L)) ** 2
            + (delta_C_prime / (k_C * S_C)) ** 2
            + (delta_H_prime / (k_H * S_H)) ** 2
            + 1e-8
        )

        return delta_E

    def forward(self, pred_hsi, target_hsi):
        pred_hsi = torch.clamp(pred_hsi, min=0.0, max=1.0)
        target_hsi = torch.clamp(target_hsi, min=0.0, max=1.0)

        # Convert HSI to XYZ
        pred_xyz = self._hsi_to_xyz(pred_hsi)
        target_xyz = self._hsi_to_xyz(target_hsi)

        # Convert XYZ to Lab
        pred_lab = self._xyz_to_lab(pred_xyz)
        target_lab = self._xyz_to_lab(target_xyz)

        # Calculate DeltaE00
        delta_e = self._delta_e_00(pred_lab, target_lab)
        return delta_e.mean()
