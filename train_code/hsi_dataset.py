from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py


def _ensure_cube_shape(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape}")
    if cube.shape[0] in (31, 61, 62, 64) and cube.shape[0] < cube.shape[-1]:
        cube = np.transpose(cube, (1, 2, 0))
    return cube


def _resize_cube_to_match(cube: np.ndarray, height: int, width: int) -> np.ndarray:
    if cube.shape[1] == height and cube.shape[2] == width:
        return cube
    resized = np.empty((cube.shape[0], height, width), dtype=np.float32)
    for band_idx in range(cube.shape[0]):
        resized[band_idx] = cv2.resize(
            cube[band_idx], (width, height), interpolation=cv2.INTER_AREA
        )
    return resized

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8, upscale_factor=1, isTest=False):
        self.crop_size = crop_size
        self.arg = arg
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.hypers = []
        self.bgrs = []

        root = Path(data_root)
        split_file = root / "split_txt" / "train_list.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing train split file: {split_file}")

        with split_file.open("r") as fin:
            sample_ids = [line.strip() for line in fin if line.strip()]
        if not sample_ids:
            raise ValueError(f"No entries found in {split_file}")

        hyper_dir = root / "train" / "hsi_61"
        bgr_dir = root / "train" / "rgb_2"
        for idx, scene_id in enumerate(sample_ids):
            if isTest and idx > 3:
                break
            hyper_path = hyper_dir / f"{scene_id}.h5"
            bgr_path = bgr_dir / f"{scene_id}.png"

            if not hyper_path.exists():
                raise FileNotFoundError(f"Missing hyperspectral file: {hyper_path}")
            if not bgr_path.exists():
                raise FileNotFoundError(f"Missing RGB file: {bgr_path}")

            with h5py.File(hyper_path, "r") as mat:
                hyper = np.array(mat["cube"], dtype=np.float32)
            hyper = _ensure_cube_shape(hyper)
            hyper = np.transpose(hyper, (2, 0, 1))  # C,H,W

            bgr = cv2.imread(str(bgr_path))
            if bgr is None:
                raise ValueError(f"Fail to load RGB image: {bgr_path}")
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = bgr / 255.0
            bgr = np.transpose(bgr, (2, 0, 1))

            # Only resize if not doing super-resolution (upscale_factor=1)
            if upscale_factor == 1:
                hyper = _resize_cube_to_match(hyper, bgr.shape[1], bgr.shape[2])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f"Loaded training scene {scene_id} ({idx + 1}/{len(sample_ids)})")

        if not self.hypers:
            raise ValueError("No training samples were loaded.")

        _, h, w = self.bgrs[0].shape  # Use BGR size for patch calculation
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        if self.patch_per_line <= 0 or self.patch_per_colum <= 0:
            raise ValueError(
                f"crop_size={crop_size} and stride={stride} exceed image size {(h, w)}"
            )
        self.patch_per_img = self.patch_per_line * self.patch_per_colum
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]

        # Crop BGR with crop_size
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]

        # Crop hyper with crop_size * upscale_factor
        hyper_crop_size = crop_size * self.upscale_factor
        hyper_stride = stride * self.upscale_factor
        hyper = hyper[:, h_idx * hyper_stride:h_idx * hyper_stride + hyper_crop_size,
                      w_idx * hyper_stride:w_idx * hyper_stride + hyper_crop_size]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True, upscale_factor=1, isTest=False):
        self.hypers = []
        self.bgrs = []
        self.upscale_factor = upscale_factor
        root = Path(data_root)
        split_file = root / "split_txt" / "valid_list.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing valid split file: {split_file}")

        with split_file.open("r") as fin:
            sample_ids = [line.strip() for line in fin if line.strip()]
        if not sample_ids:
            raise ValueError(f"No entries found in {split_file}")

        hyper_dir = root / "test-public" / "hsi_61"
        bgr_dir = root / "test-public" / "rgb_2"

        for idx, scene_id in enumerate(sample_ids):
            if isTest and idx > 3:
                break
            hyper_path = hyper_dir / f"{scene_id}.h5"
            bgr_path = bgr_dir / f"{scene_id}.png"

            if not hyper_path.exists():
                raise FileNotFoundError(f"Missing hyperspectral file: {hyper_path}")
            if not bgr_path.exists():
                raise FileNotFoundError(f"Missing RGB file: {bgr_path}")

            with h5py.File(hyper_path, "r") as mat:
                hyper = np.array(mat["cube"], dtype=np.float32)
            hyper = _ensure_cube_shape(hyper)
            hyper = np.transpose(hyper, (2, 0, 1))

            bgr = cv2.imread(str(bgr_path))
            if bgr is None:
                raise ValueError(f"Fail to load RGB image: {bgr_path}")
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = bgr / 255.0
            bgr = np.transpose(bgr, (2, 0, 1))

            # Only resize if not doing super-resolution (upscale_factor=1)
            if upscale_factor == 1:
                hyper = _resize_cube_to_match(hyper, bgr.shape[1], bgr.shape[2])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f"Loaded validation scene {scene_id} ({idx + 1}/{len(sample_ids)})")

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)
