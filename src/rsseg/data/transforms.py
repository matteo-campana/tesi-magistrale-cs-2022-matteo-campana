# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data augmentation.

* :class:`JointAugment` — geometric augmentation applied identically to image and mask
  (aerial images have no canonical orientation, so flips and 90° rotations are label-preserving).
* :class:`MultiCropAugment` — DINO multi-crop (2 global + N local views), optionally with the
  milder "aerial" colour policy.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T


class JointAugment:
    def __init__(self, crop_size: int | None = None, rotate90: bool = True):
        self.crop_size = crop_size
        self.rotate90 = rotate90

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_size is not None and image.shape[0] > self.crop_size:
            h, w = mask.shape
            y = random.randint(0, h - self.crop_size)
            x = random.randint(0, w - self.crop_size)
            image = image[y : y + self.crop_size, x : x + self.crop_size]
            mask = mask[y : y + self.crop_size, x : x + self.crop_size]
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        if self.rotate90:
            k = random.randint(0, 3)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        return image, mask


class _GaussianBlur:
    def __init__(self, p: float, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p, self.radius_min, self.radius_max = p, radius_min, radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


class _Solarize:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.solarize(img) if random.random() < self.p else img


class _Rotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.rotate(90 * random.randint(0, 3))


class MultiCropAugment:
    """DINO multi-crop (Caron et al., 2021). Outputs channels-last float32 arrays in ``[0, 255]``."""

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        local_crops_number: int = 6,
        global_scale: tuple[float, float] = (0.14, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.4),
        aerial: bool = False,
    ):
        if aerial:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
            geometric = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), _Rotate90()]
            p_gray, p_solarize = 0.1, 0.0
        else:
            jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            geometric = [T.RandomHorizontalFlip()]
            p_gray, p_solarize = 0.2, 0.2
        flip_and_color = [*geometric, T.RandomApply([jitter], p=0.8), T.RandomGrayscale(p=p_gray)]
        bicubic = T.InterpolationMode.BICUBIC
        self.global1 = T.Compose(
            [T.RandomResizedCrop(global_size, scale=tuple(global_scale), interpolation=bicubic),
             *flip_and_color, _GaussianBlur(1.0)]
        )  # fmt: skip
        self.global2 = T.Compose(
            [T.RandomResizedCrop(global_size, scale=tuple(global_scale), interpolation=bicubic),
             *flip_and_color, _GaussianBlur(0.1), _Solarize(p_solarize)]
        )  # fmt: skip
        self.local = T.Compose(
            [T.RandomResizedCrop(local_size, scale=tuple(local_scale), interpolation=bicubic),
             *flip_and_color, _GaussianBlur(0.5)]
        )  # fmt: skip
        self.local_crops_number = local_crops_number
        self.local_size = local_size

    def __call__(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        globals_ = np.stack([_to_array(self.global1(image)), _to_array(self.global2(image))])
        if self.local_crops_number:
            locals_ = np.stack(
                [_to_array(self.local(image)) for _ in range(self.local_crops_number)]
            )
        else:
            locals_ = np.zeros((0, self.local_size, self.local_size, 3), dtype=np.float32)
        return globals_, locals_


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)
