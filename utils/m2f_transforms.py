import random
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

class M2FCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, dem, mask, edge):
        for t in self.transforms:
            img, dem, mask, edge = t(img, dem, mask, edge)
        return img, dem, mask, edge

class M2FRandomScale(object):
    def __init__(self, scale_range):
        self.scale_range = scale_range
    def __call__(self, img, dem, mask, edge):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(img.size[1] * scale), int(img.size[0] * scale))
        return (
            F.resize(img, target_size, InterpolationMode.BILINEAR),
            F.resize(dem, target_size, InterpolationMode.BILINEAR),
            F.resize(mask, target_size, InterpolationMode.NEAREST),
            F.resize(edge, target_size, InterpolationMode.NEAREST),
        )

class M2FRandomCrop(object):
    def __init__(self, size, pad_if_needed=True):
        self.size = (int(size), int(size))
        self.pad_if_needed = pad_if_needed

    def __call__(self, img, dem, mask, edge):
        if self.pad_if_needed:
            # Pad width
            if img.size[0] < self.size[1]:
                pad_w = int((1 + self.size[1] - img.size[0]) / 2)
                img = F.pad(img, pad_w); dem = F.pad(dem, pad_w)
                mask = F.pad(mask, pad_w); edge = F.pad(edge, pad_w)
            # Pad height
            if img.size[1] < self.size[0]:
                pad_h = int((1 + self.size[0] - img.size[1]) / 2)
                img = F.pad(img, pad_h); dem = F.pad(dem, pad_h)
                mask = F.pad(mask, pad_h); edge = F.pad(edge, pad_h)

        w, h = img.size
        th, tw = self.size
        i = random.randint(0, h - th) if h > th else 0
        j = random.randint(0, w - tw) if w > tw else 0

        return (
            F.crop(img, i, j, th, tw),
            F.crop(dem, i, j, th, tw),
            F.crop(mask, i, j, th, tw),
            F.crop(edge, i, j, th, tw)
        )

class M2FRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, dem, mask, edge):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(dem), F.hflip(mask), F.hflip(edge)
        return img, dem, mask, edge

class M2FToTensor(object):
    def __call__(self, img, dem, mask, edge):
        # 1. 处理 Mask (主任务分割标签)
        mask_np = np.array(mask, dtype=np.int64)
        # 【核心修复】：将所有大于 0 的像素值（比如 255 纯白，或 128 灰阶伪影）全部强制转为 1
        # 这样标签就严格只包含 0 (背景) 和 1 (滑坡)
        mask_np = np.where(mask_np > 0, 1, 0).astype(np.int64)

        # 2. 处理 Edge (辅任务边界标签)
        edge_np = np.array(edge, dtype=np.float32)
        # 边界标签用于 BCE Loss，要求输入值为 float 类型的 0.0 和 1.0
        edge_np = np.where(edge_np > 0, 1.0, 0.0).astype(np.float32)

        return (
            F.to_tensor(img),
            F.to_tensor(dem),
            torch.as_tensor(mask_np),
            torch.as_tensor(edge_np).unsqueeze(0) # [1, H, W]
        )

class M2FNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, img, dem, mask, edge):
        # 仅对 RGB 进行 ImageNet 标准化，DEM 数据我们假设预处理时已经归一化到了 [0, 1]
        return F.normalize(img, self.mean, self.std), dem, mask, edge
    
    
class M2FColorJitter(object):
    """
    针对 M2F 架构的强力光学畸变 (对齐 mmseg 的 PhotoMetricDistortion)
    注意：只对 RGB 影像进行颜色抖动，DEM、Mask 和 Edge 绝对不能变！
    """
    def __init__(self, brightness=0.32, contrast=0.5, saturation=0.5, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, dem, mask, edge):
        img = self.jitter(img)  # 仅让 RGB 图片产生光照、颜色扰动
        return img, dem, mask, edge