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
    """
    对齐 mmseg 的强力尺度抖动：先放大到 Base Size，再进行随机比例缩放。
    彻底消灭黑边 (Padding)，让网络看清滑坡纹理！
    """
    def __init__(self, scale_range=(0.8, 1.2), base_size=(320, 320)):
        self.scale_range = scale_range
        self.base_size = base_size # 对应 mmseg 的 scale=(320, 320)

    def __call__(self, img, dem, mask, edge):
        # 1. 先把短边或长边强行拉伸到 base_size (320x320) 的基准线上
        # (这里采用直接 resize 到 320x320，简单粗暴且对齐 mmseg 的逻辑)
        img = F.resize(img, self.base_size, InterpolationMode.BILINEAR)
        dem = F.resize(dem, self.base_size, InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.base_size, InterpolationMode.NEAREST)
        edge = F.resize(edge, self.base_size, InterpolationMode.NEAREST)

        # 2. 在 320x320 的基础上，进行 0.8 ~ 1.2 的随机抖动
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(self.base_size[0] * scale), int(self.base_size[1] * scale))
        
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
        # 统一计算需要 padding 的宽高
        pad_w = max(0, self.size[1] - img.size[0])
        pad_h = max(0, self.size[0] - img.size[1])
        
        if self.pad_if_needed and (pad_w > 0 or pad_h > 0):
            # torchvision F.pad 顺序为 (left, top, right, bottom)
            padding = (0, 0, pad_w, pad_h)
            img = F.pad(img, padding, fill=0)
            dem = F.pad(dem, padding, fill=0)
            # 【核心修复 1】：标签必须用 255 填充 (ignore_index)，让网络忽略黑边
            mask = F.pad(mask, padding, fill=255) 
            edge = F.pad(edge, padding, fill=255) 

        i, j, h, w = T.RandomCrop.get_params(img, output_size=self.size)
        return (
            F.crop(img, i, j, h, w),
            F.crop(dem, i, j, h, w),
            F.crop(mask, i, j, h, w),
            F.crop(edge, i, j, h, w)
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
        # 2. 处理 Edge (辅任务边界标签)
        edge_np = np.array(edge, dtype=np.float32)
        
        # ================= 核心修复 1：完美的 Mask 二值化 =================
        # 逻辑：
        # 1. mask_np == 0 的像素，保持为 0 (背景)
        # 2. mask_np == 255 的像素，保持为 255 (Crop 产生的忽略黑边)
        # 3. 所有介于 0 和 255 之间的脏像素/滑坡像素，全部强制转为 1 (滑坡)
        mask_np = np.where((mask_np > 0) & (mask_np != 255), 1, mask_np)

        # ================= 核心修复 2：边缘标签的安全处理 =================
        # BCE Loss 函数不能接收 255 这种数值，会导致计算炸出 NaN (无限大)
        # 所以我们把边缘上的 255 黑边强制变回 0 (反正主分支已经忽略了黑边的 Loss，边缘随便预测个0无伤大雅)
        edge_np = np.where(edge_np == 255, 0.0, edge_np)
        edge_np = np.where(edge_np > 0, 1.0, 0.0).astype(np.float32)

        return (
            F.to_tensor(img),
            F.to_tensor(dem),  # 直接用原版的，因为你的 prepare_dem.py 已经做得很完美了！
            torch.as_tensor(mask_np),
            torch.as_tensor(edge_np).unsqueeze(0) # [1, H, W]
        )

class M2FNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, dem, mask, edge):
        # 对 RGB 进行标准化
        img = F.normalize(img, mean=self.mean, std=self.std)

        # 【新增】：对 DEM 也进行完全相同的 ImageNet 标准化！
        # 因为 DEM 骨干加载了 ImageNet 权重，它必须吃这种分布的数据
        dem = F.normalize(dem, mean=self.mean, std=self.std)

        return img, dem, mask, edge
    
    
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