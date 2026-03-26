import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import torch.utils.data as data


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        cmap[i] = np.array([r, g, b])

    return (cmap / 255.0) if normalized else cmap


class VOCSegmentation(data.Dataset):
    """VOC2012 风格语义分割数据集（更健壮：自动匹配多种图片/mask后缀）"""

    cmap = voc_cmap()

    # 你可以按需增删扩展名；顺序=优先级（越靠前越优先）
    IMG_EXTS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    MASK_EXTS: List[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform=None,
    ):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform

        # --------- 1) 定位 voc_root（支持两种 root 写法）---------
        # 写法 A：root/VOCdevkit/VOC2012
        voc2012_a = os.path.join(self.root, "VOC2012")
        # 写法 B：root 直接就是 VOC2012
        voc2012_b = self.root

        if os.path.isdir(os.path.join(voc2012_a, "JPEGImages")):
            voc_root = voc2012_a
        elif os.path.isdir(os.path.join(voc2012_b, "JPEGImages")):
            voc_root = voc2012_b
        else:
            raise RuntimeError(
                "找不到 VOC2012 风格数据集目录。\n"
                "请检查 data_root 是否满足以下任一结构：\n"
                "A) data_root/VOCdevkit/VOC2012/JPEGImages\n"
                "B) data_root/JPEGImages\n"
                f"当前给定 root = {self.root}"
            )

        self.voc_root = voc_root
        self.image_dir = os.path.join(voc_root, "JPEGImages")
        self.mask_dir = os.path.join(voc_root, "SegmentationClass")
        self.splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
        split_f = os.path.join(self.splits_dir, f"{image_set}.txt")

        # --------- 2) 检查必要文件是否存在 ---------
        if not os.path.isdir(self.image_dir):
            raise RuntimeError(f"JPEGImages 不存在：{self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise RuntimeError(f"SegmentationClass 不存在：{self.mask_dir}")
        if not os.path.isfile(split_f):
            raise ValueError(
                f"找不到 split 文件：{split_f}\n"
                "image_set 必须是 train / val / trainval 之一，且你需要准备对应的 txt 列表。"
            )

        # --------- 3) 读取 split 列表（允许写 'xxx' 或 'xxx.jpg'，统一取 stem）---------
        with open(split_f, "r", encoding="utf-8") as f:
            stems = [Path(x.strip()).stem for x in f.readlines() if x.strip()]

        if len(stems) == 0:
            raise RuntimeError(f"split 文件为空：{split_f}")

        # --------- 4) 自动匹配图片/掩膜文件（多后缀）---------
        self.images = []
        self.masks = []
        missing_images = []
        missing_masks = []

        for stem in stems:
            img_path = self._find_existing(self.image_dir, stem, self.IMG_EXTS)
            mask_path = self._find_existing(self.mask_dir, stem, self.MASK_EXTS)

            if img_path is None:
                missing_images.append(stem)
            if mask_path is None:
                missing_masks.append(stem)

            # 只有都找到才加入，避免 __getitem__ 时报错
            if (img_path is not None) and (mask_path is not None):
                self.images.append(img_path)
                self.masks.append(mask_path)

        # --------- 5) 友好报错：告诉你缺哪些（最多打印 10 个）---------
        if missing_images or missing_masks:
            msg = ["VOCSegmentation: 根据 split 列表匹配文件时发现缺失："]
            if missing_images:
                msg.append(f"- 缺少图片文件（JPEGImages）：{len(missing_images)} 个，例如 {missing_images[:10]}")
                msg.append(f"  支持的图片后缀：{self.IMG_EXTS}")
            if missing_masks:
                msg.append(f"- 缺少mask文件（SegmentationClass）：{len(missing_masks)} 个，例如 {missing_masks[:10]}")
                msg.append(f"  支持的mask后缀：{self.MASK_EXTS}")
            msg.append("请检查：文件名是否和 txt 中的 id 对应；或扩展名是否在支持列表里。")
            raise RuntimeError("\n".join(msg))

        assert len(self.images) == len(self.masks)

    @staticmethod
    def _find_existing(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
        """在 dir_path 下按优先级尝试 stem+ext，返回第一个存在的路径，否则 None。"""
        for ext in exts:
            p = os.path.join(dir_path, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        # 如果你的 mask 不是单通道/调色板（例如被存成 RGB），这里给个兜底转换：
        # 注意：RGB->L 可能会破坏类别 id，强烈建议 mask 本身就是单通道类别图（P/L）
        if target.mode not in ("L", "P"):
            target = target.convert("L")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @classmethod
    def decode_target(cls, mask: np.ndarray) -> np.ndarray:
        return cls.cmap[mask]
