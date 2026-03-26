import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class WHULandslideDataset(Dataset):
    """
    针对 M2F-DeepLab 设计的混合数据集加载器
    支持混合真实影像与生成的 top-down satellite image。
    """

    def __init__(self, root_dir, split='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform  # 必须是能同时处理 (img, dem, mask, edge) 的联合变换

        # 路径定义
        self.img_dir = os.path.join(root_dir,'VOC2012', 'JPEGImages')
        self.dem_dir = os.path.join(root_dir,'VOC2012', 'DEMImages')
        self.mask_dir = os.path.join(root_dir,'VOC2012', 'SegmentationClass')
        self.edge_dir = os.path.join(root_dir,'VOC2012', 'EdgeClass')

        # 读取切分文件 (train.txt / val.txt)
        split_f = os.path.join(root_dir,'VOC2012', 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_f, "r") as f:
            self.file_names = [x.strip() for x in f.readlines() if x.strip()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        name = self.file_names[index]

        # 1. 读取 RGB 影像 (真实图或生成的 top-down satellite image)
        img_path = os.path.join(self.img_dir, f"{name}.png")  # 根据实际后缀修改
        img = Image.open(img_path).convert('RGB')

        # 2. 读取 Mask 和 Edge 标签
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        mask = Image.open(mask_path).convert('L')  # 强制转为单通道灰度图

        edge_path = os.path.join(self.edge_dir, f"{name}.png")
        edge = Image.open(edge_path).convert('L')  # 边缘通常作为单通道灰度图读取

        # 3. 动态读取 DEM 数据
        dem_path = os.path.join(self.dem_dir, f"{name}.png")  # 假设DEM保存为png
        if os.path.exists(dem_path):
            dem = Image.open(dem_path).convert('RGB')  # 假设你预处理成了3通道(坡度,坡向,高程)
            has_dem = True
        else:
            # 对于没有 DEM 的生成数据，创建一个与 RGB 同尺寸的零张量图片
            dem = Image.new('RGB', img.size, (0, 0, 0))
            has_dem = False

        # 4. 同步数据增强 (裁剪、翻转等)
        # 注意：这里的 transform 必须是你能重写的 joint_transform，
        # 让随机种子在四张图上保持一致。
        if self.transform is not None:
            img, dem, mask, edge = self.transform(img, dem, mask, edge)

        # 将 has_dem 转为 Tensor 以便送入网络
        has_dem_tensor = torch.tensor(has_dem, dtype=torch.bool)

        return img, dem, mask, edge, has_dem_tensor