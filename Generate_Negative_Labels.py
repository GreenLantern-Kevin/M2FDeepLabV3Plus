import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_blank_labels(rgb_dir, mask_dir, edge_dir):
    """
    为没有标签的非滑坡影像（负样本）生成全黑的 Mask 和 Edge。
    """
    # 确保文件夹存在
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    # 获取所有 PNG 影像的文件名 (支持 .png),可改
    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
    count = 0

    print(f"开始扫描 {len(rgb_files)} 张影像，检查缺失的标签...")

    for rgb_file in tqdm(rgb_files):
        stem = os.path.splitext(rgb_file)[0]
        rgb_path = os.path.join(rgb_dir, rgb_file)

        mask_path = os.path.join(mask_dir, f"{stem}.png")
        edge_path = os.path.join(edge_dir, f"{stem}.png")

        # 如果 Mask 不存在，说明这可能是一张非滑坡影像
        if not os.path.exists(mask_path):
            # 1. 读取原图以获取其精准的长宽尺寸
            with Image.open(rgb_path) as img:
                width, height = img.size

            # 2. 生成全黑的单通道 numpy 数组 (像素值全为 0)
            blank_array = np.zeros((height, width), dtype=np.uint8)
            blank_img = Image.fromarray(blank_array, mode='L')

            # 3. 分别保存为 Mask 和 Edge 标签
            blank_img.save(mask_path)
            blank_img.save(edge_path)
            count += 1

    print(f"\n扫描完毕！共为 {count} 张非滑坡影像生成了全黑的背景标签 (Mask & Edge)。")
    print("现在所有的影像都拥有 1:1 对应的标签文件了，可以安全开始训练！")

# ================= 使用示例 =================
rgb_dir = "BijieLandslide/VOC2012/JPEGImages"
mask_dir = "BijieLandslide/VOC2012/SegmentationClass"
edge_dir = "BijieLandslide/VOC2012/EdgeClass"
generate_blank_labels(rgb_dir, mask_dir, edge_dir)