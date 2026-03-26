import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def analyze_image_sizes(image_dir):
    files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tif'))]

    widths = []
    heights = []

    print(f"正在分析 {len(files)} 张影像的尺寸...")
    for f in tqdm(files):
        img_path = os.path.join(image_dir, f)
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    print("\n" + "=" * 40)
    print("📊 数据集尺寸统计报告")
    print("=" * 40)
    print(f"最大尺寸: {widths.max()} x {heights.max()} (宽 x 高)")
    print(f"最小尺寸: {widths.min()} x {heights.min()} (宽 x 高)")
    print(f"平均尺寸: {widths.mean():.1f} x {heights.mean():.1f} (宽 x 高)")
    print(f"中位数尺寸: {np.median(widths):.1f} x {np.median(heights):.1f} (宽 x 高)")
    print("=" * 40)

    # 给出 Crop Size 建议 (推荐 16 或 32 的整数倍)
    avg_size = (widths.mean() + heights.mean()) / 2
    recommended_crop = int(avg_size // 32 * 32)
    print(f"💡 建议的基准 Crop Size: 约 {recommended_crop} (它是 32 的整数倍)")

# 使用你的数据集路径
analyze_image_sizes("G:/LYW/DeepLabv3Segmentation/Bijie-landslide-dataset/landslide/image")