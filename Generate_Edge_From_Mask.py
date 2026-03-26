import os
import cv2
import numpy as np


def generate_edges_from_masks(mask_dir, output_edge_dir):
    """
    通过形态学梯度，直接从实体的 Segmentation Mask 中提取边缘轮廓。
    """
    os.makedirs(output_edge_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    # 定义一个 3x3 的卷积核，决定了提取出的边缘的粗细
    kernel = np.ones((3, 3), np.uint8)

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)

        # 以灰度模式读取 Mask (滑坡为 >0 的值，背景为 0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 二值化确保只有 0 和 255
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        # 使用形态学梯度提取边缘：膨胀后的图减去腐蚀后的图，剩下的就是边界
        edge_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)

        # 保存边缘标签
        save_path = os.path.join(output_edge_dir, mask_file)
        cv2.imwrite(save_path, edge_mask)

    print("由实体 Mask 提取边缘标签完成！")