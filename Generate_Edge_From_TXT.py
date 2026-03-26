import os
import cv2
import numpy as np
from PIL import Image


def generate_edges_from_txt(txt_dir, rgb_dir, output_edge_dir, line_thickness=2):
    """
    读取包含坐标的 txt 文件，并结合原图尺寸生成边缘标签。
    line_thickness: 边界线的粗细（由于滑坡通常较大，建议线宽设为 2-3 个像素，太细网络不好学）
    """
    os.makedirs(output_edge_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        stem = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(txt_dir, txt_file)

        # 1. 寻找对应的 RGB 影像以获取尺寸
        rgb_path = os.path.join(rgb_dir, f"{stem}.png")  # 假设是 png
        if not os.path.exists(rgb_path):
            continue

        with Image.open(rgb_path) as img:
            width, height = img.size

        # 2. 创建纯黑画布
        edge_mask = np.zeros((height, width), dtype=np.uint8)

        # 3. 解析 txt 文件中的坐标
        pts = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                # 过滤掉表头，只提取纯数字坐标
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    x, y = int(parts[0]), int(parts[1])
                    pts.append([x, y])

        if len(pts) > 2:
            # 4. 将坐标转为 OpenCV 需要的 numpy 格式
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # 5. 在黑画布上画白色多边形轮廓（不填充）
            # isClosed=True 表示首尾相连，color=255 表示白色
            cv2.polylines(edge_mask, [pts], isClosed=True, color=255, thickness=line_thickness)

        # 6. 保存生成的边缘标签
        save_path = os.path.join(output_edge_dir, f"{stem}.png")
        cv2.imwrite(save_path, edge_mask)

    print("由 TXT 坐标提取边缘标签完成！")

# 使用示例
generate_edges_from_txt("Bijie-landslide-dataset/landslide/polygon_coordinate", "BijieOnlyLandslide/VOC2012/JPEGImages", "BijieOnlyLandslide/VOC2012/EdgeClass")