# M2FPredict.py
# -*- coding: utf-8 -*-
"""
最终优化版 M2F-DeepLabV3+ 推理脚本
- 找回了 --separable_conv 兼容性支持
- 恢复了 --ckpt 默认路径和 --alpha 参数
- 恢复了 --save_mask/overlay/analysis 开关控制
- 保留并完善了 TP/FP/FN 误差可视化功能
"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn

import network
from utils.create_exp_folder import create_val_exp_folder

# 误差分析颜色配置 (BGR)
COLOR_TP = [0, 255, 0]  # 🟩 绿色：预测正确 (TP)
COLOR_FP = [255, 0, 0]  # 🟦 蓝色：误报 - 本不是却预测是 (FP)
COLOR_FN = [0, 0, 255]  # 🟥 红色：漏报 - 本来是却没认出 (FN)
COLOR_MASK = [0, 255, 255]  # 🟨 黄色：基础覆盖层


def main(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 构建模型基础架构
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
        pretrained_backbone=False
    )

    # 【核心兼容性修复】：如果训练时用了深度可分离卷积，此处必须转换结构
    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)
        print("已成功配置深度可分离卷积 (Separable Conv) 结构以匹配权重。")

    # 2. 加载权重
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"未找到指定的模型权重文件: {opts.ckpt}")

    checkpoint = torch.load(opts.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    print(f"成功加载模型: {opts.model} 来自 {opts.ckpt}")

    # 3. 创建输出文件夹结构
    val_dir = create_val_exp_folder()
    out_mask_dir = os.path.join(val_dir, 'mask')
    out_overlay_dir = os.path.join(val_dir, 'overlay')
    out_analysis_dir = os.path.join(val_dir, 'analysis')

    if opts.save_mask: os.makedirs(out_mask_dir, exist_ok=True)
    if opts.save_overlay: os.makedirs(out_overlay_dir, exist_ok=True)
    if opts.save_analysis and opts.gt_dir: os.makedirs(out_analysis_dir, exist_ok=True)

    # 4. 获取待处理列表
    img_files = []
    if os.path.isdir(opts.input):
        img_files = [f for f in os.listdir(opts.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        img_path_list = [os.path.join(opts.input, f) for f in img_files]
    else:
        img_path_list = [opts.input]

    # 5. 推理循环
    with torch.no_grad():
        for img_path in tqdm(img_path_list):
            img_name = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_name)[0]

            # 读取原图
            raw_img = cv2.imread(img_path)
            h, w = raw_img.shape[:2]

            # 标准化处理
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(device)

            # 读取 DEM (如果存在)
            dem_path = os.path.join(opts.dem_dir, img_name_no_ext + '.tif')
            if os.path.exists(dem_path):
                dem_pil = Image.open(dem_path)
                dem_tensor = torch.from_numpy(np.array(dem_pil)).unsqueeze(0).unsqueeze(0).float().to(device)
            else:
                dem_tensor = torch.zeros((1, 1, h, w)).to(device)

            # 模型预测
            outputs = model(img_tensor, dem_tensor)
            pred_mask = outputs['seg'].max(1)[1].cpu().numpy()[0].astype(np.uint8)

            # 产出 1: 纯 Mask
            if opts.save_mask:
                res_mask = Image.fromarray(pred_mask)
                res_mask.save(os.path.join(out_mask_dir, img_name_no_ext + '.png'))

            # 产出 2: Overlay 透明覆盖图
            if opts.save_overlay:
                overlay_out = raw_img.copy().astype(np.float32)
                overlay_out[pred_mask == 1] = overlay_out[pred_mask == 1] * (1 - opts.alpha) + np.array(
                    COLOR_MASK) * opts.alpha
                cv2.imwrite(os.path.join(out_overlay_dir, img_name_no_ext + '.png'), overlay_out.astype(np.uint8))

            # 产出 3: TP/FP/FN 误差分析 (受 --save_analysis 控制)
            if opts.save_analysis and opts.gt_dir:
                gt_path = os.path.join(opts.gt_dir, img_name_no_ext + '.png')
                if os.path.exists(gt_path):
                    gt_mask = np.array(Image.open(gt_path))
                    gt_mask_clean = np.zeros_like(gt_mask)
                    gt_mask_clean[gt_mask == 1] = 1  # 提取滑坡类

                    analysis_out = raw_img.copy().astype(np.float32)

                    # 按照逻辑组合绘制
                    tp_mask = (pred_mask == 1) & (gt_mask_clean == 1)  # 正确
                    fp_mask = (pred_mask == 1) & (gt_mask_clean == 0)  # 误报
                    fn_mask = (pred_mask == 0) & (gt_mask_clean == 1)  # 漏报

                    analysis_out[tp_mask] = analysis_out[tp_mask] * (1 - opts.alpha) + np.array(COLOR_TP) * opts.alpha
                    analysis_out[fp_mask] = analysis_out[fp_mask] * (1 - opts.alpha) + np.array(COLOR_FP) * opts.alpha
                    analysis_out[fn_mask] = analysis_out[fn_mask] * (1 - opts.alpha) + np.array(COLOR_FN) * opts.alpha

                    cv2.imwrite(os.path.join(out_analysis_dir, img_name_no_ext + '.png'), analysis_out.astype(np.uint8))

    print(f"\n推理与多维度可视化任务完成！结果存放在: {val_dir}")


if __name__ == '__main__':
    available_models = sorted(name for name in network.modeling.__dict__ if
                              name.islower() and not name.startswith("_") and callable(network.modeling.__dict__[name]))

    parser = argparse.ArgumentParser()
    # ---------------- 输入/输出路径 ----------------
    parser.add_argument("--input", type=str, default="BijieLandslide/VOC2012/JPEGImages", help="待推理影像路径")
    parser.add_argument("--dem_dir", type=str, default="BijieLandslide/VOC2012/DEMImages", help="DEM 影像路径")
    parser.add_argument("--gt_dir", type=str, default="BijieLandslide/VOC2012/SegmentationClass",
                        help="真值标签路径 (用于误差分析)")

    # ---------------- 权重与模型设定 (恢复默认值) ----------------
    parser.add_argument("--ckpt", type=str,
                        default=r"run/train/exp6/weights/best_m2f_deeplabv3plus_xception16_os16_num2.pth",
                        help="模型权重路径")
    parser.add_argument("--model", type=str, default="m2f_deeplabv3plus_xception16", choices=available_models)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--output_stride", type=int, default=16)
    parser.add_argument("--separable_conv", action="store_true", default=False,
                        help="是否使用深度可分离卷积 (需与训练一致)")

    # ---------------- 可视化开关与参数 (恢复) ----------------
    parser.add_argument("--save_mask", action="store_true", default=True, help="是否保存纯预测 Mask")
    parser.add_argument("--save_overlay", action="store_true", default=True, help="是否保存原图叠加预测图")
    parser.add_argument("--save_analysis", action="store_true", default=True, help="是否保存 TP/FP/FN 误差分析图")
    parser.add_argument("--alpha", type=float, default=0.5, help="覆盖层与误差层的透明度 (0.0-1.0)")

    opts = parser.parse_args()
    main(opts)