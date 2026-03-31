# predict.py
# -*- coding: utf-8 -*-
"""
M2F-DeepLabV3+ (双流多模态) 推理脚本
- 支持 RGB + DEM 联合输入推理
- 自动处理缺失 DEM 的情况（填零）
"""

import argparse
import os
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

import network
from utils.create_exp_folder import create_val_exp_folder


# -----------------------------
# 1) VOC 调色板
# -----------------------------
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


def build_voc_palette(num_classes: int) -> np.ndarray:
    """
    如果是二分类，强制使用高对比度颜色（背景纯黑，滑坡纯红或纯白）
    """
    if num_classes == 2:
        # [0, 0, 0] 是黑色背景
        # [255, 0, 0] 是纯红色的滑坡 (如果你想输出纯白Mask，改为 [255, 255, 255])
        return np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)

    cmap = voc_cmap(256, normalized=False)
    return cmap[:num_classes]


# -----------------------------
# 2) checkpoint 加载兼容
# -----------------------------
def _torch_load_compat(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[len("module."):]] = v
    return new_sd


# -----------------------------
# 3) 模型构建与加载 (修复了 separable_conv 问题)
# -----------------------------
def build_model_from_opts(opts):
    model_fn = network.modeling.__dict__.get(opts.model, None)
    if model_fn is None:
        raise ValueError(f"[Predict] Unknown model name: {opts.model}")

    kwargs = {
        "num_classes": opts.num_classes,
        "output_stride": opts.output_stride,
        "pretrained_backbone": False,
    }

    sig = inspect.signature(model_fn)
    allowed = set(sig.parameters.keys())
    safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    model = model_fn(**safe_kwargs)
    return model


def load_checkpoint_strict(model, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[Predict] ckpt not found: {ckpt_path}")

    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    return ckpt


def load_model_for_predict(opts, device):
    if opts.ckpt is None or not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"找不到权重文件：{opts.ckpt}")

    # 1. 实例化基础模型
    model = build_model_from_opts(opts)

    # 2. 【核心修复】应用深度可分离卷积，与 train.py 保持绝对一致
    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)
        print("[Predict] Applied separable convolution to classifier.")

    # 3. 严格加载权重
    load_checkpoint_strict(model, opts.ckpt)

    model.to(device)
    model.eval()
    return model


# -----------------------------
# 4) 多模态输入预处理 + 推理 (修复了双流输入问题)
# -----------------------------
def preprocess_rgb(pil_img: Image.Image):
    """ RGB 标准化 (与 m2f_transforms.py 中 M2FNormalize 一致) """
    img = pil_img.convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).float().unsqueeze(0)


def preprocess_dem(dem_path: Path, target_size: tuple):
    """ DEM 转 Tensor (无需 mean/std 标准化，只需 /255.0) """
    if dem_path.exists():
        dem = Image.open(dem_path).convert("RGB")
    else:
        # 如果没有 DEM，直接生成对应尺寸的零张量图片
        dem = Image.new("RGB", target_size, (0, 0, 0))

    img = np.array(dem).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).float().unsqueeze(0)


# @torch.no_grad()
# def infer_one_image(model, device, img_path: Path, dem_dir: Path):
#     """ 双流推理（带硬核 BN 层崩坏诊断） """
#     # 1. 加载并处理 RGB
#     pil_img = Image.open(img_path).convert("RGB")
#     x_rgb = preprocess_rgb(pil_img).to(device)
#
#     # 2. 寻找真实 DEM 文件
#     dem_path_found = None
#     for ext in ['.png', '.tif', '.jpg', '.bmp']:
#         temp_path = dem_dir / f"{img_path.stem}{ext}"
#         if temp_path.exists():
#             dem_path_found = temp_path
#             break
#
#     if dem_path_found is not None:
#         print(f"  [+] 成功加载真实 DEM: {dem_path_found.name}")
#         x_dem = preprocess_dem(dem_path_found, pil_img.size).to(device)
#     else:
#         print(f"  [-] 警告: 未找到对应的 DEM 文件，输入纯黑特征...")
#         x_dem = preprocess_dem(Path("not_exist"), pil_img.size).to(device)
#
#     # ---------------------------------------------------------
#     # 🏥 终极诊断逻辑开始
#     # ---------------------------------------------------------
#
#     # 【模式 1】：正常的 Eval 模式 (使用历史 BN 统计量)
#     model.eval()
#     out_eval = model(x_rgb, x_dem)['seg']
#     prob_eval = torch.softmax(out_eval, dim=1)[0, 1].max().item()
#
#     # 【模式 2】：强制 Train 模式 (骗过 BN 层，使用当前图片的特征)
#     model.train()
#     # 注：BN 层要求 batch_size > 1 才能计算方差，所以我们把图片复制一份 [2, C, H, W]
#     x_rgb_fake_batch = torch.cat([x_rgb, x_rgb], dim=0)
#     x_dem_fake_batch = torch.cat([x_dem, x_dem], dim=0)
#     out_train = model(x_rgb_fake_batch, x_dem_fake_batch)['seg']
#     prob_train = torch.softmax(out_train, dim=1)[0, 1].max().item()
#
#     print(f"\n  [🏥 深度诊断报告] - {img_path.name}")
#     print(f"  [1] Eval 模式最高概率 : {prob_eval:.4f} (这就是你之前全黑的原因)")
#     print(f"  [2] Train模式最高概率 : {prob_train:.4f} ")
#
#     # --------- 分析与动态修复 ---------
#     if prob_train > 0.5 and prob_eval < 0.5:
#         print("  [💡 结论] 铁证如山！确认是 Batch Normalization 统计量崩溃！")
#         print("            已为你临时应用 Train 模式特征来生成预测结果。")
#         final_logits = out_train[0:1]  # 取回骗过 BN 后的第一张图
#     else:
#         if prob_train < 0.5:
#             print("  [💡 结论] 两个概率都很低。这说明 BN 没问题，而是【推理尺度】与训练时严重不符！")
#             print("            原因：你输入的这张原图太大或太小，导致网络提取不到 513x513 视野下的滑坡特征。")
#         final_logits = out_eval
#
#     # 恢复 Eval 模式，保持良好习惯
#     model.eval()
#
#     # 取出最终预测图
#     pred = torch.argmax(final_logits, dim=1)[0].cpu().numpy().astype(np.uint8)
#
#     unique_classes = np.unique(pred)
#     print(f"  [>] 最终输出像素类别: {unique_classes}")
#
#     return pred, pil_img


@torch.no_grad()
def infer_one_image(model, device, img_path: Path, dem_dir: Path):
    """ 双流推理（标准纯净版） """
    # 1. 加载并处理 RGB
    pil_img = Image.open(img_path).convert("RGB")
    x_rgb = preprocess_rgb(pil_img).to(device)

    # 2. 智能寻找 DEM 文件
    dem_path_found = None
    for ext in ['.png', '.tif', '.jpg', '.bmp']:
        temp_path = dem_dir / f"{img_path.stem}{ext}"
        if temp_path.exists():
            dem_path_found = temp_path
            break

    if dem_path_found is not None:
        x_dem = preprocess_dem(dem_path_found, pil_img.size).to(device)
    else:
        # 如果是生成的影像，没有真实 DEM，自动填零
        x_dem = preprocess_dem(Path("not_exist"), pil_img.size).to(device)

    # 3. 严格使用 Eval 模式前向传播
    model.eval()
    outputs = model(x_rgb, x_dem)
    logits = outputs['seg']

    # 取出最终预测图
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    return pred, pil_img

def colorize_mask(mask: np.ndarray, palette: np.ndarray):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        out[mask == cls_id] = palette[cls_id]
    return out


def overlay_image(rgb_img: np.ndarray, color_mask: np.ndarray, alpha: float):
    rgb = rgb_img.astype(np.float32)
    m = color_mask.astype(np.float32)
    out = rgb * (1 - alpha) + m * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# 5) 批量推理
# -----------------------------
def is_image_file(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def run_on_images(model, device, input_path: Path, dem_dir: Path, out_dir: Path,
                  palette: np.ndarray, save_mask: bool, save_overlay: bool, alpha: float):
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    if save_mask: mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay: overlay_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        img_list = [input_path]
    else:
        img_list = sorted([p for p in input_path.iterdir() if p.is_file() and is_image_file(p)])

    if len(img_list) == 0:
        print(f"[Predict] No images found in: {input_path}")
        return

    for img_path in tqdm(img_list, desc="Infer Images", ncols=100):
        # 调用双流推理函数
        pred, pil_img = infer_one_image(model, device, img_path, dem_dir)

        color_mask = colorize_mask(pred, palette)
        stem = img_path.stem

        if save_mask:
            Image.fromarray(color_mask).save(mask_dir / f"{stem}_mask.png")

        if save_overlay:
            rgb = np.array(pil_img).astype(np.uint8)
            ov = overlay_image(rgb, color_mask, alpha=alpha)
            Image.fromarray(ov).save(overlay_dir / f"{stem}_overlay.png")


# -----------------------------
# 6) 主函数
# -----------------------------
def main(opts):
    exp_dir = Path(create_val_exp_folder())
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Folder] Predict exp dir: {exp_dir}")

    device = torch.device(opts.device if torch.cuda.is_available() and "cuda" in opts.device else "cpu")
    print(f"Device: {device}")

    model = load_model_for_predict(opts, device=device)
    print(f"[CKPT] Loaded weights successfully: {opts.ckpt}")

    palette = build_voc_palette(opts.num_classes)
    in_path = Path(opts.input)
    dem_path = Path(opts.dem_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"[Predict] input not found: {in_path}")

    run_on_images(
        model=model, device=device, input_path=in_path, dem_dir=dem_path, out_dir=exp_dir,
        palette=palette, save_mask=opts.save_mask, save_overlay=opts.save_overlay, alpha=opts.alpha
    )
    print("[Predict] Done.")


if __name__ == "__main__":
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and not name.startswith("_") and callable(network.modeling.__dict__[name])
    )

    parser = argparse.ArgumentParser()
    # ---------------- 输入/模型 ----------------
    parser.add_argument("--input", type=str, default=r"BijieLandslide/VOC2012/JPEGImages",
                        help="输入：单张图片 / 包含 RGB 的文件夹")
    parser.add_argument("--dem_dir", type=str, default=r"BijieLandslide/VOC2012/DEMImages",
                        help="DEM 文件夹的路径（用于双流推理）")

    parser.add_argument("--ckpt", type=str,
                        default=r"run/train/exp18/weights/best_m2f_deeplabv3plus_xception16_os8_num2.pth",
                        help="模型权重路径")
    parser.add_argument("--model", type=str, default="m2f_deeplabv3plus_xception16", choices=available_models)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--output_stride", type=int, default=8)  # 请确保与你训练时保存的 stride 一致！
    parser.add_argument("--separable_conv", action="store_true", default=False,
                        help="是否使用了深度可分离卷积（与训练保持一致）")

    # ---------------- 输出形式 ----------------
    parser.add_argument("--save_mask", action="store_true", default=True)
    parser.add_argument("--save_overlay", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")

    opt = parser.parse_args()
    if (not opt.save_mask) and (not opt.save_overlay):
        opt.save_overlay = True
    main(opt)