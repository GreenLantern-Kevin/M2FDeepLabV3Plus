# utils/engine.py
# -*- coding: utf-8 -*-
"""
训练和验证共用的一些功能函数：
- 设备与随机种子初始化
- num_classes 自动推断
- 数据集 & DataLoader 构建
- 模型、优化器、学习率策略、损失函数构建
- checkpoint 保存
- 验证 validate（可选保存可视化结果）

注意：这里只提供通用逻辑，真正的参数定义和 main() 在 train.py / eval.py 中。
"""

import os
import random

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import data

import network
import utils
import cv2
# --- 旧代码 ---
# from utils import VOCSegmentation
# from utils import ext_transforms as et
# from utils.stream_metrics import StreamSegMetrics

# --- 替换为新代码 ---
from utils.Landslide import WHULandslideDataset
from utils import m2f_transforms as m2ft
from utils.stream_metrics import StreamSegMetrics

# ------------------------------------------------------------
# 1. 设备与随机种子
# ------------------------------------------------------------
def setup_device_and_seed(opts):
    """
    根据 opts.gpu_id 设置 CUDA_VISIBLE_DEVICES，
    并返回当前使用的 device（cuda / cpu）。
    同时固定随机种子，方便复现。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    return device


def resolve_num_classes(opts):
    """
    强制用户显式指定 num_classes。

    因为你后续使用的是“VOC 目录结构格式的自定义数据集”，类别数不再固定为 VOC 的 21 类，
    如果默认写死 21 很容易导致训练/验证全错而不自知。

    用法示例：
      python train.py --num_classes 3 ...
    """
    if getattr(opts, "num_classes", None) is None:
        raise ValueError(
            "你必须显式传入 --num_classes（含背景）。例如：--num_classes 3")



# ------------------------------------------------------------
# 2. 数据集 & DataLoader
# ------------------------------------------------------------
def get_dataset(opts):
    """
        构建多模态滑坡数据集 (WHULandslideDataset) 以及对应的四输入数据增强。
        支持同时处理: img(RGB), dem(高程/坡度/坡向), mask(滑坡掩码), edge(滑坡边界)
        """

    # --------- 训练集增强 (四模态同步) ----------
    train_transform = m2ft.M2FCompose([
        # 【核心修复 3】：严格对齐 mmseg 的缩放比例
        # 防止过小缩放导致全图只剩黑边
        m2ft.M2FRandomScale((0.8, 1.2)),
        # 随机裁剪到 (crop_size, crop_size)，不足则 pad
        m2ft.M2FRandomCrop(size=opts.crop_size, pad_if_needed=True),
        
        # 【修改这里：加入我们刚刚写的 ColorJitter】
        m2ft.M2FColorJitter(brightness=0.32, contrast=0.5, saturation=0.5, hue=0.1),
        
        # 随机水平翻转
        m2ft.M2FRandomHorizontalFlip(),
        # 转 tensor (其中 Mask 转为 long, Edge 转为 float)
        m2ft.M2FToTensor(),
        # 标准化（仅对 RGB 应用 ImageNet 均值方差，DEM 保持自身归一化状态）
        m2ft.M2FNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # --------- 验证集增强 (四模态同步) ----------
    # 验证时通常使用原图尺寸评估（不裁剪），以获得最真实的 mIoU
    val_transform = m2ft.M2FCompose([
        m2ft.M2FValResize((320, 320)), # ！！！核心救命代码：加在 ToTensor 前面！！！
        m2ft.M2FToTensor(),
        m2ft.M2FNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 如果你强制要求 crop_val=True（例如显存实在不够），需要自己实现 M2FCenterCrop
    if opts.crop_val:
        print("[Warning] 当前使用的是多模态验证，暂未实现 CenterCrop，将强制使用原图进行验证。")
        opts.crop_val = False

    # --------- 数据集实例化 ----------
    # 注意：这里的数据集根目录 opts.data_root 应该指向包含 JPEGImages, DEMImages 等的顶层目录
    train_dst = WHULandslideDataset(
        root_dir=opts.data_root,
        split="train",
        transform=train_transform
    )

    val_dst = WHULandslideDataset(
        root_dir=opts.data_root,
        split="val",
        transform=val_transform
    )

    return train_dst, val_dst


def build_loaders(opts):
    """
        根据 opts 构建多模态的 train_loader 和 val_loader
        """

    # 验证不 crop（原图评估）时，为避免 batch 内图片尺寸不一致导致张量堆叠失败，强制设为 1
    if not getattr(opts, 'crop_val', False):
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)

    # 训练 DataLoader
    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=4,  # 建议调大到 4 或 8 以加速数据预处理
        drop_last=True,
        pin_memory=True  # 开启锁页内存，加速 GPU 传输
    )

    # 验证 DataLoader
    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_dst, val_dst, train_loader, val_loader



# ------------------------------------------------------------
# 3. 模型 / 指标 / 优化器 / scheduler / loss
# ------------------------------------------------------------
def build_model(opts):
    """
    构建 DeepLab 模型：
    - backbone/ASPP/decoder 都由 network.modeling 对应工厂函数创建
    - 如果 opts.separable_conv 且是 v3+，则将 classifier 中的 conv 替换为 depthwise separable
    - 设置 backbone 的 BN momentum
    """
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
    )

    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)

    # BN momentum 设小一些（0.01），这在 DeepLab 系列中很常见
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    return model


def build_metrics(opts):
    """构建分割指标计算器（StreamSegMetrics：内部是混淆矩阵 + mIoU 计算）"""
    return StreamSegMetrics(opts.num_classes)


def build_optimizer(opts, model):
    """
    【终极自适应差分学习率优化器】
    自动侦测加载的权重类型，动态分配 LR 策略！
    """
    pretrained_params = []
    scratch_params = []
    
    # 获取权重名字的小写，用于智能判断
    ckpt_name = opts.ckpt.lower() if opts.ckpt else ""

    # ================= 智能策略路由 =================
    if "single" in getattr(opts, 'model', ''):
        # 局势 1：纯单流模型 -> 标准逻辑
        strategy = "SINGLE_STANDARD"
        
    elif "single" in ckpt_name:
        # 局势 2：两阶段微调（加载了单流专家权重） -> 非对称保护！
        # RGB 是专家，DEM 是随机婴儿
        strategy = "ASYMMETRIC_PROTECT"
        
    elif "m2f" in ckpt_name or "best" in ckpt_name:
        # 局势 3：热重启微调（加载了双流专家权重，如你的 0.568） -> 对称保护！
        # RGB 和 DEM 都已经是专家了
        strategy = "SYMMETRIC_PROTECT"
        
    else:
        # 局势 4：端到端单阶段（加载 ImageNet 或从零开始） -> 对称保护！
        # 需要保护底层的物理边缘检测算子不被破坏
        strategy = "SYMMETRIC_PROTECT"
    # ================================================
    
    print(f"\\n[Optimizer] 智能推断训练局势: {strategy}")

    # 执行参数分组
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if strategy == "SINGLE_STANDARD":
            if 'backbone' in name:
                pretrained_params.append(param)
            else:
                scratch_params.append(param)
                
        elif strategy == "ASYMMETRIC_PROTECT":
            # 只有 RGB 被保护 (0.1x)，DEM / 融合 / 解码器 全部狂奔 (1.0x)
            if 'rgb_backbone' in name:
                pretrained_params.append(param)
            else:
                scratch_params.append(param)
                
        elif strategy == "SYMMETRIC_PROTECT":
            # RGB 和 DEM 都被保护 (0.1x)，只有 融合 / 解码器 狂奔 (1.0x)
            if 'rgb_backbone' in name or 'dem_backbone' in name:
                pretrained_params.append(param)
            else:
                scratch_params.append(param)

    print(f"[Optimizer] 🛡️  受保护的主干参数: {len(pretrained_params)} 个 (LR: {opts.lr * 0.1})")
    print(f"[Optimizer] 🚀 活跃的冲刺层参数: {len(scratch_params)} 个 (LR: {opts.lr})\\n")

    # 构建优化器
    optimizer = torch.optim.SGD([
        {'params': pretrained_params, 'lr': opts.lr * 0.1}, 
        {'params': scratch_params, 'lr': opts.lr}           
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
    return optimizer


def build_scheduler(opts, optimizer):
    """
    构建学习率调度器（scheduler）：
    - poly：分割常用策略，随 iter 平滑衰减
    - step：每隔 step_size 把 lr 乘以 0.1
    """
    if opts.lr_policy == "poly":
        return utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=opts.step_size,
            gamma=0.1,
        )
    else:
        raise ValueError(f"Unknown lr_policy: {opts.lr_policy}")


def build_criterion(opts):
    """
    构建损失函数：
    - cross_entropy: nn.CrossEntropyLoss(ignore_index=255)
    - focal_loss   : utils.FocalLoss(ignore_index=255)
    """
    if opts.loss_type == "focal_loss":
        return utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    else:
        raise ValueError(f"Unknown loss_type: {opts.loss_type}")


# ------------------------------------------------------------
# 4. Checkpoint 保存
# ------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, cur_itrs, best_score, model_name=None):
    """
    保存训练快照（model 需要是 nn.DataParallel 包了一层的）。
    """
    ckpt = {
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }
    # 记录模型名（用于更友好的“模型不匹配”提示）
    if model_name is not None:
        ckpt["model_name"] = model_name

    torch.save(ckpt, path)


# ------------------------------------------------------------
def validate(opts, model, loader, device, metrics, criterion=None, epoch=None, total_epochs=None):
    """
    在验证集上跑一遍模型，计算指标（Overall Acc / Mean IoU 等），
    并（可选）统计平均 val loss。**不做任何可视化保存**。

    参数:
        opts         : 命令行配置对象（这里只用到 epoch/total_epochs 做显示）
        model        : 已放到 device 上的分割模型（外部已 DataParallel 包裹）
        loader       : 验证集 DataLoader
        device       : torch.device
        metrics      : StreamSegMetrics 实例，用于累计混淆矩阵并计算 mIoU 等
        criterion    : 可选的损失函数；若不为 None，则在验证集上统计平均 loss
        epoch        : 当前第几个 epoch（用于 tqdm 标题，可为 None）
        total_epochs : 总 epoch 数（用于 tqdm 标题，可为 None）

    返回:
        score (dict): metrics.get_results() 的结果字典，并额外包含：
            - score["Loss"]：验证集平均 loss（若 criterion=None，则为 0.0）
    """
    # 每次验证前先重置指标统计（清空混淆矩阵）
    metrics.reset()

    # 统计验证集上的总 loss 和 batch 数，方便算平均 loss
    val_loss_sum = 0.0
    val_batches = 0

    # 切换到 eval 模式：关闭 Dropout，BN 使用 running_mean/var
    model.eval()

    # 进度条标题
    if epoch is not None and total_epochs is not None:
        desc = f"Val {epoch}/{total_epochs}"
    else:
        desc = "Val"

    # 验证阶段不需要梯度
    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc=desc, ncols=100)

        for images, dems, labels, edges, has_dem in pbar:
            # 搬到设备
            images = images.to(device, dtype=torch.float32)
            dems = dems.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # ================= 修改点：验证集前向推理兼容单流与多模态 =================
            if "single" in getattr(opts, 'model', ''):
                # 【情况 A：单流 Baseline 模型】
                # 只传 RGB 影像，官方模型直接返回 [N, C, H, W] 的预测张量
                outputs = model(images)
                if isinstance(outputs, dict):
                    pred_mask = outputs['seg']
                else:
                    pred_mask = outputs
            else:
                # 【情况 B：多模态改进模型】
                # 传入 RGB 和 DEM，返回包含各种特征的字典，提取主分支的 'seg'
                outputs = model(images, dems)    # 前向推理 (调用双流模型)
                pred_mask = outputs['seg']    # 注意：M2F-DeepLab 的输出是一个字典，你需要提取主分支的 mask 预测
            # ============================================================================

            # 若提供了损失函数，则在验证集上也统计 loss
            if criterion is not None:
                loss = criterion(pred_mask, labels)  # 仅计算主任务分割 loss 作为参考
                val_loss_sum += float(loss.detach().cpu().numpy())
                val_batches += 1

            # 取得预测类别 & 真实标签，转到 CPU + numpy，用于更新混淆矩阵
            # ================= 终极后处理：TTA + 连通域假阳性斩杀 =================
            import cv2
            import numpy as np
            
            # 1. 原始方向预测
            probs_orig = torch.nn.functional.softmax(pred_mask.detach(), dim=1)
            
            # 2. TTA 镜像翻转预测 (极大平滑边缘)
            images_flip = torch.flip(images, dims=[3])
            if "single" in getattr(opts, 'model', ''):
                outputs_flip = model(images_flip)
                pred_mask_flip = outputs_flip['seg'] if isinstance(outputs_flip, dict) else outputs_flip
            else:
                dems_flip = torch.flip(dems, dims=[3])
                outputs_flip = model(images_flip, dems_flip)
                # 兼容你的返回值 (mask, edge, feature) 
                pred_mask_flip = outputs_flip[0] if isinstance(outputs_flip, tuple) else outputs_flip['seg']
                
            probs_flip = torch.nn.functional.softmax(pred_mask_flip.detach(), dim=1)
            probs_flip = torch.flip(probs_flip, dims=[3]) # 翻转回来
            
            # 3. 双路融合，并采用 0.55 的严格阈值卡掉虚胖边缘 (保 Precision)
            probs = (probs_orig + probs_flip) / 2.0
            preds = (probs[:, 1, :, :] > 0.55).cpu().numpy().astype(np.uint8)

            # 4. 连通域过滤：绝杀零星假阳性！(不影响真实大滑坡，只切掉背景里的噪点)
            for i in range(preds.shape[0]):
                mask_i = preds[i]
                # 获取连通域
                num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(mask_i, connectivity=8)
                for label_idx in range(1, num_labels):
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    # 滑坡是巨型地质灾害，小于 400 像素的预测斑块 100% 是网络产生的假阳性，直接归零！
                    if area < 400:  
                        mask_i[labels_map == label_idx] = 0
                preds[i] = mask_i
            # =========================================================================
            
            targets = labels.cpu().numpy()

            # 更新混淆矩阵
            metrics.update(targets, preds)

            # --------- tqdm 动态显示（可读性更好）---------
            # 注意：每个 batch 都调用 metrics.get_results() 会有一点额外开销，
            # 但 VOC 一般可接受；如果你后面觉得慢，可以改成“每 N 个 batch 更新一次”。
            score_now = metrics.get_results()
            overall_acc = score_now.get("Overall Acc", 0.0)
            mean_iou = score_now.get("Mean IoU", 0.0)

            avg_loss = (val_loss_sum / val_batches) if val_batches > 0 else 0.0

            pbar.set_postfix({
                "loss": f"{avg_loss:.3f}" if criterion is not None else "N/A",
                "PAcc": f"{overall_acc:.3f}",
                "mIoU": f"{mean_iou:.3f}",
            })

    # 所有 batch 跑完后，返回综合指标
    score = metrics.get_results()

    # 把验证平均 loss 放进 score，方便你写 CSV/画图
    # 如果 criterion=None，则用 0.0 占位（也可以改成 float("nan")）
    score["Loss"] = (val_loss_sum / val_batches) if val_batches > 0 else 0.0

    return score


def _strip_module_prefix(state_dict):
    """去掉 DataParallel 保存时带的 'module.' 前缀（如果有）"""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_state_dict_by_shape(model, ckpt_state_dict, verbose=True):
    """
    只加载 shape 完全一致的参数，跳过不一致的（常见：分类头最后一层 num_classes 不同）。
    返回：matched_keys, skipped_keys
    """
    ckpt_state_dict = _strip_module_prefix(ckpt_state_dict)

    model_state = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in ckpt_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    if verbose:
        print(f"[CKPT] Shape-filter load: matched={len(loadable)} skipped={len(skipped)} "
              f"missing(after)={len(missing)} unexpected(after)={len(unexpected)}")
        # 如果你想看 skipped 里具体有哪些，可以取消注释：
        # print("[CKPT] Skipped keys (shape mismatch or not found):")
        # for k in skipped[:30]: print("  ", k)

    return list(loadable.keys()), skipped



def restore_from_checkpoint(opts, model, optimizer, scheduler, train_loader, device=None):
    """
    从 ckpt 恢复训练状态。

    兼容点：
      - PyTorch 1.10：torch.load 不支持 weights_only 参数（由 _torch_load_compat 负责兼容）
      - PyTorch 2.x：_torch_load_compat 内部会尝试 weights_only=False（你信任 ckpt 来源时）

    预训练 vs 断点续训：
      - continue_training=True ：严格恢复（模型结构必须完全一致），并恢复 optimizer/scheduler/iters/best
      - continue_training=False：把 ckpt 当“预训练权重”，若结构不完全一致（常见：num_classes 不同），
                                自动按 shape 过滤加载（跳过不匹配层，例如分类头最后一层）

    断点续训关键点：
      - optimizer.load_state_dict 后，需要把 optimizer.state 里的 Tensor 迁移到 device，
        否则会出现 cuda/cpu 混用报错（你之前那个 RuntimeError 就是这个原因）。
    """
    best_score = 0.0
    cur_itrs = 0
    start_epoch = 0

    # --------------------------
    # 内部小工具：对齐/过滤 state_dict
    # --------------------------
    def _is_state_dict_like(d):
        """判断一个 dict 是否像 state_dict：value 大多是 Tensor"""
        if not isinstance(d, dict) or len(d) == 0:
            return False
        # 只抽样检查前几个 key
        n = 0
        for _, v in d.items():
            n += 1
            if n > 20:
                break
            if not torch.is_tensor(v):
                return False
        return True

    def _align_ckpt_keys_to_model(ckpt_sd, model_sd):
        """
        处理 DataParallel 'module.' 前缀问题：
        - ckpt 里有 module. 但 model 没有 -> 去掉
        - ckpt 里没有 module. 但 model 有 -> 加上
        """
        if not ckpt_sd:
            return ckpt_sd

        model_has_module = any(k.startswith("module.") for k in model_sd.keys())
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_sd.keys())

        if ckpt_has_module and (not model_has_module):
            # 去掉 module.
            new_sd = {}
            for k, v in ckpt_sd.items():
                if k.startswith("module."):
                    new_sd[k[len("module."):]] = v
                else:
                    new_sd[k] = v
            return new_sd

        if (not ckpt_has_module) and model_has_module:
            # 加上 module.
            return {("module." + k): v for k, v in ckpt_sd.items()}

        return ckpt_sd

    def _load_by_shape(model, ckpt_sd, verbose=True):
        """
        只加载 shape 完全一致的参数（用于 num_classes 不同 / head 不匹配 等情况）。
        返回：matched_keys, skipped_keys
        """
        model_sd = model.state_dict()
        ckpt_sd = _align_ckpt_keys_to_model(ckpt_sd, model_sd)

        loadable = {}
        skipped = []

        for k, v in ckpt_sd.items():
            if (k in model_sd) and (model_sd[k].shape == v.shape):
                loadable[k] = v
            else:
                skipped.append(k)

        missing, unexpected = model.load_state_dict(loadable, strict=False)

        if verbose:
            print(f"[CKPT] Shape-filter load: matched={len(loadable)} skipped={len(skipped)} "
                  f"missing(after)={len(missing)} unexpected(after)={len(unexpected)}")

            # 给你一个更友好的提示：通常就是 classifier 最后一层不匹配
            # （避免打印几百行 keys，强迫症友好）
            hint = []
            for k in skipped:
                if "classifier" in k or "aspp" in k:
                    hint.append(k)
                if len(hint) >= 5:
                    break
            if len(hint) > 0:
                print("[CKPT] Skipped examples (usually due to num_classes mismatch):")
                for k in hint:
                    print(f"  - {k}")

        return list(loadable.keys()), skipped

    # --------------------------
    # 1) ckpt 路径检查 + 读取
    # --------------------------
    if opts.ckpt is None or (not os.path.isfile(opts.ckpt)):
        print("[!] No valid ckpt found, train from scratch.")
        return model, best_score, cur_itrs, start_epoch

    # 额外防一手：空文件/损坏文件
    if os.path.getsize(opts.ckpt) == 0:
        raise RuntimeError(f"[CKPT] checkpoint 文件大小为 0，可能保存中断或损坏：{opts.ckpt}")

    # 兼容加载（由你工程里的 _torch_load_compat 负责处理 PyTorch 1.10/2.x 差异）
    try:
        checkpoint = _torch_load_compat(opts.ckpt, map_location="cpu")
    except EOFError:
        raise RuntimeError(f"[CKPT] checkpoint 读取失败（EOFError），文件可能不完整：{opts.ckpt}") from None

    # 支持两种格式：
    # A) 我们保存的完整 dict：{"model_state":..., "optimizer_state":..., ...}
    # B) 外部只给了 state_dict（纯参数）
    if isinstance(checkpoint, dict) and ("model_state" in checkpoint):
        ckpt_state = checkpoint["model_state"]
        ckpt_model_name = checkpoint.get("model_name", None)
    elif _is_state_dict_like(checkpoint):
        ckpt_state = checkpoint
        ckpt_model_name = None
        # 统一成 dict，方便后续逻辑复用
        checkpoint = {"model_state": ckpt_state}
    else:
        raise RuntimeError(f"[CKPT] 无法识别的 checkpoint 格式：{opts.ckpt}")

    # --------------------------
    # 2) 如果 ckpt 写了 model_name，就做强一致检查（避免你用 mobilenet 去加载 xception）
    # --------------------------
    if ckpt_model_name is not None and ckpt_model_name != opts.model:
        print("[CKPT] 模型与权重不匹配：")
        print(f"  当前命令行指定的 --model : {opts.model}")
        print(f"  ckpt 中记录的 model_name   : {ckpt_model_name}")
        print("  请修改 --model 或换用对应模型训练得到的权重。")
        raise RuntimeError("Model name mismatch between opts.model and checkpoint['model_name'].")

    # --------------------------
    # 3) 恢复模型权重
    #    - continue_training=True：严格加载，失败就报错（因为还要恢复 optimizer/scheduler）
    #    - continue_training=False：失败则 shape-filter 加载（适配 num_classes 不同）
    # --------------------------
    # 处理 DP 前缀
    ckpt_state = _align_ckpt_keys_to_model(ckpt_state, model.state_dict())

    try:
        # 严格加载（默认 strict=True）
        model.load_state_dict(ckpt_state)
        print(f"[CKPT] Model weights restored from {opts.ckpt}")
    except RuntimeError:
        # 如果是断点续训，就必须严格一致
        if getattr(opts, "continue_training", False):
            print("[CKPT] load_state_dict 失败：断点续训要求模型结构与 ckpt 完全一致。")
            print(f"  当前 --model: {opts.model}")
            if ckpt_model_name is not None:
                print(f"  ckpt 记录的 model_name: {ckpt_model_name}")
            print("  典型原因：你改了 num_classes / 改了模型结构 / 换了 backbone。")
            raise RuntimeError(
                "Resume training failed: model architecture and checkpoint do not match."
            ) from None

        # 否则当“预训练权重”用：按 shape 过滤加载（解决你 num_classes=3 加载 VOC21 权重的问题）
        print("[CKPT] strict load 失败，尝试按 shape 过滤加载（常见原因：num_classes 不同导致分类头不匹配）。")
        _load_by_shape(model, ckpt_state, verbose=True)
        print(f"[CKPT] Loaded pretrained weights (shape-filter) from {opts.ckpt}")

    # --------------------------
    # 4) 是否继续训练：恢复 optimizer/scheduler/iters/best
    # --------------------------
    if getattr(opts, "continue_training", False):
        if "optimizer_state" in checkpoint and "scheduler_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])

            # 如果外面没传 device，这里自动推断一个（解决你之前 cuda/cpu 混用报错）
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # 关键：把 optimizer 缓存（momentum_buffer 等）迁移到 device
            optimizer_state_to_device(optimizer, device)

        cur_itrs = checkpoint.get("cur_itrs", 0)
        best_score = checkpoint.get("best_score", 0.0)

        if len(train_loader) > 0:
            start_epoch = cur_itrs // len(train_loader)

        print(f"[CKPT] Continue training from epoch {start_epoch}, cur_itrs={cur_itrs}, best_score={best_score:.4f}")
    else:
        print("[CKPT] Only loaded model weights, not optimizer/scheduler state.")

    return model, best_score, cur_itrs, start_epoch



def format_seg_metrics_line(score: dict, use_color: bool = True) -> str:
    """
    把 StreamSegMetrics 的输出 dict 格式化成一行：
    PAcc / mAcc / mIoU / FWIoU
    并可选给标签上色（ANSI）。
    """
    # 从 score 里取值（key 用你现有的）
    pacc = score.get("Overall Acc", 0.0)
    macc = score.get("Mean Acc", 0.0)
    miou = score.get("Mean IoU", 0.0)
    fwiou = score.get("FreqW Acc", 0.0)  # 很多实现里这项就是 FWIoU

    if not use_color:
        return f"PAcc={pacc:.4f}  mAcc={macc:.4f}  mIoU={miou:.4f}  FWIoU={fwiou:.4f}"

    # ANSI 颜色（不同标签不同色）
    C_PACC = "\033[96m"  # 青色
    C_MACC = "\033[92m"  # 绿色
    C_MIOU = "\033[93m"  # 黄色
    C_FWIOU = "\033[95m" # 紫色
    R = "\033[0m"

    return (
        f"{C_PACC}PAcc{R}={pacc:.4f}  "
        f"{C_MACC}mAcc{R}={macc:.4f}  "
        f"{C_MIOU}mIoU{R}={miou:.4f}  "
        f"{C_FWIOU}FWIoU{R}={fwiou:.4f}"
    )


def optimizer_state_to_device(optimizer, device):
    """
    把 optimizer.state 里的所有 Tensor（如 SGD momentum_buffer）迁移到指定 device。
    断点续训常见问题：checkpoint 用 map_location='cpu' 加载后，optimizer buffer 留在 CPU。
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# ====== 1) 新增：torch.load 兼容封装（放在 engine.py 里任意位置，建议在 restore 前）======
def _torch_load_compat(ckpt_path: str, map_location="cpu"):
    """
    兼容 PyTorch 1.x / 2.x 的 torch.load：
    - PyTorch >= 2.6 支持 weights_only 参数（你这里用 weights_only=False 以兼容老 ckpt 里的对象）
    - PyTorch 1.10 不支持 weights_only，会抛 TypeError，需要 fallback
    """
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch 1.x fallback
        return torch.load(ckpt_path, map_location=map_location)
