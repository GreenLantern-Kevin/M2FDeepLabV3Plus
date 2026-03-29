# train.py
# -*- coding: utf-8 -*-
"""
DeepLabV3/DeepLabV3+ 训练脚本（按 epoch 训练，并使用 tqdm 动态进度条显示训练/验证过程）。
"""
import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm  # 用于动态进度条

import network
from utils.engine import (setup_device_and_seed, resolve_num_classes,
                          build_loaders, build_model, build_metrics, build_optimizer, build_scheduler,
                          build_criterion, save_checkpoint, validate, restore_from_checkpoint, format_seg_metrics_line)

from utils.create_exp_folder import create_exp_folder
from utils.seg_metrics_logger import SegMetricsLogger
from utils.loss import FocalLoss, DiceLoss, LandslideDiceLoss, BoundaryDiceLoss

def main(opts):
    # ---------- 1) 补全 num_classes ----------
    # 如果你没有显式传 --num_classes，就按数据集默认：
    resolve_num_classes(opts)

    # ---------- 2) 设备与随机种子 ----------
    device = setup_device_and_seed(opts)
    print(f"Device: {device}")

    # ---------- 3) 数据集与 DataLoader ----------
    train_dst, val_dst, train_loader, val_loader = build_loaders(opts)
    print(f"Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    # ---------- 3.1) 创建本次实验的文件夹结构：run/train/exp*/weights ----------
    #   exp_dir   : 用来放本次实验的训练日志、图表等（后面可以往这里写曲线图、配置文件）
    #   weights_dir : 用来放 latest / best 的权重文件
    exp_dir, weights_dir = create_exp_folder()
    print(f"[Folder] Experiment dir: {exp_dir}")
    print(f"[Folder] Weights dir   : {weights_dir}")
    # ---------- 3.2) 指标记录器：每个 epoch 写入 exp_dir/metrics.csv，并在训练结束后保存曲线图 ----------
    metrics_logger = SegMetricsLogger(exp_dir)

    # 计算总迭代次数（供 PolyLR 使用）= 训练轮数 * 每个 epoch 的 iteration 数
    if len(train_loader) == 0:
        raise RuntimeError("train_loader 为空，检查数据集路径或划分是否正确。")
    opts.total_itrs = opts.epochs * len(train_loader)

    # ---------- 4) 模型 / 指标 / 优化器 / scheduler / loss ----------
    model = build_model(opts)
    metrics = build_metrics(opts)
    optimizer = build_optimizer(opts, model)
    scheduler = build_scheduler(opts, optimizer)

    # 主任务的损失函数（Focal Loss）
    # 实例化所有可能用到的主任务损失函数
    criterion_ce = nn.CrossEntropyLoss(ignore_index=255).to(device)
    criterion_focal = FocalLoss(gamma=2.0, ignore_index=255).to(device) # 这里强制 gamma=2
    criterion_dice = LandslideDiceLoss(ignore_index=255).to(device)
    criterion_boundary = BoundaryDiceLoss(ignore_index=255).to(device) # 【新增】边界损失
    
    # 【新增】：封装一个统一的 criterion 给后期的 validate 函数算 Val Loss 用
    def criterion(preds, labels):
        if opts.loss_type == "cross_entropy":
            return criterion_ce(preds, labels)
        elif opts.loss_type == "focal_loss":
            return criterion_focal(preds, labels)
        elif opts.loss_type == "dice_loss":
            return criterion_dice(preds, labels)
        elif opts.loss_type == "focal_dice":
            return 0.7 * criterion_focal(preds, labels) + 0.3 * criterion_dice(preds, labels)
        elif opts.loss_type == "focal_dice_boundary": # 【新增】三合一顶级损失
            return 0.6 * criterion_focal(preds, labels) + 0.3 * criterion_dice(preds, labels) + 0.1 * criterion_boundary(preds, labels)
        return criterion_ce(preds, labels) # 兜底

    # 辅任务的损失函数：边缘检测（二元交叉熵）
    # 注意：需要将其放到设备上
    criterion_edge = nn.BCEWithLogitsLoss().to(device)

    # ---------- 5) checkpoint 恢复与“外科手术式”预训练加载 ----------
    best_score, cur_itrs, start_epoch = 0.0, 0, 0
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # 兼容 PyTorch 2.6+ 的安全加载机制
        try:
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        except TypeError:
            # 兼容老版本 PyTorch (不支持 weights_only 参数)
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        # 兼容不同的保存格式
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        
        # 去除可能存在的 module. 前缀
        old_state_dict = {}
        for k, v in state_dict.items():
            old_state_dict[k.replace("module.", "")] = v

        if opts.continue_training:
            # 如果是意外中断的续训，直接严格加载我们自己跑出来的双流模型权重
            model.load_state_dict(old_state_dict, strict=True)
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint.get("cur_itrs", 0)
            best_score = checkpoint.get("best_score", 0.0)
            print(f"[!] 成功恢复训练，来自: {opts.ckpt}")
        else:
            # 【全新进阶版：满血移植】将你跑出的单模态王者权重，全部注入到多模态网络中！
            print(f"[!] 检测到单模态预训练权重，正在进行【骨干 + 解码器】满血移植...")
            new_state_dict = model.state_dict()
            transplant_count = 0
            
            for old_k, v in old_state_dict.items():
                old_k = old_k.replace("module.", "") # 容错处理
                
                # 1. 转移 Backbone 部分到双流 (RGB 和 DEM 都继承它)
                if old_k.startswith("backbone."):
                    base_k = old_k[len("backbone."):]
                    rgb_k = "backbone.rgb_backbone." + base_k
                    dem_k = "backbone.dem_backbone." + base_k
                    
                    if rgb_k in new_state_dict and new_state_dict[rgb_k].shape == v.shape:
                        new_state_dict[rgb_k] = v
                        transplant_count += 1
                    if dem_k in new_state_dict and new_state_dict[dem_k].shape == v.shape:
                        new_state_dict[dem_k] = v
                        transplant_count += 1
                        
                # 2. 转移 Classifier (ASPP, Decoder 等) 核心部分！
                elif old_k.startswith("classifier."):
                    if old_k in new_state_dict and new_state_dict[old_k].shape == v.shape:
                        new_state_dict[old_k] = v
                        transplant_count += 1

            # 使用 strict=False 加载，因为 M2F 独有的特征融合模块是新的，需要随机初始化
            model.load_state_dict(new_state_dict, strict=False)
            print(f"[+] 满血权重移植完成！共成功映射 {transplant_count} 个参数张量！")
            print("[+] 现在双流模型已经继承了单流 0.77+ 的全部解码实力，仅融合模块从零微调。")
    else:
        print("[!] 未提供有效的 --ckpt，模型将完全从零开始训练。")

    # ---------- 6) DataParallel + to(device) ----------
    model = nn.DataParallel(model)
    model.to(device)

    # --------------------------
    # 7) 训练主循环（按 epoch）
    # --------------------------
    best_epoch = start_epoch  # 记录最优 mIoU 出现的 epoch（1-based 会用 epoch+1）
    best_ckpt_path = None  # 记录最优权重的保存路径
    for epoch in range(start_epoch, opts.epochs):
        # 每个epoch开始前，把模型切到“训练模式”（启用 BN、Dropout 的训练行为）
        model.train()

        # 统计本 epoch 的训练信息
        epoch_loss = 0.0  # 累计当前 epoch 中所有 batch 的 loss 之和
        correct_pixels = 0  # 累计预测正确的像素数（忽略 label=255 的像素）
        total_pixels = 0  # 累计参与统计的像素总数（label != 255）
        num_batches = 0  # 当前 epoch 已经跑了多少个 batch

        # tqdm 进度条显示当前 epoch 的训练进度
        # iterable=train_loader：迭代的是训练集 DataLoader
        # total=len(train_loader)：进度条总步数 = 该 epoch 中 batch 数
        # desc：进度条左侧标题，例如 "epoch 1/50"
        # ncols：进度条在控制台的宽度（可以根据自己终端宽度调节）
        pbar = tqdm(
            iterable=train_loader,
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{opts.epochs}",  # 头上的 “epoch 1/20” 样式
            ncols=100,
        )

        # 一个 epoch 内按 batch 遍历训练集
        # 注意：这里的数据解包增加了 dems (地形图), edges (边缘标签), has_dem (有效标志)
        for images, dems, labels, edges, has_dem in pbar:
            cur_itrs += 1  # 全局迭代步数 +1（方便保存 ckpt 和恢复训练）
            num_batches += 1  # 该 epoch 内的 batch 计数 +1

            # 把数据搬到对应设备（GPU / CPU），并设定数据类型
            images = images.to(device, dtype=torch.float32)  # 图像用 float32
            labels = labels.to(device, dtype=torch.long)  # 标签是类别 id，用 long
            dems = dems.to(device, dtype=torch.float32)
            edges = edges.to(device, dtype=torch.float32)  # 边缘通常用 BCE Loss，需要 float
            has_dem = has_dem.to(device, dtype=torch.bool)

            optimizer.zero_grad()  # 清空上一轮迭代的梯度

            # --------- 前向传播 ---------
            # 修改点 1：模型现在需要接收两个输入
            # 修改点 2：模型返回的是一个字典，包含多个输出，以支持多任务
            outputs = model(images, dems)  # 前向传播，得到预测 logits，形状 [N, C, H, W]

            pred_mask = outputs['seg']  # 主分支：预测的滑坡 mask
            pred_edge = outputs['edge']  # 辅分支：预测的滑坡边界
            feat_fake = outputs['feat_fake']  # RGB 提取的“幻觉地形特征”
            feat_real = outputs['feat_real']  # 真实的“地形特征”
            
            # 【新增：提取辅分支并上采样到原图尺寸】
            pred_aux = None
            if 'aux' in outputs:
                pred_aux = nn.functional.interpolate(
                    outputs['aux'], 
                    size=(images.shape[2], images.shape[3]), 
                    mode='bilinear', align_corners=False
                )

            # --------- 损失计算 ---------
            # 1. 主任务：根据传入的 loss_type 动态选择
            if opts.loss_type == "cross_entropy":
                loss_seg = criterion_ce(pred_mask, labels)
            elif opts.loss_type == "focal_loss":
                loss_seg = criterion_focal(pred_mask, labels)
            elif opts.loss_type == "dice_loss":
                loss_seg = criterion_dice(pred_mask, labels)
            elif opts.loss_type == "focal_dice":
                loss_focal = criterion_focal(pred_mask, labels)
                loss_dice = criterion_dice(pred_mask, labels)
                loss_seg = 0.7 * loss_focal + 0.3 * loss_dice
            elif opts.loss_type == "focal_dice_boundary": # 【新增】
                loss_focal = criterion_focal(pred_mask, labels)
                loss_dice = criterion_dice(pred_mask, labels)
                loss_boundary = criterion_boundary(pred_mask, labels)
                loss_seg = 0.6 * loss_focal + 0.3 * loss_dice + 0.1 * loss_boundary
            else:
                raise NotImplementedError(f"不支持的 loss_type: {opts.loss_type}")
                
            # 【新增：叠加上 Aux Loss，权重系数为 mmseg 的经典值 0.4】
            if pred_aux is not None:
                if "cross_entropy" in opts.loss_type:
                    loss_aux = criterion_ce(pred_aux, labels)
                else:
                    loss_aux = criterion_focal(pred_aux, labels) # 如果用 Focal 体系，辅分类器也用 Focal
                loss_seg = loss_seg + 0.4 * loss_aux   # 核心：按 mmseg 叠加 0.4 权重

            # 2. 边缘检测损失 (辅任务，使用 BCEWithLogitsLoss 等)
            loss_edge = criterion_edge(pred_edge, edges)

            # 3. 特征幻觉一致性损失 (仅在真实 DEM 存在时计算)
            # 筛选出有真实 DEM 的样本进行 MSE 计算
            if has_dem.any():
                loss_consist = nn.functional.mse_loss(
                    feat_fake[has_dem],
                    feat_real[has_dem].detach()  # detach 很关键，防止梯度传导到真实 DEM 编码器
                )
            else:
                loss_consist = torch.tensor(0.0).to(device)

            # 4. 动态加权总损失
            # 你可以在 argparse 中设置 alpha 和 beta 来调节权重
            loss = loss_seg + opts.alpha * loss_edge + opts.beta * loss_consist  # 计算当前 batch 的损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 根据梯度更新模型参数

            # ------- 统计训练 loss（用于进度条和 epoch 平均 loss） -------
            batch_loss = float(loss.detach().cpu().numpy())  # 转成 float，避免张量残留在图里
            epoch_loss += batch_loss  # 累加当前 batch loss

            # ------- 统计像素级准确率（忽略 label=255 的像素）-------
            # 这里在 no_grad 下做统计，避免额外梯度开销
            with torch.no_grad():
                # 取每个像素预测类别：在通道维上取 argmax，得到 [N, H, W]
                preds = pred_mask.detach().max(dim=1)[1]

                # 有效像素掩码：label=255 的像素是 ignore，不参与准确率计算
                valid_mask = (labels != 255)

                # 统计这一个 batch 有多少有效像素
                total_pixels += valid_mask.sum().item()
                # 在有效像素位置上预测正确的像素数
                correct_pixels += ((preds == labels) & valid_mask).sum().item()

            # 当前 batch 结束后，计算“到目前为止”的平均 loss 和训练像素准确率
            avg_loss = epoch_loss / num_batches
            train_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

            # 更新 tqdm 尾部显示：loss / Acc / Size
            # 类似你 ViT 训练时控制台的动态显示效果
            pbar.set_postfix({
                "loss": f"{avg_loss:.3f}",
                "PAcc": f"{train_acc:.3f}",
                "Size": f"{images.shape[-1]}",  # 图像尺寸：这里假设裁剪成方形（H=W），直接取最后一维即可
            })

            # 学习率调度器仍然按 iteration 更新（每个 batch 调一次）
            scheduler.step()

        # ------- 每个 epoch 结束后，在验证集上做一次完整 validate -------
        val_score = validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=metrics,
            criterion=criterion,  # 把 loss 函数传进去，这样验证时也能算 val loss
            epoch=epoch + 1,
            total_epochs=opts.epochs,
        )
        
        # 【替换原本的打印逻辑，提取你最关心的 6 个核心指标并高亮输出】
        mIoU = val_score.get("Mean IoU", 0.0)
        ls_iou = val_score.get("Class IoU", {}).get(1, 0.0)
        ls_prec = val_score.get("Class Precision", {}).get(1, 0.0)
        ls_rec = val_score.get("Class Recall", {}).get(1, 0.0)
        ls_f1 = val_score.get("Class F1", {}).get(1, 0.0)
        ls_mcc = val_score.get("Class MCC", {}).get(1, 0.0)

        print(f"\n[Validation] Epoch {epoch + 1} specific metrics:")
        print(f" -> MIoU:          {mIoU:.4f}")
        print(f" -> Landslide IoU: {ls_iou:.4f}")
        print(f" -> Precision:     {ls_prec:.4f}")
        print(f" -> Recall:        {ls_rec:.4f}")
        print(f" -> F1-Score:      {ls_f1:.4f}")
        print(f" -> MCC:           {ls_mcc:.4f}\n")


        # ---------- 记录本 epoch 的 train/val 指标到 CSV ----------
        # 训练平均 loss（按 epoch）
        train_loss_epoch = epoch_loss / max(1, num_batches)
        # 训练像素准确率（你在循环里一直在累计 correct_pixels/total_pixels，这里直接复用）
        train_pacc_epoch = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        # 当前学习率：建议记录 classifier/head 的 lr（你的 optimizer 第 2 组是 classifier）
        # 如果你后面改了 param_groups 顺序，这里相应调整即可
        lr_now = optimizer.param_groups[-1]["lr"]

        # 验证平均 loss：建议从 val_score 里拿（需要 validate() 把 Loss 写进去）
        val_loss_epoch = float(val_score.get("Loss", 0.0))

        metrics_logger.log_epoch(
            epoch=epoch + 1,
            lr=lr_now,
            train_loss=train_loss_epoch,
            train_pacc=train_pacc_epoch,
            val_loss=val_loss_epoch,
            val_score=val_score,
        )


        # ------- 保存 latest ckpt（“当前进度”的快照）到当前实验的 weights 下 -------
        latest_path = os.path.join(weights_dir,f"last_{opts.model}_os{opts.output_stride}_num{opts.num_classes}.pth")
        save_checkpoint(latest_path, model, optimizer, scheduler, cur_itrs, best_score, model_name=opts.model)

        # ------- 若当前验证 Mean IoU 刷新历史最优，则保存 best ckpt 到 weights 下 -------
        if val_score["Mean IoU"] > best_score:
            best_score = val_score["Mean IoU"]
            best_epoch = epoch + 1  # 记录发生在第几个 epoch（1-based）
            best_ckpt_path = os.path.join(
                weights_dir,
                f"best_{opts.model}_os{opts.output_stride}_num{opts.num_classes}.pth",
            )
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, cur_itrs, best_score, model_name=opts.model)

    # 所有 epoch 跑完
    print("Training finished.")

    # ---------- 训练结束：保存曲线图到 exp_dir ----------
    metrics_logger.save_plots()
    print(f"[Plot] Saved curves to: {exp_dir} (loss_curve.png / pacc_curve.png / val_metrics_curve.png)")

    # 如果在本次训练过程中出现过“更好的 mIoU”，打印最优轮次和权重路径
    if best_ckpt_path is not None:
        print(f"[Best] Best mIoU = {best_score:.4f} at epoch {best_epoch}")
        print(f"[Best] Best weights path: {best_ckpt_path}")
    else:
        # 没有超过初始 best_score（例如从一个 best ckpt 续训，但这次训练没刷新）
        print(f"[Best] No new best mIoU in this run. Current best_score = {best_score:.4f}")
        print(f"[Best] Check previous run's best checkpoint if needed.")


if __name__ == "__main__":
    # ---------------- 模型相关 ----------------
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower()
        and not (name.startswith("__") or name.startswith("_"))
        and callable(network.modeling.__dict__[name]))

    # 这里定义“训练脚本”的命令行参数
    parser = argparse.ArgumentParser()
    # ---------------- 训练超参数（按 epoch） ----------------
    parser.add_argument("--data_root", type=str, default="BijieLandslide", help="数据集根目录路径")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数（含背景）。不写，写错都会报错")
    parser.add_argument("--model", type=str, default="m2f_deeplabv3plus_xception16", choices=available_models,
                        help="模型名称，对应 network.modeling 里的工厂函数")
    parser.add_argument("--ckpt", type=str,default="weight/best_deeplabv3plus_xception16_os8_num2.pth", help="用于加载预训练/中断权重")
    parser.add_argument("--continue_training", action="store_true", default=False, help="是否从 ckpt 继续训练（恢复 optimizer/scheduler/cur_itrs）")
    parser.add_argument("--loss_type", type=str, default="focal_dice_boundary", choices=["cross_entropy", "focal_loss", "dice_loss", "focal_dice", "focal_dice_boundary"], help="损失函数类型")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减系数（L2 正则），默认 1e-4")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16], help="编码器输出特征的下采样倍数：8 或 16")
    parser.add_argument("--separable_conv", action="store_true", default=False, help="是否在 ASPP 和 decoder 中使用 depthwise separable conv")
    parser.add_argument("--epochs", type=int, default=200, help="训练总轮数（epoch 数），默认 50")
    parser.add_argument("--lr", type=float, default=0.01, help="基础学习率（backbone 用 0.1*lr，classifier 用 lr）")
    parser.add_argument("--lr_policy", type=str, default="poly",  choices=["poly", "step"], help="学习率策略：poly / step")
    parser.add_argument("--step_size", type=int, default=1000, help="当 lr_policy=step 时，每多少个 iteration 衰减一次学习率")
    parser.add_argument("--batch_size", type=int, default=16, help="训练 batch size")
    parser.add_argument("--val_batch_size", type=int, default=1, help="验证 batch size（VOC + 不 crop 时会被强制改为 1）")
    parser.add_argument("--crop_size", type=int, default=256, help="训练/验证裁剪尺寸（默认 513）")
    parser.add_argument("--crop_val", action="store_true", default=False, help="验证时是否也做 crop（默认 False，为原图评估）")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID（写入 CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--random_seed", type=int, default=42,  help="随机种子，方便复现实验")
    # ---------------- M-E DeepLab 特有超参数 ----------------
    parser.add_argument("--alpha", type=float, default=0.1, help="边缘损失 (Edge Loss) 的权重")
    parser.add_argument("--beta", type=float, default=0.05, help="幻觉一致性损失 (Consistency Loss) 的权重")
    opt = parser.parse_args()
    main(opt)


