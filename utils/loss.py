import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss（常用于类别不均衡）：
      FL = alpha * (1 - pt)^gamma * CE
    - pt = exp(-CE) 代表预测为真类的概率（近似）
    - gamma 越大：越强调“难样本”
    - ignore_index：忽略某些像素（VOC常用 255 表示 ignore）

    main.py 中 opts.loss_type == 'focal_loss' 时会用它
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predict, target):
        """
        predict: [B, C, H, W]
        target: [B, H, W]
        """
        # 1. 提取有效像素掩码 (彻底排除 ignore_index 的干扰)
        valid_mask = (target != self.ignore_index)
        
        # 2. 抽取有效像素 -> predict 变为 [N, C], target 变为 [N]
        predict_valid = predict.permute(0, 2, 3, 1)[valid_mask]
        target_valid = target[valid_mask]
        
        # 极端情况防崩溃：如果整张图全是 255，返回 0 梯度
        if target_valid.numel() == 0:
            return predict.sum() * 0.0

        # 3. 对干净的像素计算 CE 和 Focal
        ce_loss = F.cross_entropy(predict_valid, target_valid, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 4. 计算均值时，分母现在是纯净的 N (有效像素个数)
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        
        
class DiceLoss(nn.Module):
    """
    Dice Loss 用于直接优化 IoU（特别适合不规则、样本极度不平衡的遥感分割）。
    """
    # 【修复】：加回 __init__，接收 train.py 传来的 ignore_index
    def __init__(self, ignore_index=255, **kwargs):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, predict, target):
        valid_mask = (target != self.ignore_index)
        predict_valid = predict.permute(0, 2, 3, 1)[valid_mask]
        target_valid = target[valid_mask]
        
        if target_valid.numel() == 0:
            return predict.sum() * 0.0
            
        predict_valid = F.softmax(predict_valid, dim=1)
        target_one_hot = F.one_hot(target_valid, num_classes=predict.shape[1]).float()
        
        intersection = torch.sum(predict_valid * target_one_hot, dim=0)
        union = torch.sum(predict_valid, dim=0) + torch.sum(target_one_hot, dim=0)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        
        return 1.0 - torch.mean(dice)

    
class LandslideDiceLoss(nn.Module):
    """
    Landslide-specific Dice Loss (用于遥感滑坡分割任务)
    该版本只计算滑坡类别的 Dice 系数，且彻底无视 ignore_index 黑边。
    """
    # 【修复】：加回 __init__
    def __init__(self, ignore_index=255, **kwargs):
        super(LandslideDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, predict, target):
        valid_mask = (target != self.ignore_index)
        predict_valid = predict.permute(0, 2, 3, 1)[valid_mask]
        target_valid = target[valid_mask]
        
        if target_valid.numel() == 0:
            return predict.sum() * 0.0
            
        predict_valid = F.softmax(predict_valid, dim=1)
        predict_ls = predict_valid[:, 1] # 仅抽取滑坡类别(index=1) 概率
        target_ls = (target_valid == 1).float() # 滑坡类别标签
        
        intersection = torch.sum(predict_ls * target_ls)
        union = torch.sum(predict_ls) + torch.sum(target_ls)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        
        return 1.0 - dice
    
class BoundaryDiceLoss(nn.Module):
    """
    边缘预测 Loss。由于 M2FToTensor 里已将 edge_target 的 255 强转为 0，
    这里无需再做 valid_mask，直接算即可。
    """
    # 【修复】：加回 __init__，防止 train.py 万一误传参数崩溃
    def __init__(self, ignore_index=255, **kwargs):
        super(BoundaryDiceLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, predict, target):
        # predict: [B, 1, H, W]
        # target: [B, 1, H, W] 
        # 边缘 target 中的 255 已在 M2FToTensor 被转为 0，所以这里直接算
        predict = torch.sigmoid(predict)
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - dice