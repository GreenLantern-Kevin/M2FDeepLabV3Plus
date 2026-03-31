import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    """
    Focal Loss（常用于类别不均衡）：
      FL = alpha * (1 - pt)^gamma * CE
    - pt = exp(-CE) 代表预测为真类的概率（近似）
    - gamma 越大：越强调“难样本”
    - ignore_index：忽略某些像素（VOC常用 255 表示 ignore）

    main.py 中 opts.loss_type == 'focal_loss' 时会用它
    """
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) logits
        targets: (N, H, W) 标签（每个像素一个类别 id）
        """
        # 逐像素 CE，先不做 mean/sum（reduction='none'）
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)

        # pt = exp(-CE)
        pt = torch.exp(-ce_loss)

        # focal reweight：难样本（pt 小）权重更大
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        # size_average=True：取 mean；否则 sum
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
        
class DiceLoss(nn.Module):
    """
    Dice Loss 用于直接优化 IoU（特别适合不规则、样本极度不平衡的遥感分割）。
    """
    def __init__(self, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth # 平滑系数，防止分母为 0

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] 网络的原始输出 (未经过 softmax)
        targets: [B, H, W] 真实的类别标签
        """
        num_classes = logits.shape[1]
        
        # 1. 将 logits 转换为概率
        probs = F.softmax(logits, dim=1)
        
        # 2. 生成过滤掉 ignore_index (255) 的掩码
        valid_mask = (targets != self.ignore_index)
        
        total_dice_loss = 0.0
        
        # 3. 遍历每个类别计算 Dice
        for c in range(num_classes):
            # 获取当前类的预测概率
            prob_c = probs[:, c, :, :]
            # 获取当前类的真实 One-Hot 标签
            target_c = (targets == c).float()
            
            # 仅在有效像素区域计算 (过滤 255)
            prob_c = prob_c[valid_mask]
            target_c = target_c[valid_mask]
            
            # Dice 系数计算: 2 * |A ∩ B| / (|A| + |B|)
            intersection = (prob_c * target_c).sum()
            union = prob_c.sum() + target_c.sum()
            
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            
            # Loss 是 1 - Dice，加到总 loss 中
            total_dice_loss += (1.0 - dice_c)
            
        # 返回所有类别的平均 Dice Loss
        return total_dice_loss / num_classes

    
class LandslideDiceLoss(nn.Module):
    """
    Landslide-specific Dice Loss (用于遥感滑坡分割任务)
    该版本只计算滑坡类别的 Dice 系数，忽略背景（通常是类别0）的影响。
    """
    def __init__(self, ignore_index=255, smooth=1.0):
        super(LandslideDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth  # 平滑系数，防止分母为 0

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] 网络的原始输出 (未经过 softmax)
        targets: [B, H, W] 真实的类别标签
        """
        num_classes = logits.shape[1]
        
        # 1. 将 logits 转换为概率
        probs = F.softmax(logits, dim=1)
        
        # 2. 生成过滤掉 ignore_index (255) 的掩码
        valid_mask = (targets != self.ignore_index)
        
        # 只选择滑坡类别进行计算，这里假设滑坡类别为1
        landslide_class = 1
        
        # 3. 获取滑坡类别的预测概率和真实标签
        prob_landslide = probs[:, landslide_class, :, :]
        target_landslide = (targets == landslide_class).float()
        
        # 4. 过滤掉 ignore_index 的像素
        prob_landslide = prob_landslide[valid_mask]
        target_landslide = target_landslide[valid_mask]
        
        # 5. Dice 系数计算: 2 * |A ∩ B| / (|A| + |B|)
        intersection = (prob_landslide * target_landslide).sum()
        union = prob_landslide.sum() + target_landslide.sum()
        
        dice_landslide = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Loss 是 1 - Dice
        return 1.0 - dice_landslide
    
class BoundaryDiceLoss(nn.Module):
    """
    真正的 Boundary Dice Loss (边界戴斯损失)
    利用 PyTorch 的 MaxPool2d 模拟形态学膨胀，动态提取预测和标签的边界线条，然后计算线条的 Dice。
    """
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. 获取滑坡类的预测概率和 One-hot 标签，并增加 Channel 维度以便做 Pooling
        probs = torch.softmax(logits, dim=1)
        prob = probs[:, 1, :, :].unsqueeze(1)      # [B, 1, H, W]
        target = (targets == 1).float().unsqueeze(1) # [B, 1, H, W]
        valid_mask = (targets != self.ignore_index).unsqueeze(1)

        # 2. 动态提取边界 (膨胀图 - 原图 = 边界线条)
        # kernel_size=3, padding=1 相当于向外膨胀 1 个像素
        prob_dilated = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
        prob_boundary = prob_dilated - prob
        
        target_dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        target_boundary = target_dilated - target

        # 3. 过滤 ignore_index
        prob_boundary = prob_boundary[valid_mask]
        target_boundary = target_boundary[valid_mask]

        # 4. 计算边界的 Dice 系数
        intersection = (prob_boundary * target_boundary).sum()
        union = prob_boundary.sum() + target_boundary.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice