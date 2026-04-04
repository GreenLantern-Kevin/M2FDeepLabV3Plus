import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class _SimpleSegmentationModel(nn.Module):
    """
    语义分割模型的通用 wrapper（本工程中用于 DeepLabV3+）

    流程：
      1) features = backbone(x)  -> 返回 dict，例如 {'out': ..., 'low_level': ...}
      2) x = classifier(features) -> head 输出 logits
      3) logits 上采样回输入大小，得到最终分割输出
    """
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # features 包含 'out', 'low_level', 'aux'
        features = self.backbone(x)
        res = self.classifier(features)
        
        # 【修改点：支持返回带有 aux 的字典】
        if isinstance(res, dict):
            res['seg'] = F.interpolate(res['seg'], size=input_shape, mode='bilinear', align_corners=False)
            if 'aux' in res:
                res['aux'] = F.interpolate(res['aux'], size=input_shape, mode='bilinear', align_corners=False)
            return res
        else:
            x = F.interpolate(res, size=input_shape, mode='bilinear', align_corners=False)
            return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    从 backbone 中按“子模块名”提取中间层输出，返回一个 OrderedDict。

    使用方式（在 modeling.py 中配置 return_layers）：
      - ResNet:   {'layer4': 'out', 'layer1': 'low_level'}
      - Xception: {'conv4': 'out',  'block1': 'low_level'}
      - MobileNet:{'high_level_features': 'out', 'low_level_features': 'low_level'}

    注意：
      - 只能抓 model 的“直接子模块”（named_children）名称。
      - return_layers 必须是 model.named_children() 的子集，否则报错。
      - 会在收集到所有目标层后提前 break，避免不必要的 forward 计算。
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class _MEDeepLabModelWrapper(nn.Module):
    """
    M-E DeepLab 的外壳 Wrapper，支持双模态输入 (RGB + DEM)
    """

    def __init__(self, dual_backbone, classifier):
        super(_MEDeepLabModelWrapper, self).__init__()
        self.backbone = dual_backbone
        self.classifier = classifier

    def forward(self, x_rgb, x_dem):
        input_shape = x_rgb.shape[-2:]

        # 1. 经过异构双流 Backbone，返回融合特征和用于计算 Loss 的幻觉/真实特征
        features = self.backbone(x_rgb, x_dem)

        # 2. 经过带有边缘分支的 Decoder
        head_outs = self.classifier(features)

        # 3. 将主任务和辅助任务的预测结果上采样回原图尺寸
        seg = F.interpolate(head_outs['seg'], size=input_shape, mode='bilinear', align_corners=False)
        edge = F.interpolate(head_outs['edge'], size=input_shape, mode='bilinear', align_corners=False)

        return {
            'seg': seg,  # [B, num_classes, H, W] 滑坡预测
            'edge': edge,  # [B, 1, H, W] 边缘预测
            'feat_fake': features['feat_fake'],  # 用于算一致性 Loss
            'feat_real': features['feat_real']  # 用于算一致性 Loss
        }