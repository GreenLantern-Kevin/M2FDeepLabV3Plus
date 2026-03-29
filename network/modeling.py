from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3Plus
from .backbone import resnet, mobilenetv2, xception

from torchvision.models.resnet import resnet18 # 引入轻量级ResNet作为DEM编码器
from .utils import _MEDeepLabModelWrapper # 引入刚才写的外壳
from ._deeplab import MEDeepLabHeadV3Plus # 引入刚才写的双头Decoder
import torch
import torch.nn as nn

def _segm_resnet(backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    构建 ResNet backbone 的 DeepLabV3+。

    output_stride(OS) 的实现要点：
    - OS=16：只取消 layer4 的 stride=2（用 dilation 替代） -> replace_stride_with_dilation=[False, False, True]
    - OS=8 ：取消 layer3/layer4 的 stride=2（用 dilation 替代） -> replace_stride_with_dilation=[False, True, True]

    同时根据 OS 切换 ASPP 的 atrous rates：
    - OS=16： [6, 12, 18]
    - OS=8 ： [12, 24, 36]
    """
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    # 构建 backbone（resnet50/resnet101），并配置 dilation 替换策略
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    # ResNet layer4 输出通道固定 2048；layer1 输出通道 256（Bottleneck expansion=4）
    inplanes = 2048
    low_level_planes = 256

    # DeepLabV3+：需要 out + low_level
    return_layers = {"layer4": "out", "layer1": "low_level"}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    # 让 backbone 输出变成 dict：{'out': ..., 'low_level': ...}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3Plus(backbone, classifier)
    return model


def _segm_xception(backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    构建基于 Xception 主干的 DeepLabV3+ 模型。

    支持三种字符串：
        - 'xception'   : 默认等价于使用完整 Middle flow 的 'xception16'
        - 'xception16' : Middle flow 使用 16 个 block（论文中较深的版本）
        - 'xception8'  : Middle flow 使用 8 个 block（相对轻量版）

    参数：
        backbone_name: 见上
        num_classes:   类别数（含背景）
        output_stride: 输出步长，8 或 16
        pretrained_backbone: 是否加载（ImageNet / VOC）预训练的 Xception 主干
    """

    # 1) ASPP 的空洞率设置，和 ResNet / MobileNet 版本保持一致
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:  # output_stride == 16
        aspp_dilate = [6, 12, 18]

    # 2) 根据 backbone_name 选择具体的 Xception 变体
    #    - 'xception' / 'xception16' -> variant='xception16'
    #    - 'xception8'               -> variant='xception8'
    if backbone_name in ("xception", "xception16"):
        variant = "xception16"
    elif backbone_name == "xception8":
        variant = "xception8"
    else:
        raise ValueError(f"Unknown xception backbone_name: {backbone_name}")

    # 构建 Xception 主干（封装后 forward 返回 dict：{"out", "low_level"}）
    backbone = xception.XceptionBackbone(
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        variant=variant,
    )

    # 根据 Xception 实现：
    #   - high-level 特征：通道数 2048
    #   - low-level 特征：通道数 256
    inplanes = 2048
    low_level_planes = 256

    # 3) DeepLabV3+ 的 head（ASPP + Decoder）
    classifier = DeepLabHeadV3Plus(
        in_channels=inplanes,
        low_level_channels=low_level_planes,
        num_classes=num_classes,
        aspp_dilate=aspp_dilate,
    )

    # 4) 封装成统一的 DeepLabV3Plus 模型外壳
    # 注意：这里不再用 IntermediateLayerGetter，
    # 因为 XceptionBackbone 自己已经返回 dict 了
    model = DeepLabV3Plus(backbone, classifier)
    return model


def _segm_mobilenet(backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    构建 MobileNetV2 backbone 的 DeepLabV3+。

    mobilenet_v2 内部支持 output_stride：
    - 到达目标 OS 后停止继续下采样（stride->1），改用 dilation 累积。
    """
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(
        pretrained=pretrained_backbone,
        output_stride=output_stride,
    )

    # 关键技巧：把 features 切成 low/high 两段，供 V3+ decoder 使用
    # low_level_features: features[0:4]  （一般是 stride=4 左右的浅层细节）
    # high_level_features: features[4:-1]（更深层语义）
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    # mobilenetv2 的 high_level 最终输出通道 320，low_level 输出通道 24
    inplanes = 320
    low_level_planes = 24

    return_layers = {"high_level_features": "out", "low_level_features": "low_level"}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3Plus(backbone, classifier)
    return model


def _load_model(backbone, num_classes, output_stride, pretrained_backbone):
    """
    统一入口：根据 backbone 字符串选择不同的 DeepLabV3+ 组装函数。
    已支持：
        - 'resnet50' / 'resnet101'
        - 'mobilenetv2'
        - 'xception'   / 'xception16'（完整 Middle flow）
        - 'xception8'  （轻量 Middle flow）
    """
    if backbone == "mobilenetv2":
        model = _segm_mobilenet(
            backbone, num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone.startswith("resnet"):
        model = _segm_resnet(
            backbone, num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone in ("xception", "xception16", "xception8"):
        model = _segm_xception(
            backbone, num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        raise NotImplementedError(f"Backbone {backbone} is not supported.")
    return model


# --------------------------
# 只保留 Deeplab v3+ 工厂函数
# --------------------------
def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model(
        "resnet50", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model(
        "resnet101", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model(
        "mobilenetv2", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_xception8(num_classes=21, output_stride=8, pretrained_backbone=False):
    """
    Xception 轻量版（middle_blocks = 8）对应的 DeepLabV3+。
    """
    return _load_model(
        "xception8", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_xception16(num_classes=21, output_stride=8, pretrained_backbone=False):
    """
    Xception 完整版（middle_blocks = 16）对应的 DeepLabV3+。
    """
    return _load_model(
        "xception16", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


class FeatureHallucinationModule(nn.Module):
    """【创新点 2：特征幻觉模块 FHM】从 RGB 特征映射到地形特征"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.map(x)


class ModalityElasticFusion(nn.Module):
    """【创新点 3 升级版：带 SE 注意力的模态弹性融合 MEF】动态筛选 RGB 和 DEM 的有效特征"""

    def __init__(self, rgb_ch, topo_ch, out_ch):
        super().__init__()
        in_ch = rgb_ch + topo_ch
        
        # 1. 基础特征降维融合
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 2. SE (Squeeze-and-Excitation) 通道注意力机制
        # 作用：让网络动态学习，如果当前区域 RGB 更可靠就放大 RGB 通道的权重，反之放大 DEM 权重
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局空间池化
            nn.Conv2d(out_ch, out_ch // 16, 1, bias=False), # 降维，默认缩放比16
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 16, out_ch, 1, bias=False), # 升维回 out_ch
            nn.Sigmoid() # 归一化到 0~1 之间作为权重
        )

    def forward(self, feat_rgb, feat_topo):
        # 拼接
        x = torch.cat([feat_rgb, feat_topo], dim=1)
        # 融合
        x = self.fuse(x)
        # 计算注意力权重
        attn_weight = self.se(x)
        # 将权重乘回特征图 (Broadcast)
        return x * attn_weight


class MEDeepLabDualStreamBackbone(nn.Module):
    """【创新点 1：异构双流骨干】包含 RGB 与 DEM 编码器，并实现动态路由"""
    # 修复点：在这里增加 rgb_ch 和 dem_ch 作为参数，以便兼容 ResNet18(512) 和 Xception16(2048)

    def __init__(self, rgb_backbone, dem_backbone, rgb_ch=2048, dem_ch=512):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.dem_backbone = dem_backbone

        # 使用传入的通道数，动态构建幻觉模块和融合模块
        self.fhm = FeatureHallucinationModule(rgb_ch, dem_ch)

        # 注意：无论 DEM 通道是 512 还是 2048，融合后统一降维到 2048，以便对接后续标准的 ASPP 模块
        self.mef = ModalityElasticFusion(rgb_ch, dem_ch, out_ch=2048)

    def forward(self, x_rgb, x_dem):
        # 1. RGB 提取
        rgb_feats = self.rgb_backbone(x_rgb)
        feat_rgb_high = rgb_feats['out']
        feat_rgb_aux = rgb_feats.get('aux', None)  # 【新增】获取 layer3 特征
        feat_rgb_low = rgb_feats['low_level']

        # 2. 生成幻觉地形特征(尺寸：[B, 512, 32, 32])
        feat_fake = self.fhm(feat_rgb_high)

        # 3. 动态判断 DEM 是否有效 (依据：如果是空数据占位符，sum通常极小)
        B = x_rgb.size(0)
        has_dem = (x_dem.abs().reshape(B, -1).sum(dim=1) > 1e-4)

        # 提前初始化真实特征的容器（全0），强制与 fake 尺寸一致
        feat_real = torch.zeros_like(feat_fake)

        if has_dem.any():
            # 只有当 Batch 中有真实 DEM 时，才进行前向传播以节省算力
            dem_feats = self.dem_backbone(x_dem)
            out_real = dem_feats['out'] # 原始尺寸：[B, 512, 8, 8]

            # 【核心修复】：强制将 DEM 分支的小特征图上采样，对齐 RGB 分支的大特征图
            if out_real.shape[2:] != feat_fake.shape[2:]:
                import torch.nn.functional as F
                out_real = F.interpolate(
                    out_real, 
                    size=feat_fake.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 将上采样后的真实特征覆盖进去
            feat_real = out_real

        # 4. 模态路由：有真实DEM用真实的，没有就用幻觉的
        # has_dem: [B] -> 变为 [B, 1, 1, 1] 以便 broadcast
        mask = has_dem.reshape(B, 1, 1, 1).to(feat_fake.device)
        feat_topo = torch.where(mask, feat_real, feat_fake)

        # 5. 融合特征
        feat_fused = self.mef(feat_rgb_high, feat_topo)

        return {
            'out': feat_fused,  # 送给 ASPP 的主特征
            'aux': feat_rgb_aux,         # 【新增】传出 aux
            'low_level': feat_rgb_low,  # 送给 Decoder 的低层细节
            'feat_fake': feat_fake,  # 用于算一致性 Loss
            'feat_real': feat_real  # 用于算一致性 Loss
        }


def me_deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """
    暴露给外部的组装函数：构建 M-E DeepLab (基于 ResNet50 + ResNet18)
    """
    replace_stride_with_dilation = [False, True, True] if output_stride == 8 else [False, False, True]
    aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]

    # 1. 组装 RGB Stream (ResNet50)
    from .backbone import resnet
    rgb_net = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation
    )
    rgb_backbone = IntermediateLayerGetter(rgb_net, return_layers={"layer4": "out", "layer1": "low_level"})

    # 2. 组装 DEM Stream (ResNet18 - 轻量级，不要dilation，纯提特征)
    dem_net = resnet18(pretrained=pretrained_backbone)
    # 强制修改其 conv1 以接收 3 通道 (坡度,坡向,高程 或 复制3次)
    dem_backbone = IntermediateLayerGetter(dem_net, return_layers={"layer4": "out"})

    # 3. 组装双流 Backbone
    dual_backbone = MEDeepLabDualStreamBackbone(rgb_backbone, dem_backbone)

    # 4. 组装 Decoder (这里 in_channels 为 2048，因为 MEF 融合后输出 2048)
    classifier = MEDeepLabHeadV3Plus(
        in_channels=2048,
        low_level_channels=256,
        num_classes=num_classes,
        aspp_dilate=aspp_dilate
    )

    # 5. 套上外壳
    model = _MEDeepLabModelWrapper(dual_backbone, classifier)
    return model


def m2f_deeplabv3plus_xception16(num_classes=21, output_stride=8, pretrained_backbone=True):
    """
    顶配版 M2F-DeepLab: 双 Xception16 骨干网络
    """
    from .backbone import xception

    # 1. 组装 RGB Stream (Xception16)
    rgb_net = xception.XceptionBackbone(
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        variant='xception16'
    )
    # XceptionBackbone 默认直接返回 dict {"out", "low_level"}
    rgb_backbone = rgb_net

    # 2. 组装 DEM Stream (Xception16)
    dem_net = xception.XceptionBackbone(
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        variant='xception16'
    )
    dem_backbone = dem_net

    # 3. 组装 M2F 双流 Backbone
    # Xception16 的 out 通道是 2048
    dual_backbone = MEDeepLabDualStreamBackbone(rgb_backbone, dem_backbone, rgb_ch=2048, dem_ch=2048)

    # 4. 组装带有边缘感知的双分支 Decoder
    # 找到这行代码并注释掉：
    # aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
    
    # 替换为这个小尺寸专用的 ASPP 感受野：
    aspp_dilate = [4, 8, 12]
    classifier = MEDeepLabHeadV3Plus(
        in_channels=2048,
        low_level_channels=256,
        num_classes=num_classes,
        aspp_dilate=aspp_dilate
    )

    # 5. 套上支持多输入输出的外壳
    model = _MEDeepLabModelWrapper(dual_backbone, classifier)
    return model


def m2f_deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """
    顶配版 M2F-DeepLab (ResNet50): 绝对对称的双 ResNet50 骨干网络
    """
    from .backbone import resnet
    from .utils import IntermediateLayerGetter

    replace_stride_with_dilation = [False, True, True] if output_stride == 8 else [False, False, True]
    # 替换掉这行：
    # aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
    
    # 修改为：针对 256x256 完美贴合的 ASPP (这对应了 mmseg 的 1, 4, 8, 12)
    aspp_dilate = [4, 8, 12]

    # 1. 组装 RGB Stream (ResNet50) - 自动加载 ImageNet 预训练
    rgb_net = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation
    )
    # 修改为 (新增 layer3 提取):
    rgb_backbone = IntermediateLayerGetter(rgb_net, return_layers={"layer4": "out", "layer3": "aux", "layer1": "low_level"})

    # 2. 组装 DEM Stream (ResNet50) - 同样使用强大的 ResNet50 并加载预训练
    dem_net = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation
    )
    # DEM 流只需要 out 特征，不需要 low_level
    dem_backbone = IntermediateLayerGetter(dem_net, return_layers={"layer4": "out"})

    # 3. 组装 M2F 双流 Backbone
    # ResNet50 的 out 通道是 2048，完美对称
    dual_backbone = MEDeepLabDualStreamBackbone(rgb_backbone, dem_backbone, rgb_ch=2048, dem_ch=2048)

    # 4. 组装带有边缘感知的双分支 Decoder
    classifier = MEDeepLabHeadV3Plus(
        in_channels=2048,
        low_level_channels=256,
        num_classes=num_classes,
        aspp_dilate=aspp_dilate
    )

    # 5. 套上支持多输入输出的外壳
    model = _MEDeepLabModelWrapper(dual_backbone, classifier)
    return model