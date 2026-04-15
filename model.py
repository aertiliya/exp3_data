import torch
import torch.nn as nn
from torchvision import models
import config


class VideoClassifier(nn.Module):
    """
    ResNet18 + 时序平均池化（轻量稳定版）
    小数据集首选：避免BiGRU过拟合
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # 轻量时序融合：平均池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        dropout_rate = getattr(config, 'DROPOUT', 0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x)               # [B*T, 512, 1, 1]
        feats = feats.view(B, T, 512)          # [B, T, 512]
        feats = feats.permute(0, 2, 1)         # [B, 512, T]
        pooled = self.temporal_pool(feats).squeeze(-1)  # [B, 512]
        return self.classifier(pooled)


def create_model(num_classes=3, pretrained=True, freeze_backbone=True):
    model = VideoClassifier(num_classes=num_classes, pretrained=pretrained)

    if freeze_backbone:
        # 冻结底层特征（conv1, bn1, layer1, layer2）
        # 只训练layer3, layer4和分类头，防止过拟合干扰噪声
        freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        for name, param in model.backbone.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
        print("✅ Frozen backbone layers: conv1, bn1, layer1, layer2")

    return model
