import torch
import torch.nn as nn
from torchvision import models
import config


class VideoClassifier(nn.Module):
    """
    ResNet18 + GRU 时序建模
    有效捕捉打哈欠和闭眼的动态过程（路线C）
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # 加入双向GRU处理时序动态特征
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        dropout_rate = getattr(config, 'DROPOUT', 0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)  # 双向GRU输出维度翻倍
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x)               # [B*T, 512, 1, 1]
        feats = feats.view(B, T, 512)          # [B, T, 512]

        # 经过 GRU
        gru_out, _ = self.gru(feats)           # [B, T, 256]
        # 对所有时间步取平均，综合考虑整个视频片段的疲劳动作
        avg_out = gru_out.mean(dim=1)           # [B, 256]

        return self.classifier(avg_out)


def create_model(num_classes=3, pretrained=True, freeze_backbone=True):
    model = VideoClassifier(num_classes=num_classes, pretrained=pretrained)

    if freeze_backbone:
        # 释放 layer2，让模型能更好地适应人脸特征提取
        freeze_layers = ['conv1', 'bn1', 'layer1']
        for name, param in model.backbone.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
        print("✅ Frozen backbone layers: conv1, bn1, layer1 (layer2,3,4 trainable)")

    return model
