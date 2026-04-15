import torch
import torch.nn as nn
from torchvision import models
import config


class TemporalBiGRU(nn.Module):
    """轻量级双向GRU时序建模（路线C）"""
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(hidden_size * 2, 1)  # 注意力权重
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # x: [B, T, 512]
        gru_out, _ = self.gru(x)  # [B, T, 512]

        # 注意力池化：让模型关注关键帧（如闭眼瞬间、打哈欠峰值）
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)  # [B, T, 1]
        context = torch.sum(gru_out * attn_weights, dim=1)       # [B, 512]

        return self.norm(context)


class VideoClassifier(nn.Module):
    """
    ResNet18 + BiGRU + 注意力池化
    路线C：时序建模改进
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # 时序模块（路线C）
        self.temporal = TemporalBiGRU(input_size=512, hidden_size=256)

        # 分类头
        dropout_rate = getattr(config, 'DROPOUT', 0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 空间特征提取
        x = x.view(B * T, C, H, W)
        feats = self.features(x)  # [B*T, 512, 1, 1]
        feats = feats.view(B, T, 512)  # [B, T, 512]

        # 时序建模 + 注意力池化
        video_feat = self.temporal(feats)  # [B, 512]
        return self.classifier(video_feat)


def create_model(num_classes=3, pretrained=True):
    """创建模型"""
    model = VideoClassifier(num_classes=num_classes, pretrained=pretrained)
    return model
