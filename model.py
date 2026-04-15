import torch
import torch.nn as nn
from torchvision import models
import config

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Dropout(getattr(config, 'DROPOUT', 0.5)),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x)          # [B*T, 512, 1, 1]
        feats = feats.view(B, T, 512)     # [B, T, 512]
        video_feat = feats.mean(dim=1)    # 时序平均池化 [B, 512]
        return self.classifier(video_feat)

def create_model(num_classes=3, pretrained=True):
    return VideoClassifier(num_classes=num_classes, pretrained=pretrained)