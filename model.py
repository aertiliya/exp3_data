import torch
import torch.nn as nn
from torchvision import models
import config

class VideoClassifier(nn.Module):
    """
    基线模型：ResNet18 + 时序平均池化
    架构：对每帧提取特征 → 平均池化 → 分类
    """
    def __init__(self, num_classes=3, pretrained=True, dropout=None):
        super(VideoClassifier, self).__init__()
        
        # 使用预训练的ResNet18作为特征提取器
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 移除最后的全连接层，只保留特征提取部分
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 从配置读取dropout，如果没有则使用默认值
        dropout_rate = dropout if dropout is not None else getattr(config, 'DROPOUT', 0.5)
        
        # 新的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        x: [batch_size, num_frames, C, H, W]
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # 重塑为 [batch_size * num_frames, C, H, W]
        x = x.view(-1, c, h, w)
        
        # 提取每帧特征: [batch_size * num_frames, 512, 1, 1]
        features = self.features(x)
        
        # 展平: [batch_size * num_frames, 512]
        features = features.view(-1, 512)
        
        # 重塑回视频级别: [batch_size, num_frames, 512]
        features = features.view(batch_size, num_frames, 512)
        
        # 时序平均池化: [batch_size, 512]
        video_features = features.mean(dim=1)
        
        # 分类: [batch_size, num_classes]
        logits = self.classifier(video_features)
        
        return logits


def create_model(num_classes=3, pretrained=True):
    """创建模型"""
    model = VideoClassifier(num_classes=num_classes, pretrained=pretrained)
    return model