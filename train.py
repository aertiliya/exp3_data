import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import config
from dataset import create_dataloaders
from model import create_model
import json


class FocalLoss(nn.Module):
    """Focal Loss - gamma=1.0温和聚焦难样本"""
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for videos, labels, _ in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels, _ in tqdm(loader, desc='Validating'):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train():
    """主训练函数"""
    # 创建数据加载器
    train_loader, val_loader, _, class_weights = create_dataloaders(config)
    
    # 创建模型
    model = create_model(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(config.DEVICE)
    
    # 损失函数：Focal Loss gamma=1.0 + 类别权重（温和聚焦难样本）
    criterion = FocalLoss(alpha=class_weights.to(config.DEVICE), gamma=1.0)

    # 优化器
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)

    # 学习率调度器：CosineAnnealingLR（平滑衰减）
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    # 训练记录
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*50}")
    print(f"Starting training on {config.DEVICE}")
    print(f"{'='*50}\n")
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, config.DEVICE)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 学习率调整（CosineAnnealingLR）
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, config.OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved new best model with val_acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        # 早停
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print()
    
    # 保存训练历史
    with open(config.OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {config.OUTPUT_DIR / 'best_model.pth'}")
    print(f"{'='*50}\n")
    
    return model, history


if __name__ == '__main__':
    train()