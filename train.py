import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import config
from dataset import create_dataloaders
from model import create_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc='Training')
    for videos, labels, _ in pbar:
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for videos, labels, _ in tqdm(loader, desc='Validating'):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), correct / total

def train():
    train_loader, val_loader, _, class_weights = create_dataloaders(config)
    model = create_model(num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    
    # 稳定组合：类别权重 + 标签平滑
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(config.DEVICE),
        label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.1)
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_acc, patience_counter = 0.0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*50}\nStarting training on {config.DEVICE}\n{'='*50}\n")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}\n" + "-"*30)
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        v_loss, v_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"\nTrain Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")
        print(f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
        
        scheduler.step(v_acc)
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 'val_acc': v_acc},
                       config.OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved best model | Val Acc: {v_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
            
    with open(config.OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training completed! Best Val Acc: {best_val_acc:.4f}")
    return model, history