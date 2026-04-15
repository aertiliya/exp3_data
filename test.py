import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import config
from dataset import create_dataloaders
from model import create_model

def test_model():
    """测试模型"""
    # 加载数据
    _, _, test_loader, _ = create_dataloaders(config)
    
    # 加载模型
    model = create_model(num_classes=config.NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(config.OUTPUT_DIR / 'best_model.pth', 
                           map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"with val_acc: {checkpoint['val_acc']:.4f}\n")
    
    # 测试
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    print("Testing...")
    with torch.no_grad():
        for videos, labels, filenames in test_loader:
            videos = videos.to(config.DEVICE)
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_filenames.extend(filenames)
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 每类的指标
    class_precision = precision_score(all_labels, all_preds, average=None)
    class_recall = recall_score(all_labels, all_preds, average=None)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 打印结果
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1-Score:  {f1:.4f}")
    print("\nPer-class metrics:")
    for i, class_name in enumerate(config.CLASSES):
        print(f"  {class_name:12s}: P={class_precision[i]:.3f}, "
              f"R={class_recall[i]:.3f}, F1={class_f1[i]:.3f}")
    print("="*50 + "\n")
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': [config.CLASSES[l] for l in all_labels],
        'pred_label': [config.CLASSES[p] for p in all_preds],
        'confidence': [max(probs[i]) for i in range(len(all_preds))]
    })
    pred_df.to_csv(config.OUTPUT_DIR / 'pred_test.csv', index=False)
    print(f"Predictions saved to: {config.OUTPUT_DIR / 'pred_test.csv'}")
    
    # 保存指标
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class': {
            class_name: {
                'precision': float(class_precision[i]),
                'recall': float(class_recall[i]),
                'f1_score': float(class_f1[i])
            }
            for i, class_name in enumerate(config.CLASSES)
        }
    }
    
    import json
    with open(config.OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {config.OUTPUT_DIR / 'metrics.json'}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'confusion_matrix.png', dpi=300)
    print(f"Confusion matrix saved to: {config.OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.show()
    
    return accuracy, metrics


if __name__ == '__main__':
    test_model()