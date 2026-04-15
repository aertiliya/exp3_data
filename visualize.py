"""
疲劳驾驶行为识别 - 可视化分析
包含：训练过程曲线、混淆矩阵、各类指标对比、ROC曲线、PR曲线
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import label_binarize
import config
from dataset import create_dataloaders
from model import create_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(history_path=None):
    """绘制训练过程曲线"""
    if history_path is None:
        history_path = config.OUTPUT_DIR / 'training_history.json'

    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 过拟合分析
    ax3 = axes[1, 0]
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax3.plot(gap, label='Train-Val Gap', color='red', linewidth=2)
    ax3.axhline(y=0.05, color='green', linestyle='--', label='Good Fit (5%)')
    ax3.axhline(y=0.1, color='orange', linestyle='--', label='Overfit Warning (10%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Gap')
    ax3.set_title('Overfitting Analysis (Train - Val)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 学习率曲线（如果有）
    ax4 = axes[1, 1]
    if 'learning_rate' in history:
        ax4.plot(history['learning_rate'], linewidth=2, color='purple')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Learning Rate Data Not Available',
                ha='center', va='center', fontsize=12)
        ax4.set_title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {config.OUTPUT_DIR / 'training_curves.png'}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, normalize=False):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES,
                annot_kws={'size': 14})
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    suffix = '_normalized' if normalize else ''
    plt.savefig(config.OUTPUT_DIR / f'confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {config.OUTPUT_DIR / f'confusion_matrix{suffix}.png'}")
    plt.show()


def plot_class_metrics(y_true, y_pred):
    """绘制各类别指标对比图"""
    # 计算各类指标
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    x = np.arange(len(config.CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASSES)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Class metrics saved to: {config.OUTPUT_DIR / 'class_metrics.png'}")
    plt.show()


def plot_roc_curves(y_true, y_probs):
    """绘制ROC曲线"""
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'green', 'red']
    for i, class_name in enumerate(config.CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Each Class', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to: {config.OUTPUT_DIR / 'roc_curves.png'}")
    plt.show()


def plot_pr_curves(y_true, y_probs):
    """绘制PR曲线"""
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'green', 'red']
    for i, class_name in enumerate(config.CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f'{class_name} (AP = {pr_auc:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves for Each Class', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'pr_curves.png', dpi=300, bbox_inches='tight')
    print(f"PR curves saved to: {config.OUTPUT_DIR / 'pr_curves.png'}")
    plt.show()


def plot_prediction_confidence(y_true, y_pred, y_probs):
    """绘制预测置信度分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, class_name in enumerate(config.CLASSES):
        # 获取当前类别的样本
        mask = y_true == i
        correct_mask = (y_true == y_pred) & mask
        wrong_mask = (y_true != y_pred) & mask

        # 置信度
        correct_conf = y_probs[correct_mask].max(axis=1) if correct_mask.any() else []
        wrong_conf = y_probs[wrong_mask].max(axis=1) if wrong_mask.any() else []

        ax = axes[i]
        if len(correct_conf) > 0:
            ax.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
        if len(wrong_conf) > 0:
            ax.hist(wrong_conf, bins=20, alpha=0.7, label='Wrong', color='red')

        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{class_name}', fontsize=12)
        ax.legend()
        ax.set_xlim(0, 1)

    plt.suptitle('Prediction Confidence Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Confidence distribution saved to: {config.OUTPUT_DIR / 'confidence_distribution.png'}")
    plt.show()


def generate_report(y_true, y_pred, y_probs):
    """生成详细报告"""
    report = classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4)
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(report)

    # 保存到文件
    with open(config.OUTPUT_DIR / 'classification_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"\nReport saved to: {config.OUTPUT_DIR / 'classification_report.txt'}")


def visualize_all():
    """运行所有可视化"""
    print("="*60)
    print("STARTING VISUALIZATION")
    print("="*60)

    # 1. 训练历史
    print("\n[1/6] Plotting training history...")
    plot_training_history()

    # 加载测试数据
    print("\n[2/6] Loading test data and model...")
    _, _, test_loader, _ = create_dataloaders(config)

    model = create_model(num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(config.OUTPUT_DIR / 'best_model.pth',
                           map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()

    # 预测
    print("[3/6] Running predictions...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for videos, labels, _ in test_loader:
            videos = videos.to(config.DEVICE)
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 生成报告
    print("[4/6] Generating classification report...")
    generate_report(all_labels, all_preds, all_probs)

    # 绘制各种图表
    print("[5/6] Plotting confusion matrices...")
    plot_confusion_matrix(all_labels, all_preds, normalize=False)
    plot_confusion_matrix(all_labels, all_preds, normalize=True)

    print("[6/6] Plotting metrics and curves...")
    plot_class_metrics(all_labels, all_preds)
    plot_roc_curves(all_labels, all_probs)
    plot_pr_curves(all_labels, all_probs)
    plot_prediction_confidence(all_labels, all_preds, all_probs)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED!")
    print(f"All figures saved to: {config.OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    visualize_all()
