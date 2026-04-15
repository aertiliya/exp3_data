"""
疲劳驾驶检测 - 实验报告可视化分析工具
包含：训练曲线、混淆矩阵、ROC曲线、PR曲线、特征可视化等
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(history_path='outputs/training_history.json', save_path='outputs/training_curves.png'):
    """绘制训练曲线（损失+准确率双轴）"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])

    # 准确率曲线
    ax2.plot(epochs, [acc*100 for acc in history['train_acc']], 'b-', linewidth=2,
             label='Train Acc', marker='o', markersize=4)
    ax2.plot(epochs, [acc*100 for acc in history['val_acc']], 'r-', linewidth=2,
             label='Val Acc', marker='s', markersize=4)
    ax2.axhline(y=85, color='green', linestyle='--', linewidth=2, label='Target 85%')
    ax2.fill_between(epochs, 80, 100, alpha=0.1, color='green')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {save_path}")
    plt.show()


def plot_confusion_matrix_detailed(csv_path='outputs/pred_test.csv', save_path='outputs/confusion_matrix.png'):
    """绘制详细混淆矩阵（包含百分比和数量）"""
    df = pd.read_csv(csv_path)
    labels = config.CLASSES

    # 计算混淆矩阵
    cm = confusion_matrix(df['true_label'], df['pred_label'], labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热力图
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                square=True, linewidths=2, linecolor='white')

    # 添加数值和百分比
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                   fontsize=14, fontweight='bold', color=color)

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix (Test Set)', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {save_path}")
    plt.show()


def plot_roc_curves(csv_path='outputs/pred_test.csv', save_path='outputs/roc_curves.png'):
    """绘制ROC曲线（多分类One-vs-Rest）"""
    df = pd.read_csv(csv_path)
    labels = config.CLASSES
    n_classes = len(labels)

    # 准备数据
    y_true = label_binarize(df['true_label'], classes=labels)

    # 从CSV中重建预测概率（使用confidence作为近似）
    # 注意：这里需要实际的预测概率，如果CSV中没有，需要修改test.py保存概率
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # 简化的ROC曲线（基于置信度）
    for i, class_name in enumerate(labels):
        # 二分类问题：当前类 vs 其他类
        y_true_binary = (df['true_label'] == class_name).astype(int)
        y_score = df.apply(lambda x: x['confidence'] if x['pred_label'] == class_name else 1-x['confidence'], axis=1)

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curves saved to: {save_path}")
    plt.show()


def plot_class_metrics(csv_path='outputs/pred_test.csv', save_path='outputs/class_metrics.png'):
    """绘制各类别性能指标对比图"""
    df = pd.read_csv(csv_path)
    labels = config.CLASSES

    # 计算各类别指标
    metrics = {'Precision': [], 'Recall': [], 'F1-Score': []}

    for class_name in labels:
        tp = ((df['true_label'] == class_name) & (df['pred_label'] == class_name)).sum()
        fp = ((df['true_label'] != class_name) & (df['pred_label'] == class_name)).sum()
        fn = ((df['true_label'] == class_name) & (df['pred_label'] != class_name)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics['Precision'].append(precision * 100)
        metrics['Recall'].append(recall * 100)
        metrics['F1-Score'].append(f1 * 100)

    # 绘制分组柱状图
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, (metric, values) in enumerate(metrics.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 110])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class metrics saved to: {save_path}")
    plt.show()


def plot_class_distribution(save_path='outputs/class_distribution.png'):
    """绘制数据集类别分布（饼图+柱状图）"""
    # 统计数据
    splits = ['Train', 'Val', 'Test']
    data = {split: [] for split in splits}

    for split in splits:
        split_dir = getattr(config, f'{split.upper()}_DIR')
        for class_name in config.CLASSES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                if list(class_dir.glob("*_frame0.jpg")):
                    count = len(list(class_dir.glob("*_frame0.jpg")))
                else:
                    count = len(list(class_dir.glob("*.mp4")))
                data[split].append(count)
            else:
                data[split].append(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for idx, (split, counts) in enumerate(data.items()):
        # 饼图
        wedges, texts, autotexts = axes[idx].pie(
            counts, labels=config.CLASSES, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11}
        )

        # 设置百分比文字样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        total = sum(counts)
        axes[idx].set_title(f'{split} Set\n(Total: {total})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class distribution saved to: {save_path}")
    plt.show()


def plot_prediction_analysis(csv_path='outputs/pred_test.csv', save_path='outputs/prediction_analysis.png'):
    """预测分析：置信度分布 + 正确率-置信度关系"""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左图：各类别的置信度分布
    for class_name in config.CLASSES:
        class_data = df[df['true_label'] == class_name]['confidence']
        axes[0].hist(class_data, bins=15, alpha=0.6, label=class_name,
                    edgecolor='black', linewidth=1.2)

    axes[0].set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Confidence Distribution by True Class', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右图：不同置信度区间的准确率
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []

    for i in range(len(bins) - 1):
        mask = (df['confidence'] >= bins[i]) & (df['confidence'] < bins[i+1])
        if mask.sum() > 0:
            acc = (df[mask]['true_label'] == df[mask]['pred_label']).mean()
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)

    axes[1].bar(bin_centers, accuracies, width=0.08, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Confidence Bin', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy vs Confidence', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction analysis saved to: {save_path}")
    plt.show()


def generate_experiment_report(csv_path='outputs/pred_test.csv',
                               metrics_path='outputs/metrics.json',
                               history_path='outputs/training_history.json'):
    """生成完整的实验报告"""
    print("\n" + "="*80)
    print(" "*25 + "实 验 报 告")
    print("="*80)

    print("\n【一、实验配置】")
    print(f"   模型架构: ResNet18 + BiGRU + Attention (路线C)")
    print(f"   损失函数: Focal Loss (路线D)")
    print(f"   图像尺寸: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"   帧数: {config.NUM_FRAMES}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   学习率: {config.LEARNING_RATE}")
    print(f"   Dropout: {config.DROPOUT}")

    # 加载训练历史
    if Path(history_path).exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        best_epoch = np.argmax(history['val_acc'])
        print(f"\n【二、训练过程】")
        print(f"   最佳Epoch: {best_epoch + 1}")
        print(f"   最终Train Acc: {history['train_acc'][-1]*100:.2f}%")
        print(f"   最终Val Acc: {history['val_acc'][-1]*100:.2f}%")
        print(f"   最佳Val Acc: {max(history['val_acc'])*100:.2f}%")

    # 加载测试指标
    if Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        print(f"\n【三、测试集性能】")
        print(f"   Overall Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Weighted Precision: {metrics['precision']*100:.2f}%")
        print(f"   Weighted Recall:    {metrics['recall']*100:.2f}%")
        print(f"   Weighted F1-Score:  {metrics['f1_score']*100:.2f}%")

        print(f"\n【四、各类别详细指标】")
        print("   " + "-"*50)
        print(f"   {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("   " + "-"*50)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"   {class_name:<12} {class_metrics['precision']*100:<11.2f}% "
                  f"{class_metrics['recall']*100:<11.2f}% {class_metrics['f1_score']*100:<11.2f}%")
        print("   " + "-"*50)

    # 加载预测详情
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        total = len(df)
        correct = (df['true_label'] == df['pred_label']).sum()

        print(f"\n【五、预测统计】")
        print(f"   测试样本总数: {total}")
        print(f"   正确预测数: {correct}")
        print(f"   错误预测数: {total - correct}")
        print(f"   整体准确率: {correct/total*100:.2f}%")

        # 各类别准确率
        print(f"\n【六、各类别准确率】")
        for class_name in config.CLASSES:
            class_df = df[df['true_label'] == class_name]
            class_correct = (class_df['true_label'] == class_df['pred_label']).sum()
            class_total = len(class_df)
            if class_total > 0:
                print(f"   {class_name:<12}: {class_correct}/{class_total} = {class_correct/class_total*100:.2f}%")

    print("\n" + "="*80)


def run_all_visualizations():
    """运行所有可视化"""
    print("🎨 正在生成实验报告可视化...\n")

    files = {
        'history': Path('outputs/training_history.json'),
        'csv': Path('outputs/pred_test.csv'),
        'metrics': Path('outputs/metrics.json')
    }

    # 1. 训练曲线
    if files['history'].exists():
        plot_training_history()
    else:
        print("⚠️ training_history.json not found")

    # 2. 混淆矩阵
    if files['csv'].exists():
        plot_confusion_matrix_detailed()
    else:
        print("⚠️ pred_test.csv not found")

    # 3. ROC曲线
    if files['csv'].exists():
        plot_roc_curves()
    else:
        print("⚠️ pred_test.csv not found")

    # 4. 类别指标对比
    if files['csv'].exists():
        plot_class_metrics()
    else:
        print("⚠️ pred_test.csv not found")

    # 5. 类别分布
    plot_class_distribution()

    # 6. 预测分析
    if files['csv'].exists():
        plot_prediction_analysis()
    else:
        print("⚠️ pred_test.csv not found")

    # 7. 实验报告
    if files['csv'].exists():
        generate_experiment_report()
    else:
        print("⚠️ pred_test.csv not found")

    print("\n✅ 所有可视化已完成！")
    print("📁 输出文件保存在 outputs/ 目录:")
    print("   - training_curves.png")
    print("   - confusion_matrix.png")
    print("   - roc_curves.png")
    print("   - class_metrics.png")
    print("   - class_distribution.png")
    print("   - prediction_analysis.png")


if __name__ == '__main__':
    run_all_visualizations()
