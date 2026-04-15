import torch, numpy as np, pandas as pd, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import config
from dataset import create_dataloaders
from model import create_model

def test_model():
    _, _, test_loader, _ = create_dataloaders(config)
    model = create_model(num_classes=config.NUM_CLASSES, pretrained=False).to(config.DEVICE)
    ckpt = torch.load(config.OUTPUT_DIR / 'best_model.pth', map_location=config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']} | Val Acc: {ckpt['val_acc']:.4f}\n")

    all_preds, all_labels, all_probs, all_names = [], [], [], []
    print("Testing with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for videos, labels, names in test_loader:
            videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)
            # TTA: 原始 + 水平翻转 平均 logits
            logits = (model(videos) + model(torch.flip(videos, dims=[-1]))) / 2.0
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_names.extend(names)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*50 + "\nTEST RESULTS\n" + "="*50)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    for i, c in enumerate(config.CLASSES):
        p = precision_score(all_labels, all_preds, average=None)[i]
        r = recall_score(all_labels, all_preds, average=None)[i]
        f = f1_score(all_labels, all_preds, average=None)[i]
        print(f"  {c:12s}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")
    print("="*50 + "\n")

    pd.DataFrame({'filename': all_names, 'true_label': [config.CLASSES[l] for l in all_labels],
                  'pred_label': [config.CLASSES[p] for p in all_preds],
                  'confidence': [all_probs[i][p] for i, p in enumerate(all_preds)]}).to_csv(config.OUTPUT_DIR/'pred_test.csv', index=False)
    
    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
               'per_class': {c: {'precision': float(precision_score(all_labels, all_preds, average=None)[i]),
                                 'recall': float(recall_score(all_labels, all_preds, average=None)[i]),
                                 'f1_score': float(f1_score(all_labels, all_preds, average=None)[i])} 
                              for i, c in enumerate(config.CLASSES)}}
    with open(config.OUTPUT_DIR/'metrics.json', 'w') as f: json.dump(metrics, f, indent=2)
    
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR/'confusion_matrix.png', dpi=300)
    print("✅ Files saved: pred_test.csv, metrics.json, confusion_matrix.png")
    return acc, metrics

if __name__ == '__main__': test_model()