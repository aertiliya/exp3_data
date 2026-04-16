import os, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from torchvision import transforms
import config

class FatigueVideoDataset(Dataset):
    def __init__(self, root_dir, classes, num_frames=8, img_size=224, is_train=True):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        self.samples = []
        self.use_preprocessed = self._detect_preprocessed()
        self._load_samples()
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _detect_preprocessed(self):
        for cls in self.classes:
            if list((self.root_dir / cls).glob("*_frame0.jpg")):
                return True
        return False

    def _load_samples(self):
        for idx, cls in enumerate(self.classes):
            cls_dir = self.root_dir / cls
            if not cls_dir.exists(): continue
            if self.use_preprocessed:
                for f in cls_dir.glob("*_frame0.jpg"):
                    stem = f.stem.replace("_frame0", "")
                    self.samples.append((str(cls_dir), idx, cls, stem))
                print(f"Loaded {len(list(cls_dir.glob('*_frame0.jpg')))} preprocessed samples from {cls}")
            else:
                for v in cls_dir.glob("*.mp4"):
                    self.samples.append((str(v), idx, cls, None))
                print(f"Loaded {len(list(cls_dir.glob('*.mp4')))} videos from {cls}")

    def _load_frames(self, cls_dir, stem):
        frames = []
        for i in range(self.num_frames):
            p = Path(cls_dir) / f"{stem}_frame{i}.jpg"
            frames.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.exists() else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        return frames

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        cls_dir, label, _, stem = self.samples[idx]
        frames = self._load_frames(cls_dir, stem) if self.use_preprocessed else None # 简化逻辑，实际走预提取路径
        # 注：因已预提取，直接走图片加载分支
        frames = self._load_frames(cls_dir, stem)
        tensors = torch.stack([self.transform(f) for f in frames])
        return tensors, label, f"{stem}.mp4"

def create_dataloaders(cfg):
    tr = FatigueVideoDataset(cfg.TRAIN_DIR, cfg.CLASSES, cfg.NUM_FRAMES, cfg.IMG_SIZE, True)
    va = FatigueVideoDataset(cfg.VAL_DIR, cfg.CLASSES, cfg.NUM_FRAMES, cfg.IMG_SIZE, False)
    te = FatigueVideoDataset(cfg.TEST_DIR, cfg.CLASSES, cfg.NUM_FRAMES, cfg.IMG_SIZE, False)

    counts = [sum(1 for _, l, _, _ in tr.samples if l==i) for i in range(cfg.NUM_CLASSES)]
    total = sum(counts)
    weights = torch.FloatTensor([total/(cfg.NUM_CLASSES*c) for c in counts])
    print(f"\nClass dist: {counts}\nLoss Weights: {weights}")

    # ========== 新增：WeightedRandomSampler物理重采样 ==========
    # 为每个样本计算采样权重（少数类样本权重更高）
    sample_weights = [total / counts[label] for _, label, _, _ in tr.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(tr.samples),  # 保持每个epoch样本总数不变
        replacement=True  # 允许重复采样少数类（微睡眠）
    )
    print(f"✅ Using WeightedRandomSampler for balanced training batches")
    # ==========================================================

    kw = {'batch_size': cfg.BATCH_SIZE, 'num_workers': cfg.NUM_WORKERS,
          'pin_memory': cfg.DEVICE=='cuda', 'persistent_workers': cfg.NUM_WORKERS>0}

    # 注意：使用sampler时，必须把shuffle设为False
    return (DataLoader(tr, sampler=sampler, **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw), weights)