import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms
import random

class FatigueVideoDataset(Dataset):
    def __init__(self, root_dir, classes, num_frames=8, img_size=224, is_train=True):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        self.samples = []
        
        # 检测是否使用预提取的图片数据
        self.use_preprocessed = self._detect_preprocessed()
        
        self._load_samples()
        
        # 数据增强 - 强化版（防过拟合）
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),               # 增强：模拟头部姿态变化
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 模拟光照/模糊
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # 新增：随机擦除
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _detect_preprocessed(self):
        """检测是否使用预提取的图片数据"""
        # 检查任意一个类别目录下是否有jpg文件
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                jpg_files = list(class_dir.glob("*_frame0.jpg"))
                if jpg_files:
                    return True
        return False
    
    def _load_samples(self):
        """加载所有样本"""
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            if self.use_preprocessed:
                # 加载预提取的图片帧
                # 获取所有唯一的视频stem（去掉_frameX.jpg后缀）
                all_files = list(class_dir.glob("*_frame0.jpg"))
                for frame_path in all_files:
                    stem = frame_path.stem.replace("_frame0", "")
                    self.samples.append((str(class_dir), class_idx, class_name, stem))
                print(f"Loaded {len(all_files)} preprocessed samples from {class_name}")
            else:
                # 加载原始视频
                video_files = list(class_dir.glob("*.mp4"))
                print(f"Loaded {len(video_files)} videos from {class_name}")
                
                for video_path in video_files:
                    self.samples.append((str(video_path), class_idx, class_name, None))
    
    def _load_preprocessed_frames(self, class_dir, stem):
        """加载预提取的图片帧"""
        frames = []
        class_dir = Path(class_dir)
        for i in range(self.num_frames):
            img_path = class_dir / f"{stem}_frame{i}.jpg"
            if img_path.exists():
                frame = cv2.imread(str(img_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 如果图片不存在，用黑帧代替
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        return frames
    
    def _sample_frames_from_video(self, video_path):
        """从视频中均匀采样帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样帧索引
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # 如果视频帧数不足，重复采样
            frame_indices = np.tile(np.arange(total_frames), 
                                   (self.num_frames // total_frames + 1))[:self.num_frames]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 如果读取失败，用黑帧代替
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_preprocessed:
            class_dir, label, class_name, stem = self.samples[idx]
            frames = self._load_preprocessed_frames(class_dir, stem)
            filename = f"{stem}.mp4"
        else:
            video_path, label, class_name, _ = self.samples[idx]
            frames = self._sample_frames_from_video(video_path)
            filename = os.path.basename(video_path)
        
        if frames is None:
            # 如果加载失败，返回随机噪声
            frames = [np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8) 
                     for _ in range(self.num_frames)]
        
        # 对每帧应用变换
        frames_tensor = []
        for frame in frames:
            frame_tensor = self.transform(frame)
            frames_tensor.append(frame_tensor)
        
        # [num_frames, C, H, W]
        video_tensor = torch.stack(frames_tensor)
        
        return video_tensor, label, filename


def create_dataloaders(config):
    """创建数据加载器"""
    train_dataset = FatigueVideoDataset(
        config.TRAIN_DIR, config.CLASSES, 
        num_frames=config.NUM_FRAMES, 
        img_size=config.IMG_SIZE,
        is_train=True
    )
    
    val_dataset = FatigueVideoDataset(
        config.VAL_DIR, config.CLASSES,
        num_frames=config.NUM_FRAMES,
        img_size=config.IMG_SIZE,
        is_train=False
    )
    
    test_dataset = FatigueVideoDataset(
        config.TEST_DIR, config.CLASSES,
        num_frames=config.NUM_FRAMES,
        img_size=config.IMG_SIZE,
        is_train=False
    )
    
    # 计算类别权重（处理不平衡）
    class_counts = [0] * config.NUM_CLASSES
    for _, label, _, _ in train_dataset.samples:
        class_counts[label] += 1
    
    total_samples = sum(class_counts)
    class_weights = [total_samples / (config.NUM_CLASSES * count) for count in class_counts]
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"\nClass distribution in training set: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # 如果使用预提取数据，可以增加num_workers
    num_workers = config.NUM_WORKERS
    if train_dataset.use_preprocessed and config.DEVICE == 'cuda':
        num_workers = 2  # 预提取数据可以安全使用多进程
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader, class_weights
