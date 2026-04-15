import os
from pathlib import Path

# ========== 路径配置 ==========
DATA_ROOT = Path("数据带干扰")  # 修改为你的数据路径
TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR = DATA_ROOT / "Val"
TEST_DIR = DATA_ROOT / "Test"

# ========== 数据配置 ==========
CLASSES = ['Normal', 'Yawning', 'Microsleep']
NUM_CLASSES = 3
IMG_SIZE = 224
NUM_FRAMES = 8  # CPU友好，每个视频采样8帧
BATCH_SIZE = 4  # CPU环境用小batch
NUM_WORKERS = 0  # Windows CPU建议设为0

# ========== 训练配置 ==========
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') else 'cpu'  # Kaggle自动用GPU

# ========== 模型配置 ==========
MODEL_NAME = 'resnet18_baseline'
EARLY_STOPPING_PATIENCE = 10

# ========== 输出配置 ==========
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Num frames per video: {NUM_FRAMES}")
print(f"Batch size: {BATCH_SIZE}")