import os
from pathlib import Path

DATA_ROOT = Path("data_processed")  # 必须指向预提取后的图片目录
TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR = DATA_ROOT / "Val"
TEST_DIR = DATA_ROOT / "Test"

CLASSES = ['Normal', 'Yawning', 'Microsleep']
NUM_CLASSES = 3
IMG_SIZE = 224
NUM_FRAMES = 8
BATCH_SIZE = 16
NUM_WORKERS = 2 if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else 0

EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 10

DEVICE = 'cuda' if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else 'cpu'
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)