import os
from pathlib import Path

# ========== 路径配置 ==========
# 自动检测运行环境并设置正确的数据路径
# 优先级：预提取数据 > 原始视频数据

if os.path.exists("/kaggle/working/exp3_data/data_processed"):
    # Kaggle环境 - 预提取的帧图片（最快）
    DATA_ROOT = Path("/kaggle/working/exp3_data/data_processed")
    print("✅ Using preprocessed frame images (fastest)")
elif os.path.exists("data_processed"):
    # 本地或Kaggle - 预提取的帧图片
    DATA_ROOT = Path("data_processed")
    print("✅ Using preprocessed frame images (fastest)")
elif os.path.exists("/kaggle/input/datasets/cartiliya/videodata"):
    # Kaggle环境 - 原始数据集路径
    DATA_ROOT = Path("/kaggle/input/datasets/cartiliya/videodata")
    print("⚠️ Using original videos (slow)")
elif os.path.exists("/kaggle/working/exp3_data/data"):
    # Kaggle环境 - 软链接路径
    DATA_ROOT = Path("/kaggle/working/exp3_data/data")
    print("⚠️ Using original videos (slow)")
else:
    # 本地环境
    DATA_ROOT = Path("数据带干扰")
    print("⚠️ Using original videos (slow)")

TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR = DATA_ROOT / "Val"
TEST_DIR = DATA_ROOT / "Test"

# ========== 数据配置 ==========
CLASSES = ['Normal', 'Yawning', 'Microsleep']
NUM_CLASSES = 3

# GPU性能优化：降低分辨率减少IO瓶颈
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    # Kaggle GPU环境 - 优化配置
    IMG_SIZE = 112        # 降低分辨率加速加载
    NUM_FRAMES = 4        # 减少帧数
    BATCH_SIZE = 32       # 更大batch充分利用GPU
    NUM_WORKERS = 0
else:
    # 本地CPU环境
    IMG_SIZE = 224
    NUM_FRAMES = 8
    BATCH_SIZE = 4
    NUM_WORKERS = 0

# ========== 训练配置 ==========
EPOCHS = 30
LEARNING_RATE = 5e-5  # 降低学习率
WEIGHT_DECAY = 1e-3   # 增加权重衰减
DROPOUT = 0.7         # 增加Dropout
LABEL_SMOOTHING = 0.1 # 标签平滑
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