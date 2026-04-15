import os
from pathlib import Path

# ========== 路径配置 ==========
# 自动检测运行环境并设置正确的数据路径
# 优先级：预提取数据 > 原始视频数据

# 获取当前文件所在目录
CURRENT_DIR = Path(__file__).parent

# 检查预提取数据路径（多个可能位置）
PREPROCESSED_PATHS = [
    Path("/kaggle/working/data_processed"),              # Kaggle实际路径
    Path("/kaggle/working/exp3_data/data_processed"),  # Kaggle备用路径
    CURRENT_DIR / "data_processed",                      # 相对当前文件
    Path("data_processed"),                              # 相对工作目录
]

DATA_ROOT = None
for path in PREPROCESSED_PATHS:
    if path.exists():
        # 验证是否真的有预提取的数据（检查是否有jpg文件）
        for class_name in ['Normal', 'Yawning', 'Microsleep']:
            class_dir = path / "Train" / class_name
            if class_dir.exists() and list(class_dir.glob("*_frame0.jpg")):
                DATA_ROOT = path
                print(f"✅ Using preprocessed frame images: {path}")
                break
        if DATA_ROOT:
            break

if DATA_ROOT is None:
    # 未找到预提取数据，使用原始视频
    video_paths = [
        ("/kaggle/input/datasets/cartiliya/videodata", "Kaggle 原始数据集"),
        ("/kaggle/working/exp3_data/data", "Kaggle 软链接"),
        ("数据带干扰", "本地数据"),
    ]
    for path_str, desc in video_paths:
        if os.path.exists(path_str):
            DATA_ROOT = Path(path_str)
            print(f"⚠️ Using original videos ({desc}): {path_str}")
            break

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
