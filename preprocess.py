"""
预提取视频帧为图片，训练时直接读图片，避免实时解码
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config

def extract_frames(video_path, output_dir, num_frames=4, img_size=224):
    """均匀采样视频帧并保存为jpg"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            save_path = output_dir / f"{Path(video_path).stem}_frame{i}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    cap.release()
    return True


def preprocess_dataset(split_dir, output_root, num_frames=4, img_size=224):
    """处理一个split（Train/Val/Test）"""
    split_dir = Path(split_dir)
    output_root = Path(output_root)
    
    for class_name in config.CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        
        output_class_dir = output_root / split_dir.name / class_name
        video_files = list(class_dir.glob("*.mp4"))
        
        print(f"\nProcessing {class_name}: {len(video_files)} videos")
        for video_path in tqdm(video_files):
            extract_frames(video_path, output_class_dir, num_frames, img_size)


if __name__ == '__main__':
    # 预提取所有split
    PREPROCESSED_ROOT = Path("data_processed")

    # 自动检测原始视频路径
    if os.path.exists("/kaggle/input/datasets/cartiliya/videodata"):
        VIDEO_ROOT = Path("/kaggle/input/datasets/cartiliya/videodata")
    elif os.path.exists("/kaggle/working/exp3_data/data"):
        VIDEO_ROOT = Path("/kaggle/working/exp3_data/data")
    elif os.path.exists("数据带干扰"):
        VIDEO_ROOT = Path("数据带干扰")
    else:
        VIDEO_ROOT = Path("data")  # 默认路径

    print(f"Using video source: {VIDEO_ROOT}")

    for split in ['Train', 'Val', 'Test']:
        print(f"\n{'='*40}\nProcessing {split} split\n{'='*40}")
        preprocess_dataset(
            VIDEO_ROOT / split,
            PREPROCESSED_ROOT,
            num_frames=config.NUM_FRAMES,
            img_size=config.IMG_SIZE
        )

    print(f"\n✅ 预提取完成！数据已保存到: {PREPROCESSED_ROOT}")
    print(f"共处理: {len(list(PREPROCESSED_ROOT.glob('**/*.jpg')))} 张图片")
