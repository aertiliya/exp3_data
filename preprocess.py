"""
预提取视频帧为图片，训练时直接读图片，避免实时解码
引入人脸检测与裁剪，提升关键特征占比
使用 Decord 库加速视频解码（Kaggle环境推荐）
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import config

# 尝试导入 Decord 加速库
try:
    from decord import VideoReader, cpu
    USE_DECORD = True
    print("✅ 使用 Decord 加速视频解码")
except ImportError:
    USE_DECORD = False
    print("⚠️ 未安装 Decord，使用 OpenCV 解码（较慢）")

# 加载 OpenCV 自带的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_bbox(frame):
    """获取人脸边界框（缩小图检测加速）"""
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        
        padding = int(w * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        return (x1, y1, x2, y2)
    else:
        h, w, _ = frame.shape
        return (w//4, h//4, w*3//4, h*3//4)

def extract_frames_decord(video_path, output_dir, num_frames=4, img_size=224):
    """使用 Decord 快速提取视频帧（首帧检测+全视频复用坐标）"""
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames <= 0:
            return False
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(frame_indices).asnumpy()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        x1, y1, x2, y2 = get_face_bbox(frames[0])
        
        for i, frame in enumerate(frames):
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.size == 0:
                cropped_frame = frame
                
            resized_frame = cv2.resize(cropped_frame, (img_size, img_size))
            
            save_path = output_dir / f"{Path(video_path).stem}_frame{i}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        return True
    except Exception:
        return False

def extract_frames_opencv(video_path, output_dir, num_frames=4, img_size=224):
    """使用 OpenCV 提取视频帧（首帧检测+全视频复用坐标）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return False
    
    x1, y1, x2, y2 = get_face_bbox(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_frame = frame_rgb[y1:y2, x1:x2]
            
            if cropped_frame.size == 0:
                cropped_frame = frame_rgb
                
            resized_frame = cv2.resize(cropped_frame, (img_size, img_size))
            
            save_path = output_dir / f"{Path(video_path).stem}_frame{i}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
    
    cap.release()
    return True

def extract_frames(video_path, output_dir, num_frames=4, img_size=224):
    """自动选择最快的方式提取视频帧"""
    if USE_DECORD:
        return extract_frames_decord(video_path, output_dir, num_frames, img_size)
    else:
        return extract_frames_opencv(video_path, output_dir, num_frames, img_size)


def preprocess_dataset(split_dir, output_root, num_frames=4, img_size=224, max_workers=4):
    """处理一个split（Train/Val/Test），使用多线程加速"""
    split_dir = Path(split_dir)
    output_root = Path(output_root)
    
    for class_name in config.CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        
        output_class_dir = output_root / split_dir.name / class_name
        video_files = list(class_dir.glob("*.mp4"))
        
        print(f"\nProcessing {class_name}: {len(video_files)} videos")
        
        # 使用多线程加速处理
        task = lambda vp: extract_frames(vp, output_class_dir, num_frames, img_size)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 list() 让 tqdm 能正常显示进度条
            list(tqdm(executor.map(task, video_files), total=len(video_files)))


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
