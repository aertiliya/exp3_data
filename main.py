"""
疲劳驾驶行为识别 - 基线模型
运行顺序：
1. python main.py --stage train   # 训练
2. python main.py --stage test    # 测试
"""

import argparse
from train import train
from test import test_model

def main():
    parser = argparse.ArgumentParser(description='Fatigue Detection Baseline')
    parser.add_argument('--stage', type=str, default='train', 
                       choices=['train', 'test'],
                       help='Stage to run: train or test')
    
    args = parser.parse_args()
    
    if args.stage == 'train':
        train()
    elif args.stage == 'test':
        test_model()

if __name__ == '__main__':
    main()