"""
疲劳驾驶行为识别 - 完整流程
运行顺序：
1. python main.py --stage train      # 训练
2. python main.py --stage test       # 测试
3. python main.py --stage visualize  # 可视化
"""

import argparse
from train import train
from test import test_model
from visualize import run_all_visualizations

def main():
    parser = argparse.ArgumentParser(description='Fatigue Detection')
    parser.add_argument('--stage', type=str, default='train',
                       choices=['train', 'test', 'visualize'],
                       help='Stage to run: train, test, or visualize')

    args = parser.parse_args()

    if args.stage == 'train':
        train()
    elif args.stage == 'test':
        test_model()
    elif args.stage == 'visualize':
        run_all_visualizations()

if __name__ == '__main__':
    main()