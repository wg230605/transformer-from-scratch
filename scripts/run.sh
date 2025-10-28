#!/bin/bash

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:.

# 确保结果和 checkpoints 目录存在
mkdir -p checkpoints
mkdir -p results

# 运行训练脚本
# 我们使用 --seed 42 (在 train.py 中硬编码) 来保证可复现性
echo "开始运行 Transformer 训练..."

python train.py

echo "训练完成。"

# (可选) 在这里添加一个 Python 脚本来绘制 results/training_history.json
# echo "正在生成训练曲线图..."
# python scripts/plot_results.py