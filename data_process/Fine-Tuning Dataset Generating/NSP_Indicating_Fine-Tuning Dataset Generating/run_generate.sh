#!/bin/bash

# NSP微调数据集生成脚本
# 用途：从MFR数据集生成.pt格式的微调数据集（前4个数据包×80 tokens）

echo "=========================================="
echo "NSP微调数据集生成工具"
echo "=========================================="
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 显示配置信息
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version 2>&1)"
echo "日志文件: $(pwd)/NSP_Indicating_Fine-Tuning_Dataset_Generating_随机化_50pct.log"
echo ""

# 检查源数据目录是否存在
SOURCE_DIR="/root/datasets_new/MFR数据集_微调_随机化/"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源数据目录不存在: $SOURCE_DIR"
    exit 1
fi

echo "源数据目录: $SOURCE_DIR"
echo "数据集格式: 前4个数据包×80 tokens = 320 tokens/流"
echo "数据集划分: 训练集:验证集:测试集 = 8:1:1"
echo "输出文件: 每个数据集生成3个文件 (*_train.pt, *_dev.pt, *_test.pt)"
echo "开始生成数据集（使用nohup后台运行）..."
echo ""

# 获取当前时间戳
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "任务开始时间: $START_TIME"
echo ""

# 使用nohup运行Python脚本，并将输出重定向到日志文件
nohup python NSP_Indicating_Fine-Tuning_Dataset_Generating.py --verify > NSP_Indicating_Fine-Tuning_Dataset_Generating_随机化_50pct.log 2>&1 &


