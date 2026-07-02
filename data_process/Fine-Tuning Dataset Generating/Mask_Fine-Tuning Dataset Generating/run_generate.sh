#!/bin/bash

# 微调数据集生成脚本
# 用途：从MFR数据集生成TSV格式的微调数据集

echo "=========================================="
echo "微调数据集生成工具"
echo "=========================================="
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 显示配置信息
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version 2>&1)"
echo ""

# 检查源数据目录是否存在
SOURCE_DIR="/root/datasets/MFR数据集"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源数据目录不存在: $SOURCE_DIR"
    exit 1
fi

echo "源数据目录: $SOURCE_DIR"
echo "开始生成数据集..."
echo ""

# 运行Python脚本
python Mask_Fine-Tuning_Dataset_Generating.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 数据集生成完成!"
    echo "=========================================="
    echo ""
    echo "生成的数据集位置: /root/datasets/fine-tuning数据集_Mask/"
    echo ""
    echo "查看生成的文件："
    echo "  ls -lh /root/datasets/fine-tuning数据集_Mask/fine-tuning_CICIoT2022_MFR/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 数据集生成失败!"
    echo "=========================================="
    exit 1
fi

