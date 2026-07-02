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
echo "日志文件: $(pwd)/NSP_Choice_Fine-Tuning_Dataset_Generating.log"
echo ""

# 检查源数据目录是否存在
SOURCE_DIR="/root/datasets/MFR数据集"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源数据目录不存在: $SOURCE_DIR"
    exit 1
fi

echo "源数据目录: $SOURCE_DIR"
echo "开始生成数据集（使用nohup后台运行）..."
echo ""

# 获取当前时间戳
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "任务开始时间: $START_TIME"
echo ""

# 使用nohup运行Python脚本，并将输出重定向到日志文件
nohup python NSP_Choice_Fine-Tuning_Dataset_Generating.py > NSP_Choice_Fine-Tuning_Dataset_Generating.log 2>&1 &

# 获取后台进程的PID
PYTHON_PID=$!
echo "Python脚本已在后台运行，PID: $PYTHON_PID"
echo "日志实时输出到: $(pwd)/NSP_Choice_Fine-Tuning_Dataset_Generating.log"
echo ""

echo "可以使用以下命令监控日志："
echo "  tail -f NSP_Choice_Fine-Tuning_Dataset_Generating.log"
echo ""
echo "可以使用以下命令检查进程状态："
echo "  ps -p $PYTHON_PID"
echo ""
echo "如果要终止进程，可以使用："
echo "  kill $PYTHON_PID"
echo ""

# 等待进程完成
echo "等待任务完成..."
wait $PYTHON_PID

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 数据集生成完成!"
    echo "=========================================="
    echo ""
    echo "生成的数据集位置: /root/datasets/fine-tuning数据集_NSP/"
    echo "日志文件: $(pwd)/NSP_Choice_Fine-Tuning_Dataset_Generating.log"
    echo ""
    echo "查看生成的文件："
    echo "  ls -lh /root/datasets/fine-tuning数据集_NSP/"
    echo ""
    echo "生成的文件："
    echo "  - finetuning_nsp_CICIoT2022_MFR.pt"
    echo "  - finetuning_nsp_ISCXTor2016_MFR.pt"
    echo "  - finetuning_nsp_ISCXVPN2016_MFR.pt"
    echo "  - finetuning_nsp_USTC-TFC2016_MFR.pt"
    echo ""
    
    # 显示日志文件大小
    if [ -f "NSP_Choice_Fine-Tuning_Dataset_Generating.log" ]; then
        LOG_SIZE=$(du -h NSP_Choice_Fine-Tuning_Dataset_Generating.log | cut -f1)
        echo "日志文件大小: $LOG_SIZE"
    fi
else
    echo ""
    echo "=========================================="
    echo "✗ 数据集生成失败!"
    echo "=========================================="
    echo "请查看日志文件获取详细信息:"
    echo "  tail -100 NSP_Choice_Fine-Tuning_Dataset_Generating.log"
    exit 1
fi