#!/bin/bash

# ============================================================================
# 多任务预训练脚本（整合两个预训练任务 + 两级注意力机制）
# 
# 数据共享策略：
# - 每个step加载N个流样本（N = batch_size，例如8个）
# - 这N个流同时被两个任务处理，生成2N个样本（每个流产生2个样本）
# - 两个任务共享同一批流数据，样本数始终相等
# 
# 两个预训练任务：
# 1. NSP_Indicating (Packet Masking) - 50%
#    - 每个流产生1个样本
#    - 只mask流中的一个数据包（mask率90%）
#    - 用剩下4个没被mask的数据包预测被mask的数据包
# 2. MFR-Masking (Header/Payload Masking) - 50%
#    - 每个流产生5个样本（每个数据包1个）
#    - 三种关系：h-h (30%), p-p (30%), h-p (40%)
#    - mask率：h-h和p-p为80%，h-p为95%
# 
# 两级注意力机制：
# 1. 包级注意力（PacketAttention）：对每个数据包内部的header-payload关系建模
# 2. 流级注意力（FlowAttention）：对5个数据包之间的跨包依赖关系建模
# 
# 输入数据：流MFR矩阵（前5个数据包，每个80 tokens）
# 运行方式：nohup后台运行，日志保存在当前目录
# ============================================================================

# 激活conda环境（Mine环境有CUDA支持）
source /root/anaconda3/bin/activate TrafficAttention

# 脚本目录
SCRIPT_DIR="/root/TrafficAttention/全TLA_pre-training_two-tasks_with_two-attention/"
cd "${SCRIPT_DIR}"

# DATASET_PATH="/root/datasets/pt_packet_datasets_NSP/dataset_pretrain.pt"
# 用随机化之后的数据集求一个心安
DATASET_PATH="/root/datasets/pt_packet_datasets_随机化/dataset_pretrain_随机化.pt"

# 词表路径
VOCAB_PATH="/root/datasets/vocab.txt"

# 配置文件路径
CONFIG_PATH="/root/TrafficAttention/models/bert/base_config.json"

# 输出模型路径
OUTPUT_MODEL_PATH="/root/models/全TLA_pre-trained_model_two_tasks_with_two-attention/全TLA_pre-trained_model_two_tasks_with_two_attention.bin"

# 训练参数
TOTAL_STEPS=500000
SAVE_CHECKPOINT_STEPS=5000
REPORT_STEPS=100
# 显存优化策略：保持batch_size=32，通过梯度累积分散显存压力
# - BATCH_SIZE=32：每次DataLoader加载32个流
# - ACCUMULATION_STEPS=1：不使用梯度累积（直接更新）
# 注意：如果遇到OOM，可以调整为 BATCH_SIZE=16, ACCUMULATION_STEPS=2
#      或 BATCH_SIZE=8, ACCUMULATION_STEPS=4 以保持等效训练效果
BATCH_SIZE=16         # 每个GPU的batch大小
ACCUMULATION_STEPS=4  # 梯度累积步数（如遇OOM可改为2或4）
SEQ_LENGTH=410
LEARNING_RATE=2e-4

# 两个任务的权重
NSP_INDICATING_WEIGHT=0.50  # NSP_Indicating: 50%
MFR_MASKING_WEIGHT=0.50     # MFR-Masking: 50%

# NSP_Indicating任务参数
PACKET_SELECTION_PROB=0.25  # 每个数据包被选中的概率（25%）
PACKET_MASK_RATIO=0.90      # 被选中数据包的mask率（90%）
TOKENS_PER_PACKET=80        # 每个数据包的token数量

# MFR-Masking任务参数
MFR_HEADER_MASK_RATIO=0.80  # Header区域mask率（90%）
MFR_PAYLOAD_MASK_RATIO=0.90 # Payload区域mask率（90%）

# 两级注意力机制参数
USE_PACKET_ATTENTION="--use_packet_attention"  # 启用包级注意力
USE_FLOW_ATTENTION="--use_flow_attention"      # 启用流级注意力
NUM_ATTENTION_HEADS=8                          # 注意力头数
HEADER_SIZE=40                                 # Header token数
WINDOW_SIZE=20                                 # Payload窗口注意力窗口大小

# GPU配置
GPU_RANKS="0 1 2"  # 使用GPU 0、1、2
WORLD_SIZE=3     # 使用3个GPU

# 数据加载优化参数
DATALOADER_NUM_WORKERS=4  # 多进程数据加载workers数量（推荐：2-8）
PREFETCH_FACTOR=2          # 每个worker预加载的batch数量（推荐：2-4）
PIN_MEMORY="--pin_memory"  # 启用pin_memory加速CPU->GPU传输

echo "============================================================================"
echo "多任务预训练 - 整合两个预训练任务 + 两级注意力机制"
echo "============================================================================"
echo "数据集: ${DATASET_PATH}"
echo "词表: ${VOCAB_PATH}"
echo "配置文件: ${CONFIG_PATH}"
echo "输出模型: ${OUTPUT_MODEL_PATH}"
echo ""
echo "训练参数:"
echo "  - Total steps: ${TOTAL_STEPS}"
echo "  - Batch size: ${BATCH_SIZE} (每个GPU)"
echo "  - Accumulation steps: ${ACCUMULATION_STEPS}"
echo "  - Effective batch size: $((BATCH_SIZE * ACCUMULATION_STEPS)) (per GPU)"
echo "  - Sequence length: ${SEQ_LENGTH}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo ""
echo "数据共享策略:"
echo "  - 每次前向传播加载 ${BATCH_SIZE} 个流样本（每个GPU）"
echo "  - 这 ${BATCH_SIZE} 个流同时被两个任务处理"
SAMPLES_PER_FORWARD=$((BATCH_SIZE * 6))
EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUMULATION_STEPS))
EFFECTIVE_SAMPLES=$((EFFECTIVE_BATCH * 6))
echo "  - 每次前向传播产生 ${SAMPLES_PER_FORWARD} 个训练样本 (${BATCH_SIZE}×6)"
echo "  - 有效batch (累积${ACCUMULATION_STEPS}次): ${EFFECTIVE_BATCH} 个流 = ${EFFECTIVE_SAMPLES} 个训练样本"
echo ""
echo "两个预训练任务及其权重:"
echo "  1. NSP_Indicating (Packet Masking): ${NSP_INDICATING_WEIGHT} (50%)"
echo "     - 只mask流中的一个数据包（mask率90%）"
echo "     - 用剩下4个没被mask的数据包预测被mask的数据包"
echo ""
echo "  2. MFR-Masking (Header/Payload Masking): ${MFR_MASKING_WEIGHT} (50%)"
echo "     - 每个流的5个数据包各生成1个MFR样本"
echo "     - 三种关系：h-h (30%), p-p (30%), h-p (40%)"
echo "     - Mask率: h-h和p-p为80%，h-p为95%"
echo "     - 每个数据包: Header(40 tokens) 或 Payload(40 tokens)"
echo ""
echo "两级注意力机制配置:"
echo "  - 包级注意力: 启用"
echo "    * h-h关系: Header内部全局注意力 (${HEADER_SIZE} tokens)"
echo "    * h-p关系: Payload作q，Header作k&v的交叉注意力"
echo "    * p-p关系: Payload内部窗口注意力 (window=${WINDOW_SIZE})"
echo "  - 流级注意力: 启用"
echo "    * Payload池化: 40 tokens -> 4 tokens (每10个平均池化)"
echo "    * 滑动窗口注意力: 首尾包窗口=2，中间包窗口=3"
echo "    * h查h, p查p: 分别对header和payload应用注意力"
echo "  - 注意力头数: ${NUM_ATTENTION_HEADS}"
echo ""
echo "总损失函数: Loss = ${NSP_INDICATING_WEIGHT}*Loss_Indicating + ${MFR_MASKING_WEIGHT}*Loss_MFR"
echo ""
echo "GPU配置:"
echo "  - GPU ranks: ${GPU_RANKS}"
echo "  - World size: ${WORLD_SIZE}"
TOTAL_FORWARD_SAMPLES=$((BATCH_SIZE * WORLD_SIZE * 6))
TOTAL_EFFECTIVE_FLOWS=$((BATCH_SIZE * ACCUMULATION_STEPS * WORLD_SIZE))
TOTAL_EFFECTIVE_SAMPLES=$((TOTAL_EFFECTIVE_FLOWS * 6))
echo "  - 每次前向传播: ${TOTAL_FORWARD_SAMPLES} 个样本 (${BATCH_SIZE}×${WORLD_SIZE}×6)"
echo "  - 每个step有效batch: ${TOTAL_EFFECTIVE_FLOWS} 个流 = ${TOTAL_EFFECTIVE_SAMPLES} 个样本"
echo ""
echo "数据加载优化配置:"
echo "  - 多进程workers: ${DATALOADER_NUM_WORKERS}"
echo "  - 预加载因子: ${PREFETCH_FACTOR}"
echo "  - Pin Memory: 启用"
echo "  - ✓ 数据加载和GPU训练并行执行"
echo "  - ✓ GPU不会等待数据加载，提升训练效率"
echo "============================================================================"
echo ""

# 检查数据集是否存在
if [ ! -f "${DATASET_PATH}" ]; then
    echo "❌ 错误: 数据集文件不存在: ${DATASET_PATH}"
    exit 1
fi

# 检查词表是否存在
if [ ! -f "${VOCAB_PATH}" ]; then
    echo "❌ 错误: 词表文件不存在: ${VOCAB_PATH}"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "❌ 错误: 配置文件不存在: ${CONFIG_PATH}"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR=$(dirname ${OUTPUT_MODEL_PATH})
mkdir -p ${OUTPUT_DIR}

# 日志文件路径（保存在当前脚本目录）
LOG_FILE="${SCRIPT_DIR}/Two-task_with_two-attention_run_pretrain.log"

echo "日志文件: ${LOG_FILE}"
echo ""

# 检查是否存在最新的模型检查点
echo "检查模型检查点..."
LATEST_CHECKPOINT=""
if [ -d "${OUTPUT_DIR}" ]; then
    # 查找最新的检查点文件（按修改时间排序）
    # 支持两种格式：*.bin 和 *.bin-* (检查点格式)
    LATEST_CHECKPOINT=$(ls -t "${OUTPUT_DIR}"/*.bin* 2>/dev/null | grep -E '\.(bin|bin-[0-9]+)$' | head -1)
fi

if [ -n "${LATEST_CHECKPOINT}" ]; then
    echo "✓ 发现最新的模型检查点: ${LATEST_CHECKPOINT}"
    echo ""
    echo "是否从最新的模型检查点继续训练？"
    echo "  1) 是，从最新检查点继续训练（推荐）"
    echo "  2) 否，从头开始训练"
    echo ""
    read -p "请选择 [1/2] (默认: 1): " RESUME_CHOICE
    RESUME_CHOICE=${RESUME_CHOICE:-1}

    if [ "${RESUME_CHOICE}" = "1" ]; then
        echo "✓ 将从最新检查点继续训练"
        PRETRAINED_MODEL_ARG="--pretrained_model_path ${LATEST_CHECKPOINT}"
        echo "  检查点: ${LATEST_CHECKPOINT}"
        # 从文件名提取步数（支持 *.bin-5000 格式）
        CHECKPOINT_STEP=$(basename "${LATEST_CHECKPOINT}" | sed 's/.*-\([0-9]*\)$/\1/')
        if [ -n "${CHECKPOINT_STEP}" ] && [ "${CHECKPOINT_STEP}" -gt 0 ] 2>/dev/null; then
            echo "  恢复步数: ${CHECKPOINT_STEP}"
            REMAINING_STEPS=$((TOTAL_STEPS - CHECKPOINT_STEP))
            echo "  剩余步数: ${REMAINING_STEPS}"
        fi
    else
        echo "✓ 将从头开始训练"
        PRETRAINED_MODEL_ARG=""
    fi
else
    echo "✗ 未发现模型检查点，将从头开始训练"
    PRETRAINED_MODEL_ARG=""
fi

echo ""

# 使用nohup后台运行预训练
echo "启动多任务预训练任务（nohup后台运行，带两级注意力机制）..."
nohup python -u "two-task_with_two-attention_pretrain.py" \
    --dataset_path ${DATASET_PATH} \
    --vocab_path ${VOCAB_PATH} \
    --config_path ${CONFIG_PATH} \
    --output_model_path ${OUTPUT_MODEL_PATH} \
    ${PRETRAINED_MODEL_ARG} \
    --world_size ${WORLD_SIZE} \
    --gpu_ranks ${GPU_RANKS} \
    --total_steps ${TOTAL_STEPS} \
    --save_checkpoint_steps ${SAVE_CHECKPOINT_STEPS} \
    --report_steps ${REPORT_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --embedding word_pos_seg \
    --encoder transformer \
    --tokenizer space \
    --seq_length ${SEQ_LENGTH} \
    --nsp_indicating_weight ${NSP_INDICATING_WEIGHT} \
    --mfr_masking_weight ${MFR_MASKING_WEIGHT} \
    --packet_selection_prob ${PACKET_SELECTION_PROB} \
    --packet_mask_ratio ${PACKET_MASK_RATIO} \
    --tokens_per_packet ${TOKENS_PER_PACKET} \
    --mfr_header_mask_ratio ${MFR_HEADER_MASK_RATIO} \
    --mfr_payload_mask_ratio ${MFR_PAYLOAD_MASK_RATIO} \
    ${USE_PACKET_ATTENTION} \
    ${USE_FLOW_ATTENTION} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --header_size ${HEADER_SIZE} \
    --window_size ${WINDOW_SIZE} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    ${PIN_MEMORY} > "${LOG_FILE}" 2>&1 &

# 获取后台进程ID
PID=$!

# 保存PID到文件，方便后续管理
PID_FILE="${SCRIPT_DIR}/Two-task_with_two-attention_pretrain.pid"
echo ${PID} > ${PID_FILE}

echo ""
echo "============================================================================"
echo "✓ 多任务预训练任务已在后台启动！"
echo "============================================================================"
echo "进程ID (PID): ${PID}"
echo "PID文件: ${PID_FILE}"
echo "日志文件: ${LOG_FILE}"
echo "模型保存路径: ${OUTPUT_MODEL_PATH}"
if [ -n "${LATEST_CHECKPOINT}" ] && [ "${RESUME_CHOICE}" = "1" ]; then
    echo ""
    echo "训练模式: 从检查点恢复"
    echo "  - 检查点: ${LATEST_CHECKPOINT}"
    if [ -n "${CHECKPOINT_STEP}" ] && [ "${CHECKPOINT_STEP}" -gt 0 ]; then
        echo "  - 恢复步数: ${CHECKPOINT_STEP}"
        echo "  - 剩余步数: ${REMAINING_STEPS}"
    fi
else
    echo ""
    echo "训练模式: 从头开始"
fi
echo ""
echo "监控命令:"
echo "  - 查看实时日志: tail -f \"${LOG_FILE}\""
echo "  - 查看进程状态: ps -p ${PID}"
echo "  - 查看GPU使用: nvidia-smi"
echo ""
echo "停止训练:"
echo "  - 停止进程: kill ${PID}"
echo "  - 使用PID文件: kill \$(cat ${PID_FILE})"
echo "  - 强制停止: kill -9 ${PID}"
echo ""
echo "✓ 训练将在后台继续运行，您可以安全关闭终端"
echo "============================================================================"
