#!/bin/bash

# ============================================================================
# Two-task微调训练脚本（two-attention版本）- 四个数据集批量训练
# 基于4个数据包结构的流量分类微调（每个数据包80 tokens）
# 双重注意力机制: 包级注意力 + 流级注意力
# ============================================================================
#
# 使用方法:
#   1. 交互式运行（会询问是否开始训练）:
#      bash run_finetuning_two-task_two-attention.sh
#
#   2. 后台运行（自动开始训练，脚本退出后训练继续）:
#      nohup bash run_finetuning_two-task_two-attention.sh > run_all.log 2>&1 &
#      然后可以安全关闭终端，训练任务会继续运行
#
#   3. 查看训练进度:
#      tail -f logs/ISCXTor2016_MFR_train.log
#      tail -f Two-task_fine-tuning_two-attention.log
#
# ============================================================================

# 脚本目录
SCRIPT_DIR="/root/全TLA_Mine/全TLA-fine-tuning_two-task_two-attention/"
cd ${SCRIPT_DIR}

# 预训练模型路径
# TLA在第一层+使用随机化之后的预训练数据
PRETRAINED_MODEL="/root/models/全TLA_pre-trained_model_two_tasks_with_two-attention/全TLA_pre-trained_model_two_tasks_with_two_attention.bin-125000"

# 配置文件路径
CONFIG_PATH="/root/全TLA_Mine/models/bert/base_config.json"

# 词表路径
VOCAB_PATH="/root/datasets_new/pt_packet_datasets_随机化/vocab.txt"

# 训练参数
EPOCHS_NUM=50
BATCH_SIZE=32
LEARNING_RATE=2e-5
SEQ_LENGTH=320  # 4个数据包格式: [CLS] + p1(80) + [SEP] + p2(80) + [SEP] + p3(80) + [SEP] + p4(80)

# 数据集配置（.pt格式数据集基础路径，不含_train.pt等后缀）
# 脚本会自动查找 _train.pt, _dev.pt, _test.pt 三个文件
declare -A DATASETS
DATASETS["NUDT_MobileTraffic Dataset_all_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_NUDT_MobileTraffic Dataset_all_MFR"
DATASETS["USTC-TFC2016_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_USTC-TFC2016_MFR"
DATASETS["ISCXVPN2016_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_ISCXVPN2016_MFR"
DATASETS["CICIoT2023_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_CICIoT2023_MFR"
DATASETS["ISCXVPN2016-APP_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_ISCXVPN2016-APP_MFR"

# 输出模型基础目录
OUTPUT_BASE_DIR="/root/models/全TLA_fine-tuning_two-task_two-attention"

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}"

# 日志目录
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 主日志文件
MAIN_LOG="${SCRIPT_DIR}/Two-task_fine-tuning_two-attention.log"

# 检查是否有训练进程正在运行
echo "============================================================================"
echo "检查现有训练进程..."
echo "============================================================================"
RUNNING_PROCS=$(ps aux | grep "Two-task_fine-tuning_two-attention.py" | grep -v grep | wc -l)
if [ ${RUNNING_PROCS} -gt 0 ]; then
    echo "⚠️  警告: 发现 ${RUNNING_PROCS} 个训练进程正在运行！"
    echo ""
    ps aux | grep "Two-task_fine-tuning_two-attention.py" | grep -v grep
    echo ""
    echo "请选择操作："
    echo "  1) 停止所有现有训练任务并重新开始"
    echo "  2) 查看现有任务状态后退出"
    echo "  3) 继续添加新任务（不推荐，会导致资源冲突）"
    read -p "请输入选项 (1/2/3): " -n 1 -r
    echo ""
    
    if [[ $REPLY == "1" ]]; then
        echo "正在停止所有训练任务..."
        pkill -f "Two-task_fine-tuning_two-attention.py"
        sleep 3
        echo "✓ 已停止所有训练任务"
    elif [[ $REPLY == "2" ]]; then
        echo "退出脚本"
        exit 0
    else
        echo "⚠️  继续运行（可能导致资源冲突）"
    fi
    echo ""
fi

# 备份旧日志（如果存在）
if [ -f "${MAIN_LOG}" ]; then
    BACKUP_LOG="${MAIN_LOG}.$(date +%Y%m%d_%H%M%S).bak"
    echo "备份旧日志: ${BACKUP_LOG}"
    mv "${MAIN_LOG}" "${BACKUP_LOG}"
fi

echo "============================================================================"
echo "Two-task微调训练（two-attention版本）- 四个数据集串行训练（依次执行）"
echo "============================================================================"
echo "执行模式: 串行训练（一个任务完成后才开始下一个）"
echo "预训练模型: ${PRETRAINED_MODEL}"
echo "配置文件: ${CONFIG_PATH}"
echo "词表文件: ${VOCAB_PATH}"
echo "训练参数: epochs=${EPOCHS_NUM}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}, seq_length=${SEQ_LENGTH}"
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "主日志文件: ${MAIN_LOG}"
echo "============================================================================"
echo ""

# 记录开始时间
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "任务开始时间: ${START_TIME}" | tee ${MAIN_LOG}

# 检查预训练模型是否存在
if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "❌ 错误: 预训练模型文件不存在: ${PRETRAINED_MODEL}" | tee -a ${MAIN_LOG}
    echo "   请检查路径是否正确" | tee -a ${MAIN_LOG}
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "❌ 错误: 配置文件不存在: ${CONFIG_PATH}" | tee -a ${MAIN_LOG}
    exit 1
fi

# 检查词表文件是否存在
if [ ! -f "${VOCAB_PATH}" ]; then
    echo "❌ 错误: 词表文件不存在: ${VOCAB_PATH}" | tee -a ${MAIN_LOG}
    exit 1
fi

# 函数：训练单个数据集
train_dataset() {
    local dataset_name=$1
    local dataset_base_path=$2  # 基础路径（不含_train.pt等后缀）
    
    echo "" | tee -a ${MAIN_LOG}
    echo "==========================================" | tee -a ${MAIN_LOG}
    echo "开始训练: ${dataset_name}" | tee -a ${MAIN_LOG}
    echo "==========================================" | tee -a ${MAIN_LOG}
    
    # 根据基础路径生成三个数据集的路径
    local train_path="${dataset_base_path}_train.pt"
    local dev_path="${dataset_base_path}_dev.pt"
    local test_path="${dataset_base_path}_test.pt"
    
    # 检查三个数据集文件是否存在
    local missing_files=""
    if [ ! -f "${train_path}" ]; then
        missing_files="${missing_files} ${train_path}"
    fi
    if [ ! -f "${dev_path}" ]; then
        missing_files="${missing_files} ${dev_path}"
    fi
    if [ ! -f "${test_path}" ]; then
        missing_files="${missing_files} ${test_path}"
    fi
    
    if [ -n "${missing_files}" ]; then
        echo "❌ 错误: 数据集文件不存在:" | tee -a ${MAIN_LOG}
        echo "   训练集: ${train_path}" | tee -a ${MAIN_LOG}
        echo "   验证集: ${dev_path}" | tee -a ${MAIN_LOG}
        echo "   测试集: ${test_path}" | tee -a ${MAIN_LOG}
        echo "   缺失文件:${missing_files}" | tee -a ${MAIN_LOG}
        echo "   请确保.pt格式数据集已生成并位于正确路径" | tee -a ${MAIN_LOG}
        return 1
    fi
    
    # 输出模型路径
    local output_dir="${OUTPUT_BASE_DIR}/${dataset_name}"
    local output_model="${output_dir}/finetuned_model.bin"
    
    # 创建输出目录
    mkdir -p "${output_dir}"
    
    # 单个数据集的日志文件
    local log_file="${LOG_DIR}/${dataset_name}_train.log"
    
    echo "训练集: ${train_path}" | tee -a ${MAIN_LOG}
    echo "验证集: ${dev_path}" | tee -a ${MAIN_LOG}
    echo "测试集: ${test_path}" | tee -a ${MAIN_LOG}
    echo "输出模型: ${output_model}" | tee -a ${MAIN_LOG}
    echo "日志文件: ${log_file}" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 构建训练命令
    local cmd="python -u Two-task_fine-tuning_two-attention.py \
        --pretrained_model_path '${PRETRAINED_MODEL}' \
        --vocab_path '${VOCAB_PATH}' \
        --config_path '${CONFIG_PATH}' \
        --train_path '${train_path}' \
        --dev_path '${dev_path}' \
        --output_model_path '${output_model}' \
        --epochs_num ${EPOCHS_NUM} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --embedding word_pos_seg \
        --encoder transformer \
        --mask fully_visible \
        --pooling first \
        --tokenizer bert \
        --seq_length ${SEQ_LENGTH} \
        --early_stopping \
        --patience 6 \
        --min_delta 0.001"
    
    # 添加测试路径（测试集必须存在）
    cmd="${cmd} --test_path '${test_path}'"
    
    # 串行训练：使用nohup后台运行，但等待任务完成才继续
    echo "启动训练任务（串行执行，等待完成）..." | tee -a ${MAIN_LOG}
    echo "训练命令: ${cmd}" >> ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 记录开始时间
    local task_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  开始时间: ${task_start_time}" | tee -a ${MAIN_LOG}
    
    # 使用nohup后台运行，但使用wait等待完成（串行执行）
    nohup bash -c "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    
    echo "  进程ID: ${pid}" | tee -a ${MAIN_LOG}
    echo "  日志文件: ${log_file}" | tee -a ${MAIN_LOG}
    echo "  ⏳ 等待此任务完成后才会开始下一个任务..." | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 等待进程完成（阻塞直到完成）
    wait ${pid}
    local exit_code=$?
    
    local task_end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "" | tee -a ${MAIN_LOG}
    echo "  结束时间: ${task_end_time}" | tee -a ${MAIN_LOG}
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✓ 训练任务完成成功" | tee -a ${MAIN_LOG}
    else
        echo "✗ 训练任务失败 (退出码: ${exit_code})" | tee -a ${MAIN_LOG}
    fi
    
    echo "  数据集: ${dataset_name}" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    return ${exit_code}
}

# 主循环：遍历所有数据集
echo "准备训练以下数据集:" | tee -a ${MAIN_LOG}
for dataset_name in "${!DATASETS[@]}"; do
    dataset_base="${DATASETS[${dataset_name}]}"
    echo "  - ${dataset_name}:" | tee -a ${MAIN_LOG}
    echo "    训练集: ${dataset_base}_train.pt" | tee -a ${MAIN_LOG}
    echo "    验证集: ${dataset_base}_dev.pt" | tee -a ${MAIN_LOG}
    echo "    测试集: ${dataset_base}_test.pt" | tee -a ${MAIN_LOG}
done
echo "" | tee -a ${MAIN_LOG}

# 检测是否为交互式终端
if [ -t 0 ]; then
    # 交互式模式：询问是否继续
    read -p "是否开始训练? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "训练已取消" | tee -a ${MAIN_LOG}
        exit 0
    fi
else
    # 非交互式模式：自动开始
    echo "非交互式模式：自动开始训练" | tee -a ${MAIN_LOG}
fi

# 串行训练函数（将在后台执行）
run_serial_training() {
    local start_time=$(date +%s)
    
    echo "============================================================================" | tee -a ${MAIN_LOG}
    echo "开始串行训练所有数据集（一次训练一个，避免GPU内存不足）..." | tee -a ${MAIN_LOG}
    echo "============================================================================" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 记录成功和失败的数据集
    local success_count=0
    local failed_count=0
    local success_list=""
    local failed_list=""
    
    # 统计数据集总数
    local total_datasets=${#DATASETS[@]}
    local current_index=0
    
    for dataset_name in "${!DATASETS[@]}"; do
        current_index=$((current_index + 1))
        dataset_dir="${DATASETS[${dataset_name}]}"
        
        echo "============================================================================" | tee -a ${MAIN_LOG}
        echo "进度: [${current_index}/${total_datasets}] 正在训练: ${dataset_name}" | tee -a ${MAIN_LOG}
        echo "============================================================================" | tee -a ${MAIN_LOG}
        echo "" | tee -a ${MAIN_LOG}
        
        # 执行训练
        if train_dataset "${dataset_name}" "${dataset_dir}"; then
            success_count=$((success_count + 1))
            success_list="${success_list}    ✓ ${dataset_name}\n"
        else
            failed_count=$((failed_count + 1))
            failed_list="${failed_list}    ✗ ${dataset_name}\n"
            echo "⚠️  ${dataset_name} 训练失败，继续下一个数据集..." | tee -a ${MAIN_LOG}
        fi
        
        echo "" | tee -a ${MAIN_LOG}
        echo "已完成: ${current_index}/${total_datasets} 个数据集" | tee -a ${MAIN_LOG}
        echo "" | tee -a ${MAIN_LOG}
        
        # 在数据集之间短暂休息，释放GPU内存（串行执行）
        if [ ${current_index} -lt ${total_datasets} ]; then
            echo "" | tee -a ${MAIN_LOG}
            echo "⏸️  任务间隔：等待5秒释放GPU内存..." | tee -a ${MAIN_LOG}
            echo "   已完成 ${current_index}/${total_datasets} 个数据集，准备开始下一个..." | tee -a ${MAIN_LOG}
            echo "" | tee -a ${MAIN_LOG}
            sleep 5
        fi
    done
    
    # 训练完成后的总结
    echo "" | tee -a ${MAIN_LOG}
    echo "============================================================================" | tee -a ${MAIN_LOG}
    echo "✓ 所有训练任务已完成!" | tee -a ${MAIN_LOG}
    echo "============================================================================" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 显示训练结果摘要
    echo "训练结果摘要:" | tee -a ${MAIN_LOG}
    echo "  成功: ${success_count} 个数据集" | tee -a ${MAIN_LOG}
    if [ ${success_count} -gt 0 ]; then
        echo -e "${success_list}" | tee -a ${MAIN_LOG}
    fi
    
    if [ ${failed_count} -gt 0 ]; then
        echo "  失败: ${failed_count} 个数据集" | tee -a ${MAIN_LOG}
        echo -e "${failed_list}" | tee -a ${MAIN_LOG}
    fi
    echo "" | tee -a ${MAIN_LOG}
    
    echo "日志文件位置:" | tee -a ${MAIN_LOG}
    echo "  主日志: ${MAIN_LOG}" | tee -a ${MAIN_LOG}
    for dataset_name in "${!DATASETS[@]}"; do
        echo "  ${dataset_name}: ${LOG_DIR}/${dataset_name}_train.log" | tee -a ${MAIN_LOG}
    done
    echo "" | tee -a ${MAIN_LOG}
    
    echo "输出模型位置:" | tee -a ${MAIN_LOG}
    for dataset_name in "${!DATASETS[@]}"; do
        model_path="${OUTPUT_BASE_DIR}/${dataset_name}/finetuned_model.bin"
        if [ -f "${model_path}" ]; then
            echo "  ✓ ${dataset_name}: ${model_path}" | tee -a ${MAIN_LOG}
        else
            echo "  ✗ ${dataset_name}: ${model_path} (未生成)" | tee -a ${MAIN_LOG}
        fi
    done
    echo "" | tee -a ${MAIN_LOG}
    
    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    local END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "脚本结束时间: ${END_TIME}" | tee -a ${MAIN_LOG}
    echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒" | tee -a ${MAIN_LOG}
    echo "============================================================================" | tee -a ${MAIN_LOG}
}

# 创建后台训练脚本
BACKGROUND_SCRIPT="${SCRIPT_DIR}/background_training_runner.sh"

cat > ${BACKGROUND_SCRIPT} << 'EOFSCRIPT'
#!/bin/bash

# 导入变量
SCRIPT_DIR="/root/全TLA_Mine/全TLA-fine-tuning_two-task_two-attention/"
cd ${SCRIPT_DIR}

# 数据集配置（基础路径，不含_train.pt等后缀）
declare -A DATASETS
DATASETS["NUDT_MobileTraffic Dataset_all_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_NUDT_MobileTraffic Dataset_all_MFR"
DATASETS["USTC-TFC2016_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_USTC-TFC2016_MFR"
DATASETS["ISCXVPN2016_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_ISCXVPN2016_MFR"
DATASETS["CICIoT2023_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_CICIoT2023_MFR"
DATASETS["ISCXVPN2016-APP_MFR"]="/root/datasets_new/fine-tuning数据集/finetuning_ISCXVPN2016-APP_MFR"


OUTPUT_BASE_DIR="/root/models/全TLA_fine-tuning_two-task_two-attention"
LOG_DIR="${SCRIPT_DIR}/logs"
MAIN_LOG="${SCRIPT_DIR}/Two-task_fine-tuning_two-attention.log"
PRETRAINED_MODEL="/root/models/全TLA_pre-trained_model_two_tasks_with_two-attention/全TLA_pre-trained_model_two_tasks_with_two_attention.bin-125000"
VOCAB_PATH="/root/datasets_new/pt_packet_datasets_随机化/vocab.txt"
CONFIG_PATH="/root/全TLA_Mine/models/bert/base_config.json"
EPOCHS_NUM=50
BATCH_SIZE=32
LEARNING_RATE=2e-5
SEQ_LENGTH=320

# 训练单个数据集的函数
train_dataset() {
    local dataset_name=$1
    local dataset_base_path=$2  # 基础路径（不含_train.pt等后缀）
    
    echo "" | tee -a ${MAIN_LOG}
    echo "==========================================" | tee -a ${MAIN_LOG}
    echo "开始训练: ${dataset_name}" | tee -a ${MAIN_LOG}
    echo "==========================================" | tee -a ${MAIN_LOG}
    
    # 根据基础路径生成三个数据集的路径
    local train_path="${dataset_base_path}_train.pt"
    local dev_path="${dataset_base_path}_dev.pt"
    local test_path="${dataset_base_path}_test.pt"
    
    # 检查三个数据集文件是否存在
    local missing_files=""
    if [ ! -f "${train_path}" ]; then
        missing_files="${missing_files} ${train_path}"
    fi
    if [ ! -f "${dev_path}" ]; then
        missing_files="${missing_files} ${dev_path}"
    fi
    if [ ! -f "${test_path}" ]; then
        missing_files="${missing_files} ${test_path}"
    fi
    
    if [ -n "${missing_files}" ]; then
        echo "❌ 错误: 数据集文件不存在:" | tee -a ${MAIN_LOG}
        echo "   训练集: ${train_path}" | tee -a ${MAIN_LOG}
        echo "   验证集: ${dev_path}" | tee -a ${MAIN_LOG}
        echo "   测试集: ${test_path}" | tee -a ${MAIN_LOG}
        echo "   缺失文件:${missing_files}" | tee -a ${MAIN_LOG}
        echo "   请确保.pt格式数据集已生成并位于正确路径" | tee -a ${MAIN_LOG}
        return 1
    fi
    local output_dir="${OUTPUT_BASE_DIR}/${dataset_name}"
    local output_model="${output_dir}/finetuned_model.bin"
    
    mkdir -p "${output_dir}"
    local log_file="${LOG_DIR}/${dataset_name}_train.log"
    
    echo "训练集: ${train_path}" | tee -a ${MAIN_LOG}
    echo "验证集: ${dev_path}" | tee -a ${MAIN_LOG}
    echo "测试集: ${test_path}" | tee -a ${MAIN_LOG}
    echo "输出模型: ${output_model}" | tee -a ${MAIN_LOG}
    echo "日志文件: ${log_file}" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    local cmd="python -u Two-task_fine-tuning_two-attention.py \
        --pretrained_model_path '${PRETRAINED_MODEL}' \
        --vocab_path '${VOCAB_PATH}' \
        --config_path '${CONFIG_PATH}' \
        --train_path '${train_path}' \
        --dev_path '${dev_path}' \
        --output_model_path '${output_model}' \
        --epochs_num ${EPOCHS_NUM} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --embedding word_pos_seg \
        --encoder transformer \
        --mask fully_visible \
        --pooling first \
        --tokenizer bert \
        --seq_length ${SEQ_LENGTH} \
        --early_stopping \
        --patience 6 \
        --min_delta 0.001 \
        --test_path '${test_path}'"
    
    echo "启动训练任务（串行执行，等待完成）..." | tee -a ${MAIN_LOG}
    echo "训练命令: ${cmd}" >> ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    local task_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  开始时间: ${task_start_time}" | tee -a ${MAIN_LOG}
    
    nohup bash -c "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    
    echo "  进程ID: ${pid}" | tee -a ${MAIN_LOG}
    echo "  日志文件: ${log_file}" | tee -a ${MAIN_LOG}
    echo "  ⏳ 等待此任务完成后才会开始下一个任务..." | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 等待进程完成（阻塞直到完成）
    wait ${pid}
    local exit_code=$?
    
    local task_end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "" | tee -a ${MAIN_LOG}
    echo "  结束时间: ${task_end_time}" | tee -a ${MAIN_LOG}
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✓ 训练任务完成成功" | tee -a ${MAIN_LOG}
    else
        echo "✗ 训练任务失败 (退出码: ${exit_code})" | tee -a ${MAIN_LOG}
    fi
    
    echo "  数据集: ${dataset_name}" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    return ${exit_code}
}

# 串行训练主逻辑
start_time=$(date +%s)

echo "============================================================================" | tee -a ${MAIN_LOG}
echo "开始串行训练所有数据集" | tee -a ${MAIN_LOG}
echo "执行策略: 依次执行，每次只运行一个任务" | tee -a ${MAIN_LOG}
echo "说明: 第1个任务完成后才会开始第2个，以此类推，避免GPU内存不足" | tee -a ${MAIN_LOG}
echo "============================================================================" | tee -a ${MAIN_LOG}
echo "" | tee -a ${MAIN_LOG}

success_count=0
failed_count=0
success_list=""
failed_list=""

total_datasets=${#DATASETS[@]}
current_index=0

for dataset_name in "${!DATASETS[@]}"; do
    current_index=$((current_index + 1))
    dataset_dir="${DATASETS[${dataset_name}]}"
    
    echo "============================================================================" | tee -a ${MAIN_LOG}
    echo "进度: [${current_index}/${total_datasets}] 正在训练: ${dataset_name}" | tee -a ${MAIN_LOG}
    echo "⏳ 串行执行模式：等待此任务完成才会开始下一个" | tee -a ${MAIN_LOG}
    echo "============================================================================" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    if train_dataset "${dataset_name}" "${dataset_dir}"; then
        success_count=$((success_count + 1))
        success_list="${success_list}    ✓ ${dataset_name}\n"
    else
        failed_count=$((failed_count + 1))
        failed_list="${failed_list}    ✗ ${dataset_name}\n"
        echo "⚠️  ${dataset_name} 训练失败，继续下一个数据集..." | tee -a ${MAIN_LOG}
    fi
    
    echo "" | tee -a ${MAIN_LOG}
    echo "已完成: ${current_index}/${total_datasets} 个数据集" | tee -a ${MAIN_LOG}
    echo "" | tee -a ${MAIN_LOG}
    
    # 在数据集之间短暂休息，释放GPU内存（串行执行）
    if [ ${current_index} -lt ${total_datasets} ]; then
        echo "" | tee -a ${MAIN_LOG}
        echo "⏸️  任务间隔：等待5秒释放GPU内存..." | tee -a ${MAIN_LOG}
        echo "   已完成 ${current_index}/${total_datasets} 个数据集，准备开始下一个..." | tee -a ${MAIN_LOG}
        echo "" | tee -a ${MAIN_LOG}
        sleep 5
    fi
done

# 训练完成总结
echo "" | tee -a ${MAIN_LOG}
echo "============================================================================" | tee -a ${MAIN_LOG}
echo "✓ 所有训练任务已完成!" | tee -a ${MAIN_LOG}
echo "============================================================================" | tee -a ${MAIN_LOG}
echo "" | tee -a ${MAIN_LOG}

echo "训练结果摘要:" | tee -a ${MAIN_LOG}
echo "  成功: ${success_count} 个数据集" | tee -a ${MAIN_LOG}
if [ ${success_count} -gt 0 ]; then
    echo -e "${success_list}" | tee -a ${MAIN_LOG}
fi

if [ ${failed_count} -gt 0 ]; then
    echo "  失败: ${failed_count} 个数据集" | tee -a ${MAIN_LOG}
    echo -e "${failed_list}" | tee -a ${MAIN_LOG}
fi
echo "" | tee -a ${MAIN_LOG}

echo "输出模型位置:" | tee -a ${MAIN_LOG}
for dataset_name in "${!DATASETS[@]}"; do
    model_path="${OUTPUT_BASE_DIR}/${dataset_name}/finetuned_model.bin"
    if [ -f "${model_path}" ]; then
        echo "  ✓ ${dataset_name}: ${model_path}" | tee -a ${MAIN_LOG}
    else
        echo "  ✗ ${dataset_name}: ${model_path} (未生成)" | tee -a ${MAIN_LOG}
    fi
done
echo "" | tee -a ${MAIN_LOG}

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "脚本结束时间: ${END_TIME}" | tee -a ${MAIN_LOG}
echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒" | tee -a ${MAIN_LOG}
echo "============================================================================" | tee -a ${MAIN_LOG}

# 清理自身
rm -f ${SCRIPT_DIR}/background_training_runner.sh
EOFSCRIPT

chmod +x ${BACKGROUND_SCRIPT}

# 在后台运行训练脚本
echo "============================================================================" | tee -a ${MAIN_LOG}
echo "启动后台串行训练任务..." | tee -a ${MAIN_LOG}
echo "============================================================================" | tee -a ${MAIN_LOG}
echo "" | tee -a ${MAIN_LOG}

nohup ${BACKGROUND_SCRIPT} >> ${MAIN_LOG} 2>&1 &
MASTER_PID=$!

echo "✓ 后台串行训练任务已启动" | tee -a ${MAIN_LOG}
echo "  主控进程ID: ${MASTER_PID}" | tee -a ${MAIN_LOG}
echo "  主日志文件: ${MAIN_LOG}" | tee -a ${MAIN_LOG}
echo "" | tee -a ${MAIN_LOG}

echo "数据集训练日志:" | tee -a ${MAIN_LOG}
for dataset_name in "${!DATASETS[@]}"; do
    echo "  ${dataset_name}: ${LOG_DIR}/${dataset_name}_train.log" | tee -a ${MAIN_LOG}
done
echo "" | tee -a ${MAIN_LOG}

echo "监控命令:" | tee -a ${MAIN_LOG}
echo "  查看主日志: tail -f ${MAIN_LOG}" | tee -a ${MAIN_LOG}
echo "  查看训练日志: tail -f ${LOG_DIR}/[数据集名称]_train.log" | tee -a ${MAIN_LOG}
echo "  查看进程状态: ps aux | grep Two-task_fine-tuning_two-attention" | tee -a ${MAIN_LOG}
echo "  查看GPU使用: nvidia-smi" | tee -a ${MAIN_LOG}
echo "" | tee -a ${MAIN_LOG}

echo "停止训练命令:" | tee -a ${MAIN_LOG}
echo "  停止所有训练（推荐，会停止父进程和子进程）:" | tee -a ${MAIN_LOG}
echo "    pkill -9 -f 'Two-task_fine-tuning_two-attention'" | tee -a ${MAIN_LOG}
echo "    pkill -9 -f 'background_training_runner'" | tee -a ${MAIN_LOG}
echo "  或者只停止主控进程（子进程可能继续运行）:" | tee -a ${MAIN_LOG}
echo "    kill -9 ${MASTER_PID}" | tee -a ${MAIN_LOG}
echo "" | tee -a ${MAIN_LOG}

echo "============================================================================" | tee -a ${MAIN_LOG}
echo "✓ 脚本已启动后台串行训练任务，现在可以关闭终端" | tee -a ${MAIN_LOG}
echo "  训练任务将依次在后台完成，请通过日志文件查看进度" | tee -a ${MAIN_LOG}
echo "============================================================================" | tee -a ${MAIN_LOG}
