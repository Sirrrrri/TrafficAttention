# 多任务预训练 - 整合两个预训练任务

## 概述

这个目录包含了整合两个预训练任务的代码，实现了多任务预训练策略。

### 两个预训练任务及权重

1. **NSP_Indicating (Packet Masking)** - 50%
   - 使用流级别Packet Masking策略（h-h-h-h-h和p-p-p-p-p关系版本）
   - 预训练任务: MLM (仅Masked Language Model)
   - 不使用下一句预测任务(NSP)
   - 输入: 流MFR矩阵（前4个数据包，每个80 tokens）
   - 每个数据包: Header 40 tokens + Payload 40 tokens
   - Sequence Length: 330 tokens (包含CLS、SEP、padding)
   - 总token数: 320 tokens
   
   Masking策略（两种操作，随机选择）:
   
   操作1（h-h-h-h-h关系）:
     - 将流的四个数据包MFR矩阵的payload都设置为CNS（不预测，不用于预测）
     - 随机mask某一个数据包MFR矩阵的header，mask率 90%
     - 用流的四个数据包MFR矩阵中剩余没被mask的header预测被mask的header
     - 每个Header: 40 tokens × 90% = 36 tokens被mask
   
   操作2（p-p-p-p-p关系）:
     - 将流的四个数据包MFR矩阵的header都设置为CNS（不预测，不用于预测）
     - 随机mask某一个数据包MFR矩阵的payload，mask率 90%
     - 用流的四个数据包MFR矩阵中剩余没被mask的payload预测被mask的payload
     - 每个Payload: 40 tokens × 90% = 36 tokens被mask
   
   can_not_see说明:
     - 用特殊标识符[CNS]替代原始数据
     - 被替代的数据无需被预测
     - 也不用于预测被mask的数据
     - 序列长度保持不变

2. **MFR-Masking (Header/Payload Masking)** - 50%
   - 输入数据格式:
     - 对每条流，从4个数据包MFR矩阵中随机选择1个数据包进行处理
     - 每个数据包：80 tokens（Header 40 + Payload 40）
     - 数据包独立处理，不组合成流
     - Sequence Length: 82 tokens (CLS + 80 + SEP)，padding到330以匹配NSP_Indicating
   
   can_not_see说明:
     - 用特殊标识符[CNS]替代原始数据
     - 被替代的数据无需被预测
     - 也不用于预测被mask的数据
     - 三类数据包的序列长度相同
   
   Masking策略分配（按batch_size内的流划分）:
   
   1. h-h关系 (30%的流):
      - 找header-header关系
      - Payload用[CNS]替代（不预测，不用于预测）
      - 对Header随机mask 80%
      - 用Header剩余tokens预测被mask的tokens
      - 每个Header: 40 tokens × 80% = 32 tokens被mask
   
   2. p-p关系 (30%的流):
      - 找payload-payload关系
      - Header用[CNS]替代（不预测，不用于预测）
      - 对Payload随机mask 80%
      - 用Payload剩余tokens预测被mask的tokens
      - 每个Payload: 40 tokens × 80% = 32 tokens被mask
   
   3. h-p关系 (40%的流):
      - 找header-payload关系
      - 对Header随机mask 95%
      - 用Payload预测Header中被mask的tokens
      - 每个Header: 40 tokens × 95% = 38 tokens被mask

### 数据共享策略

- 每个流产生2个样本：1个NSP_Indicating样本 + 1个MFR-Masking样本
- 两个任务共享同一批流数据
- 保证两个任务的样本数始终相等
- 例如：batch_size=8时，每个batch产生16个样本（8个NSP_Indicating + 8个MFR-Masking）

### 总损失函数

```
Loss = 0.50 × Loss_Indicating + 0.50 × Loss_MFR
```

## 输入数据格式

输入数据格式与NSP_Indicating相同：
- 流MFR矩阵：前4个数据包，每个80 tokens
- 格式: `[CLS] + packet1(80) + packet2(80) + packet3(80) + packet4(80)`
- 序列长度: 330 tokens (包含CLS、SEP、padding)
- 数据集路径: `/root/datasets/pt_packet_datasets_NSP/dataset_pretrain.pt`

## 文件结构

```
pre-training_two tasks/
├── two task_pretrain.py              # 主预训练脚本（参数解析和启动）
├── multi_task_trainer.py             # 多任务训练器实现（标准版）
├── multi_task_trainer_optimized.py   # 多任务训练器实现（优化版：异步数据加载）
├── Two task_run_pretrain.sh          # 运行脚本（后台运行）
├── README.md                         # 本文档
└── uer/                              # UER模块（需要创建软链接）
```

## 使用方法

### 1. 准备环境

确保已安装所需的依赖：
```bash
conda activate Mine
```

### 2. 准备数据

确保数据集和词表文件存在：
- 数据集: `/root/datasets/pt_packet_datasets_NSP/dataset_pretrain.pt`
- 词表: `/root/datasets/pt_packet_datasets_NSP/vocab.txt`
- 配置文件: `/root/我的代码和数据集/Mine/models/bert/base_config.json`

### 3. 启动训练

#### 后台运行（推荐）
```bash
cd "/root/我的代码和数据集/Mine/pre-training/pre-training_two tasks"
bash "Two task_run_pretrain.sh"
```

#### 查看日志
```bash
tail -f "Two task_run_pretrain.log"
```

#### 查看GPU使用情况
```bash
nvidia-smi
```

#### 停止训练
```bash
# 使用PID文件停止
kill $(cat pretrain.pid)

# 或者查找进程并停止
ps aux | grep "two task_pretrain.py"
kill <PID>
```

### 4. 训练参数

默认训练参数（可在`Two task_run_pretrain.sh`中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOTAL_STEPS` | 500000 | 总训练步数 |
| `BATCH_SIZE` | 8 | 批次大小（每个GPU） |
| `ACCUMULATION_STEPS` | 4 | 梯度累积步数 |
| `LEARNING_RATE` | 2e-4 | 学习率 |
| `SEQ_LENGTH` | 320 | 序列长度 |
| `SAVE_CHECKPOINT_STEPS` | 5000 | 保存检查点间隔 |
| `REPORT_STEPS` | 100 | 报告间隔 |

任务权重参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NSP_INDICATING_WEIGHT` | 0.50 | NSP_Indicating任务权重 |
| `MFR_MASKING_WEIGHT` | 0.50 | MFR-Masking任务权重 |

### 5. 输出

训练过程中会生成以下文件：
- **模型检查点**: `/root/models/pre-trained_model_two_tasks/pre-trained_model_two_tasks.bin-<step>`
  - 每5000步保存一次
  - 格式: `pre-trained_model_two_tasks.bin-5000`, `pre-trained_model_two_tasks.bin-10000`, ...
  
- **最终模型**: `/root/models/pre-trained_model_two_tasks/pre-trained_model_two_tasks.bin`

- **训练日志**: `Two task_run_pretrain.log`
  - 包含详细的训练进度和统计信息

- **进程ID文件**: `pretrain.pid`
  - 用于管理后台训练进程

## 训练监控

训练日志会显示两个任务的详细统计信息：

```
================================================================================
| Step:    100/500000 | Time:  10.23s |
| nsp_indicating  | Loss: 3.5678 | Acc: 45.67% | Samples:   32 |
| mfr_masking     | Loss: 3.2345 | Acc: 48.90% | Samples:   32 |
--------------------------------------------------------------------------------
```

说明：
- `Loss`: 任务平均损失
- `Acc`: 任务准确率（基于token级别）
- `Samples`: 本次报告中该任务的样本数

## 技术细节

### 多任务学习策略

1. **数据共享**: 两个任务共享同一批流数据
   - 每个step加载N个流样本
   - 每个流被两个任务分别处理，产生2N个训练样本

2. **批次处理**: 支持混合批次
   - 同一个批次包含两个任务的样本
   - 按任务类型分组进行前向传播

3. **损失计算**: 加权损失
   - 每个任务计算自己的损失
   - 最终损失 = Σ(任务损失 × 任务权重 × 样本比例)

### 模型架构

- **Base Model**: Transformer Encoder（共享）
- **任务头**:
  - NSP_Indicating: MLM头（Linear层 → vocab_size）
  - MFR-Masking: MLM头（Linear层 → vocab_size）

### 数据处理

两个任务共享同一个数据加载器`MultiTaskDataset`，它会：
1. 读取流MFR矩阵数据（.pt格式）
2. 根据任务类型生成不同的训练样本
3. 应用相应的masking策略

### Masking策略

1. **NSP_Indicating (Packet Masking)**:
   - 随机选择一个数据包（从4个中选1个）
   - 对选中的数据包mask 90%的tokens
   - 目标：用其他3个数据包预测被mask的数据包

2. **MFR-Masking (Header/Payload Masking)**:
   - 对每个数据包，50%概率mask头部，50%概率mask载荷
   - Mask率: 80%
   - 每个数据包80 tokens = Header(40) + Payload(40)

## 故障排查

### 常见问题

1. **CUDA out of memory**
   - 解决方法: 减小`BATCH_SIZE`（如改为4）或增大`ACCUMULATION_STEPS`

2. **数据集文件不存在**
   - 检查数据集路径是否正确
   - 确保使用NSP_Indicating的数据集

3. **词表文件不存在**
   - 检查词表路径是否正确
   - 确保与数据集匹配

4. **训练loss为NaN**
   - 检查学习率是否过大
   - 检查数据是否有问题

### 调试模式

如果需要调试，可以直接运行Python脚本：
```bash
python -u "two task_pretrain.py" \
    --dataset_path /root/datasets/pt_packet_datasets_NSP/dataset_pretrain.pt \
    --vocab_path /root/datasets/pt_packet_datasets_NSP/vocab.txt \
    --config_path /root/我的代码和数据集/Mine/models/bert/base_config.json \
    --output_model_path /root/models/pre-trained_model_two_tasks/pre-trained_model_two_tasks.bin \
    --world_size 1 \
    --gpu_ranks 0 \
    --total_steps 1000 \
    --batch_size 8 \
    --report_steps 10
```

## 下一步

训练完成后，可以使用预训练模型进行微调：
1. 使用`/root/models/pre-trained_model_two_tasks/pre-trained_model_two_tasks.bin`作为预训练模型
2. 参考`Mine/fine-tuning`目录下的微调脚本
3. 根据具体任务选择合适的微调策略

## 参考

- 原始预训练任务实现:
  - NSP_Indicating: `Mine/pre-training/pre-training_Burst-Predicting_Indicating/`
  - MFR-Masking: `Mine/pre-training/pre-training_MFR-Masking/`

## 更新日志

- **2025-12-23**: 简化为两个预训练任务（移除NSP_Choice），设置权重为50%/50%
- **2025-11-21**: 初始版本，整合三个预训练任务，设置权重为30%/35%/35%
