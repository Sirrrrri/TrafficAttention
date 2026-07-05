# 多任务预训练模型架构文档

## 概述

本文档详细描述了多任务预训练模型的新架构设计���该架构专门为网络流量数据的特殊结构优化，采用两级注意力机制（包级和流级），完全去掉了标准的自注意力层。

---

## 架构总体设计

### 新架构流程图

```
输入序列 (330 tokens)
    ↓
[Embedding Layer]
    ↓
[CLS] + Packet1(80) + Packet2(80) + Packet3(80) + Packet4(80) + [SEP] + Padding
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[包级注意力堆栈 - 4组串行3层结构]
  每组包含3层串行注意力：
  - 第1层：h-h关系（header内部全局注意力）
  - 第2层：h-p关系（payload从header获取信息的交叉注意力）
  - 第3层：p-p关系（payload内部窗口注意力）
  
  重复4次，共12层：
  [组1] 第1层(h-h) → 第2层(h-p) → 第3层(p-p)
  [组2] 第1层(h-h) → 第2层(h-p) → 第3层(p-p)
  [组3] 第1层(h-h) → 第2层(h-p) → 第3层(p-p)
  [组4] 第1层(h-h) → 第2层(h-p) → 第3层(p-p)
    ↓
[Pooling Layer]
  - 对每个packet的payload进行mean pooling
  - Header保持不变
    ↓
[8层 FlowAttention 堆栈]
  - 第1层：处理4个packet之间的跨包依赖
  - 第2层：同上
  - ...
  - 第8层：同上
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[Task Heads]
  ├─ NSP_Indicating MLM Head
  └─ MFR-Masking MLM Head
    ↓
输出 (logits for MLM prediction)
```

---

## 核心组件详解

### 1. Embedding Layer

**输入**：
- `src`: [batch_size, seq_length=330] - 输入token序列
- `seg`: [batch_size, seq_length=330] - segment标记

**输出**：
- [batch_size, 330, hidden_size] - 嵌入表示

**说明**：
- 使用标准的token embedding + position embedding
- 将token序列转换为连续的向量表示

---

### 2. PacketAttentionStack（4组串行3层结构）

#### 2.1 整体结构

```
输入：4个独立的packet embeddings
  ├─ Packet1: [batch_size, 80, hidden_size]
  ├─ Packet2: [batch_size, 80, hidden_size]
  ├─ Packet3: [batch_size, 80, hidden_size]
  └─ Packet4: [batch_size, 80, hidden_size]
    ↓
对每个packet独立应用4组串行3层注意力
    ↓
[组1 - 串行3层]
  ├─ 第1层：HeaderAttentionLayer (h-h关系)
  ├─ 第2层：CrossAttentionLayer (h-p关系)
  └─ 第3层：PayloadAttentionLayer (p-p关系)
    ↓
[组2 - 串行3层]
  ├─ 第1层：HeaderAttentionLayer (h-h关系)
  ├─ 第2层：CrossAttentionLayer (h-p关系)
  └─ 第3层：PayloadAttentionLayer (p-p关系)
    ↓
[组3 - 串行3层]
  ├─ 第1层：HeaderAttentionLayer (h-h关系)
  ├─ 第2层：CrossAttentionLayer (h-p关系)
  └─ 第3层：PayloadAttentionLayer (p-p关系)
    ↓
[组4 - 串行3层]
  ├─ 第1层：HeaderAttentionLayer (h-h关系)
  ├─ 第2层：CrossAttentionLayer (h-p关系)
  └─ 第3层：PayloadAttentionLayer (p-p关系)
    ↓
输出：4个处理后的packet embeddings
```

#### 2.2 三层串行注意力详解

**第1层：HeaderAttentionLayer (h-h关系)**

对Header的40个tokens应用全局自注意力，Payload保持不变：

```
输入: [batch_size, 80, hidden_size]
  ├─ Header: [batch_size, 40, hidden_size]
  └─ Payload: [batch_size, 40, hidden_size]
    ↓
[Header全局自注意力]
  Query = Linear(Header)
  Key = Linear(Header)
  Value = Linear(Header)
  Attention(Q, K, V) → Header_output
    ↓
[残差连接 + LayerNorm]
  Header_final = LayerNorm(Header + Header_output)
  Payload保持不变
    ↓
输出: [Header_final, Payload]
```

**第2层：CrossAttentionLayer (h-p关系)**

Payload作为Query，从Header获取信息，Header保持不变：

```
输入: [batch_size, 80, hidden_size]
  ├─ Header: [batch_size, 40, hidden_size] (来自第1层)
  └─ Payload: [batch_size, 40, hidden_size]
    ↓
[Payload-Header交叉注意力]
  Query = Linear(Payload)
  Key = Linear(Header)
  Value = Linear(Header)
  Attention(Q, K, V) → Cross_output
    ↓
[残差连接 + LayerNorm]
  Payload_final = LayerNorm(Payload + Cross_output)
  Header保持不变
    ↓
输出: [Header, Payload_final]
```

**第3层：PayloadAttentionLayer (p-p关系)**

对Payload应用窗口自注意力，Header保持不变：

```
输入: [batch_size, 80, hidden_size]
  ├─ Header: [batch_size, 40, hidden_size] (来自第2层)
  └─ Payload: [batch_size, 40, hidden_size] (来自第2层)
    ↓
[Payload窗口自注意力]
  Query = Linear(Payload)
  Key = Linear(Payload)
  Value = Linear(Payload)
  
  创建窗口mask (window_size=20):
    对每个位置i，只能看到[i-10, i+10]范围内的tokens
  
  Attention(Q, K, V, mask) → Payload_output
    ↓
[残差连接 + LayerNorm]
  Payload_final = LayerNorm(Payload + Payload_output)
  Header保持不变
    ↓
输出: [Header, Payload_final]
```

#### 2.3 串行处理流程

每个packet经过以下串行处理（共12层）：

```
Packet_0 (初始embedding)
  ↓
[组1]
  Packet_1 = HeaderAttentionLayer(Packet_0)      # 第1层: h-h
  Packet_2 = CrossAttentionLayer(Packet_1)       # 第2层: h-p
  Packet_3 = PayloadAttentionLayer(Packet_2)     # 第3层: p-p
  ↓
[组2]
  Packet_4 = HeaderAttentionLayer(Packet_3)      # 第4层: h-h
  Packet_5 = CrossAttentionLayer(Packet_4)       # 第5层: h-p
  Packet_6 = PayloadAttentionLayer(Packet_5)     # 第6层: p-p
  ↓
[组3]
  Packet_7 = HeaderAttentionLayer(Packet_6)      # 第7层: h-h
  Packet_8 = CrossAttentionLayer(Packet_7)       # 第8层: h-p
  Packet_9 = PayloadAttentionLayer(Packet_8)     # 第9层: p-p
  ↓
[组4]
  Packet_10 = HeaderAttentionLayer(Packet_9)     # 第10层: h-h
  Packet_11 = CrossAttentionLayer(Packet_10)     # 第11层: h-p
  Packet_12 = PayloadAttentionLayer(Packet_11)   # 第12层: p-p
  ↓
Packet_final (输出)
```

**关键特点**：
- **串行执行**：每层的输出作为下一层的输入
- **信息流动**：h-h → h-p → p-p，逐步建模header和payload的关系
- **重复4次**：形成深层的特征提取能力
- **总共12层**：4组 × 3层/组 = 12层包级注意力

#### 2.4 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_layers | 4 | 组数（每组3层串行注意力） |
| hidden_size | 768 | 隐层维度 |
| num_attention_heads | 8 | 多头注意力头数 |
| header_size | 40 | Header token数 |
| payload_size | 40 | Payload token数 |
| window_size | 20 | 窗口注意力窗口大小 |
| dropout | 0.1 | Dropout比例 |

---

### 3. PoolingLayer（池化层）

#### 3.1 功能

在PacketAttention和FlowAttention之间进行池化，减少序列长度同时保留关键信息。

#### 3.2 池化方式

**对每个packet分别处理**：

```
输入 packet: [batch_size, 80, hidden_size]
  ├─ Header: [batch_size, 40, hidden_size] → 保持不变
  └─ Payload: [batch_size, 40, hidden_size]
      ↓
    Reshape: [batch_size, 4, 10, hidden_size]
      ↓
    Mean Pooling (dim=2): [batch_size, 4, hidden_size]
      ↓
    Reshape: [batch_size, 40, hidden_size]
      ↓
输出 packet: [batch_size, 80, hidden_size]
```

**说明**：
- Payload的40个tokens被分成4组，每组10个tokens
- 对每组进行mean pooling，得到4个pooled tokens
- 再reshape回40个tokens的形状
- Header保持原样，不进行池化

**优势**：
- 减少计算量（特别是在FlowAttention中）
- 保留payload的关键信息
- 保持与原始序列长度的兼容性

---

### 4. FlowAttentionStack（8层）

#### 4.1 整体结构

```
输入：4个pooled packet embeddings
  ├─ Packet1: [batch_size, 80, hidden_size]
  ├─ Packet2: [batch_size, 80, hidden_size]
  ├─ Packet3: [batch_size, 80, hidden_size]
  └─ Packet4: [batch_size, 80, hidden_size]
    ↓
[Layer 1 - FlowAttention]
  ├─ Header Attention: 滑动窗口注意力
  └─ Payload Attention: 滑动窗口注意力
    ↓
[Layer 2 - FlowAttention] (同Layer 1)
    ↓
...
    ↓
[Layer 8 - FlowAttention] (同Layer 1)
    ↓
输出：4个处理后的packet embeddings
```

#### 4.2 FlowAttention 内部机制

每层FlowAttention处理4个packets之间的跨包依赖关系。

**窗口注意力模式（查询窗口内的所有数据包，包括自己和相邻数据包）**：

对于第i个packet：
- **Packet 0**: 关注 Packet 0 和 Packet 1（自己 + 右邻）
- **Packet 1**: 关注 Packet 0、Packet 1 和 Packet 2（左邻 + 自己 + 右邻）
- **Packet 2**: 关注 Packet 1、Packet 2 和 Packet 3（左邻 + 自己 + 右邻）
- **Packet 3**: 关注 Packet 2 和 Packet 3（左邻 + 自己）

**处理流程**：

1. **Header Attention**：
   ```
   对每个packet的header部分应用注意力
   Query: 当前packet的header
   Key/Value: 窗口内所有packet的header（包括自己和相邻packet）
   ```

2. **Payload Attention**：
   ```
   对每个packet的payload部分应用注意力
   首先对payload进行pooling: [batch_size, 40] → [batch_size, 4]
   Query: 当前packet的pooled payload
   Key/Value: 窗口内所有packet的pooled payload（包括自己和相邻packet）
   然后unpool回原始形状: [batch_size, 4] → [batch_size, 40]
   ```

3. **残差连接和归一化**：
   ```
   header_final = LayerNorm(header_input + header_attention_output)
   payload_final = LayerNorm(payload_input + payload_attention_output)
   ```

#### 4.3 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_layers | 8 | 堆栈层数 |
| hidden_size | 768 | 隐层维度 |
| num_attention_heads | 8 | 多头注意力头数 |
| header_size | 40 | Header token数 |
| payload_size | 40 | Payload token数 |
| dropout | 0.1 | Dropout比例 |

---

### 5. Task Heads

#### 5.1 NSP_Indicating MLM Head

```
输入：[batch_size, 330, hidden_size]
  ↓
[Linear Layer: hidden_size → vocab_size]
  ↓
输出：[batch_size, 330, vocab_size]
```

**用途**：预测被mask的tokens（Packet Masking策略）

#### 5.2 MFR-Masking MLM Head

```
输入：[batch_size, 330, hidden_size]
  ↓
[Linear Layer: hidden_size → vocab_size]
  ↓
输出：[batch_size, 330, vocab_size]
```

**用途**：预测被mask的tokens（MFR Masking策略）

---

## 前向传播详细流程

### 输入数据格式

```
src: [batch_size, 330]
  ├─ Token 0: [CLS]
  ├─ Tokens 1-80: Packet 1 (80 tokens)
  ├─ Tokens 81-160: Packet 2 (80 tokens)
  ├─ Tokens 161-240: Packet 3 (80 tokens)
  ├─ Tokens 241-320: Packet 4 (80 tokens)
  ├─ Token 321: [SEP]
  └─ Tokens 322-329: Padding

seg: [batch_size, 330] (全为0)
```

### 前向传播步骤

**Step 1: Embedding**
```python
emb = embedding(src, seg)  # [batch_size, 330, 768]
```

**Step 2: 提取4个packet embeddings**
```python
packet_embeddings = []
for i in range(4):
    start_idx = 1 + i * 80
    end_idx = start_idx + 80
    packet_emb = emb[:, start_idx:end_idx, :]  # [batch_size, 80, 768]
    packet_embeddings.append(packet_emb)
```

**Step 3: 4层PacketAttention处理**
```python
packet_outputs = []
for packet_emb in packet_embeddings:
    packet_out = packet_attention_stack(packet_emb)  # [batch_size, 80, 768]
    packet_outputs.append(packet_out)
```

**Step 4: 池化**
```python
pooled_packets = pooling_layer(packet_outputs)  # list of 4 tensors
```

**Step 5: 8层FlowAttention处理**
```python
flow_headers, flow_payloads = flow_attention_stack(pooled_packets)
```

**Step 6: 拼接输出**
```python
cls_token = emb[:, 0:1, :]  # [batch_size, 1, 768]
sep_token = emb[:, 321:322, :]  # [batch_size, 1, 768]

all_packets = []
for header, payload in zip(flow_headers, flow_payloads):
    packet = torch.cat([header, payload], dim=1)  # [batch_size, 80, 768]
    all_packets.append(packet)

packets_concat = torch.cat(all_packets, dim=1)  # [batch_size, 320, 768]
output = torch.cat([cls_token, packets_concat, sep_token], dim=1)  # [batch_size, 322, 768]

# 填充到原始长度
if output.size(1) < 330:
    padding = torch.zeros(batch_size, 330 - output.size(1), 768)
    output = torch.cat([output, padding], dim=1)
```

**Step 7: Task Head预测**
```python
# NSP_Indicating
indicating_logits = indicating_mlm_output(output)  # [batch_size, 330, vocab_size]

# MFR-Masking
mfr_logits = mfr_mlm_output(output)  # [batch_size, 330, vocab_size]
```

---

## 模型参数统计

### 参数量计算

假设 `hidden_size=768, num_heads=8, vocab_size=30522`

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Embedding | ~23.5M | Token embedding + Position embedding |
| PacketAttentionStack (12层) | ~56.4M | 4组 × 3层/组 × (Q/K/V + output projection) |
|   - HeaderAttentionLayer (4层) | ~18.8M | 4 × (Q/K/V + output projection) for header |
|   - CrossAttentionLayer (4层) | ~18.8M | 4 × (Q/K/V + output projection) for cross-attention |
|   - PayloadAttentionLayer (4层) | ~18.8M | 4 × (Q/K/V + output projection) for payload |
| PoolingLayer | 0 | 无参数 |
| FlowAttention (8层) | ~75.2M | 8 × (Q/K/V + output projection) × 2 (header, payload) |
| Task Heads | ~23.6M | 2 × (768 → 30522) |
| **总计** | **~178M** | 相比原来的BERT-base (110M) 增加约62% |

---

## 与原架构的对比

### 原架构
```
Embedding → 12层Encoder (标准Transformer) → Task Heads
```

**特点**：
- 使用标准的自注意力机制
- 对所有tokens平等对待
- 没有考虑网络流量的特殊结构

### 新架构
```
Embedding → 12层PacketAttention (4组串行3层) → Pooling → 8层FlowAttention → Task Heads
```

**特点**：
- **包级注意力（12层）**：4组串行3层结构
  - 每组串行执行：h-h → h-p → p-p
  - 逐步建模header和payload的关系
  - 信息流动清晰：header先自注意力 → payload从header获取信息 → payload自注意力
- **流级注意力（8层）**：处理4个数据包之间的跨包依赖
- **显式建模**：header-header、header-payload、payload-payload三种关系
- **串行处理**：每层的输出作为下一层的输入，形成深层特征提取

**优势**：
1. **更深的包级建模**：12层包级注意力（vs 原来的4层并行）
2. **串行信息流动**：h-h → h-p → p-p，逐步建模关系
3. **更好的特征提取**：每组3层串行处理，重复4次
4. **结构感知**：显式建模packet和flow的层级结构
5. **计算高效**：
   - PayloadAttention中的窗口注意力减少计算量
   - Pooling层进一步减少FlowAttention的计算
6. **可解释性**：每个组件的功能清晰，信息流动路径明确
7. **灵活性**：可以独立调整packet级和flow级的处理

---

## 训练配置

### 数据共享策略

每个batch包含 `batch_size` 个流样本：

1. **NSP_Indicating任务**：
   - 每个流生成1个样本
   - 总计：`batch_size` 个样本

2. **MFR-Masking任务**：
   - 每个流的4个packet各生成1个样本
   - 总计：`batch_size × 4` 个样本

3. **总样本数**：`batch_size × 5`

### 任务权重

```python
nsp_indicating_weight = 0.5  # 50%
mfr_masking_weight = 0.5     # 50%
```

### 损失函数

```python
loss = nsp_indicating_weight * loss_nsp + mfr_masking_weight * loss_mfr
```

其中：
- `loss_nsp`: NSP_Indicating任务的MLM loss
- `loss_mfr`: MFR-Masking任务的MLM loss

---

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_size | 768 | 隐层维度 |
| num_attention_heads | 8 | 多头注意力头数 |
| tokens_per_packet | 80 | 每个packet的token数 |
| header_size | 40 | Header的token数 |
| payload_size | 40 | Payload的token数 |
| window_size | 20 | PacketAttention中的窗口大小 |
| dropout | 0.1 | Dropout比例 |
| seq_length | 330 | 序列长度 |
| batch_size | 8 | 批大小 |
| learning_rate | 1e-4 | 学习率 |
| warmup | 0.1 | 预热比例 |
| total_steps | 100000 | 总训练步数 |

---

## 实现细节

### 关键类

1. **HeaderAttentionLayer**: 第1层包级注意力 - h-h关系（header内部全局注意力）
2. **CrossAttentionLayer**: 第2层包级注意力 - h-p关系（payload从header获取信息）
3. **PayloadAttentionLayer**: 第3层包级注意力 - p-p关系（payload内部窗口注意力）
4. **PacketAttentionStack**: 包级注意力堆栈（4组串行3层结构，共12层）
5. **PoolingLayer**: 池化层
6. **FlowAttention**: 单层流级注意力
7. **FlowAttentionStack**: 8层流级注意力堆栈
8. **BaseModel**: 整合所有组件的基础模型
9. **MultiTaskModel**: 多任务学习模型

### 内存优化

- **梯度累积**：支持梯度累积以处理更大的有效batch size
- **子batch处理**：在前向传播中将大batch分成子batch处理，减少显存占用
- **缓冲式数据加载**：使用缓冲区流式加载数据，避免一次性加载全部数据

---

## 总结

新架构通过显式建模网络流量的包级和流级结构，提供了一个更加专业化和高效的预训练框架。相比标准的Transformer，它在保持参数量相近的情况下，能够更好地捕捉网络流量数据的特殊特征。



**现有架构细化**

当前目录的预训练主流程是：

`two-task_with_two-attention_pretrain.py` 负责解析参数和启动训练，默认 `--seq_length=330`，但运行脚本实际传入 `SEQ_LENGTH=320`。所以文档里的 330 和实际训练脚本里的 320 已经不一致。

核心实现都在 [multi_task_trainer.py](/home/user/paper/TrafficAttention/全TLA_pre-training_two-tasks_with_two-attention/multi_task_trainer.py:768)：

1. 数据读取  
   `MultiTaskDataset` 从 pickle `.pt` 文件流式读取样本。它支持两类输入：
   - 已经是流矩阵的 list/tuple；
   - 原始包格式 `(flow_id, packet_idx, packet_tokens, dataset_name)`。

2. 数据组织  
   当前代码假设原始流数据是 `5 个包 × 每包 160 tokens = 800 tokens`，然后用：
   - 第 1 包：`raw_flow[0:80]`
   - 第 2 包：`raw_flow[160:240]`
   - 第 3 包：`raw_flow[320:400]`
   - 第 4 包：`raw_flow[480:560]`

   这在 [multi_task_trainer.py](/home/user/paper/TrafficAttention/全TLA_pre-training_two-tasks_with_two-attention/multi_task_trainer.py:969) 附近实现。

3. 但真实 MFR 预处理不是这种布局  
   预处理函数 [data_process.py](/home/user/paper/TrafficAttention/data_process/MFR%20Generating/for_pre-training/data_process.py:163) 生成的是 `8 行 × 100 tokens` 的流矩阵：
   - 前 2 行：5 个包的 header 横向拼接；
   - 后 6 行：5 个包的 payload 横向拼接；
   - 每行中每个包占 20 tokens。

   因此一个完整包的 80 tokens 应该从 8 行里按同一个包列块抽取：`2 行 header × 20 = 40`，再取 payload 的前 `2 行 × 20 = 40`。当前按连续 160 取前 80，确实会导致头部与载荷、包与包混杂。

4. 样本构造  
   NSP_Indicating 任务：每条流生成 1 个样本，当前只构造 `[CLS] + 320 tokens`，没有追加 `[SEP]`，见 [multi_task_trainer.py](/home/user/paper/TrafficAttention/全TLA_pre-training_two-tasks_with_two-attention/multi_task_trainer.py:1318)。  
   MFR-Masking 任务：每条流的 4 个包分别生成样本，即每流 4 个 MFR 样本，格式是 `[CLS] + packet(80) + [SEP] + padding`，见 [multi_task_trainer.py](/home/user/paper/TrafficAttention/全TLA_pre-training_two-tasks_with_two-attention/multi_task_trainer.py:1365)。

5. 模型编码  
   `BaseModel` 做：
   - embedding；
   - 从位置 1 开始，每 80 token 切一个包，固定切 4 个包；
   - 每个包进入 `PacketAttentionStack`；
   - payload pooling；
   - 进入 8 层 `FlowAttentionStack`；
   - 拼回 `[CLS] + 4*packet + [SEP] + padding`。

   关键硬编码：
   - `for i in range(4)`：固定 4 包；
   - `start_idx = 1 + i * 80`：默认包之间没有 `[SEP]`；
   - `sep_token = emb[:, 321:322, :]`：硬编码 SEP 在 321，但实际运行 `seq_length=320` 时这个 slice 为空；
   - `FlowAttention.forward()` 也固定按 4 包写窗口逻辑。

6. 两级注意力  
   包级注意力对每个 80-token 包分为：
   - header：前 40 token；
   - payload：后 40 token；
   - 4 组串行 `h-h -> h-p -> p-p`，共 12 个子层。

   流级注意力对 4 个包做滑动窗口跨包建模：
   - 包 1 看包 1/2；
   - 包 2 看包 1/2/3；
   - 包 3 看包 2/3/4；
   - 包 4 看包 3/4。

3个任务：
1. **修复 `.pt` 数据生成阶段的数据包规整错误**  
   在 `prepare_dataset_pretrain.py` 和 `NSP_Indicating_Fine-Tuning_Dataset_Generating.p`'中，将原始 `8×100 tokens` 的 flow MFR 矩阵正确重排为独立数据包序列。每个数据包应为 `80 tokens = Header 40 + Payload 40`，避免当前 320 序列中 header/payload 混杂、包边界错乱的问题。
2. **统一预训练输入格式与微调流程的包边界协议**  
   明确并实现预训练和微调输入格式，例如使用显式分隔：

   ```text
   [CLS] + packet1(80) + [SEP] + packet2(80) + ... + packetN(80) + [SEP] + padding
   ```

   同步修改数据加载、masking、标签对齐、模型切片逻辑，确保 token、label、segment 与包边界一致。

   注意，预训练和微调的架构都要修改，避免流内数据包或数据包内序列内头部与载荷内容混杂、**排布错乱**的问题
3. **将模型架构从 4 个数据包扩展为 5 个数据包**  
   将 Packet/Flow 相关逻辑从固定 4 包改为支持 5 包：

   - 输入 `5 × 80 tokens`
   - `seq_length = 410`
   - FlowAttention 滑动窗口支持 5 个 packet
   - NSP_Indicating 和 MFR-Masking 都按 5 包生成样本
   - 清理所有硬编码的 `4`、`320`、`321` 等位置常量。
   - 注意，预训练和微调的架构都要修改


**我建议先完成任务 1，再做任务 2。**

原因很明确：任务 1 是数据语义正确性问题，任务 2 是规模扩展问题。当前 320 的错乱根因高度疑似在数据重排逻辑，不先修正，直接扩成 5 包、410 tokens，只会把错误从 `4×80` 扩大到 `5×80`。

建议顺序：

1. 先把 8×100 的 MFR 流矩阵规整转换为真正的 `packet1(80) + packet2(80) + ...`，并加断言/单测验证每包 `header40 + payload40`。
2. 再统一预训练和微调的输入协议，尤其是包间 `[SEP]` 是否保留。你提出的 5 包 410 tokens 更像是 `[CLS] + 5 * (80 + ?)` 加 padding/特殊位的设计，需要先明确边界协议。
3. 最后把模型里的 `4`、`320`、`321`、窗口逻辑改成可配置 `num_packets=5`，并把 `seq_length` 改到 410。