# 两级注意力模型架构文档

## 概述

本模型采用**两级注意力机制**处理网络流量分类任务，包括**包级注意力**和**流级注意力**两个层次。

### 核心特点
- **输入格式**: 4个数据包的串联 `[CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80)`
- **总序列长度**: 324 tokens（截断到320）
- **架构深度**: 4层包级注意力 + 池化层 + 8层流级注意力 = 12层编码器
- **架构对齐**: 与预训练模型完全一致（2025年4月修复）

---

## 第一级：包级注意力（Packet-Level Attention）

### 目的
对单个数据包内部的header和payload之间的关系进行建模。

### 配置参数
- **层数**: 4层（每个数据包独立处理）
- **数据包大小**: 80 tokens
  - Header: 40 tokens
  - Payload: 40 tokens
- **注意力头数**: 8
- **窗口大小**: 20（用于payload窗口注意力）

### 架构改进（2025年4月）

#### 原始设计（已弃用）
- 单一 `PacketAttention` 类
- 参数命名: `sub1_query`, `sub1_key`, `sub2_query` 等
- 问题: 与预训练模型参数名称不匹配

#### 改进设计（当前）
采用 **3个独立层类 + PacketAttentionStack** 的结构：

```
PacketAttentionStack (4层)
  ├── Layer Group 0
  │   ├── HeaderAttentionLayer (h-h)
  │   ├── CrossAttentionLayer (h-p)
  │   └── PayloadAttentionLayer (p-p)
  ├── Layer Group 1
  │   ├── HeaderAttentionLayer (h-h)
  │   ├── CrossAttentionLayer (h-p)
  │   └── PayloadAttentionLayer (p-p)
  ├── Layer Group 2
  │   ├── HeaderAttentionLayer (h-h)
  │   ├── CrossAttentionLayer (h-p)
  │   └── PayloadAttentionLayer (p-p)
  └── Layer Group 3
      ├── HeaderAttentionLayer (h-h)
      ├── CrossAttentionLayer (h-p)
      └── PayloadAttentionLayer (p-p)
```

**优势**:
- ✓ 参数名称与预训练模型完全一致
- ✓ 可以直接加载预训练权重
- ✓ 代码结构更清晰，易于维护

### 3个子注意力机制（串行处理）

#### SubAttention1: Header内部全局注意力 (h-h)
```
类: HeaderAttentionLayer
参数: query, key, value, output, norm, dropout

输入: header_hidden [batch_size, 40, hidden_size]
处理: 
  - Query, Key, Value 都来自header
  - 全局注意力（无mask）
  - 每个header token可以attend到所有其他header tokens
输出: header_output [batch_size, 40, hidden_size]
      + 残差连接 + LayerNorm
```

**作用**: 捕捉header内部的全局依赖关系

---

#### SubAttention2: Header-Payload交叉注意力 (h-p)
```
类: CrossAttentionLayer
参数: query, key, value, output, norm, dropout

输入: 
  - payload_hidden [batch_size, 40, hidden_size] (原始payload)
  - header_output [batch_size, 40, hidden_size] (来自SubAttention1)
处理:
  - Query 来自 payload_hidden
  - Key, Value 来自 header_output
  - 交叉注意力（无mask）
  - 每个payload token可以attend到所有header tokens
输出: payload_after_sub2 [batch_size, 40, hidden_size]
      + 残差连接 + LayerNorm
```

**作用**: 让payload学习header中的信息，建立header-payload之间的关系

---

#### SubAttention3: Payload内部窗口注意力 (p-p)
```
类: PayloadAttentionLayer
参数: query, key, value, output, norm, dropout, window_size

输入: payload_after_sub2 [batch_size, 40, hidden_size] (来自SubAttention2)
处理:
  - Query, Key, Value 都来自 payload_after_sub2
  - 窗口注意力（有mask）
  - 每个payload token只能attend到窗口内的tokens
  - 窗口大小: 20 (即 ±10 范围内)
输出: payload_final [batch_size, 40, hidden_size]
      + 残差连接 + LayerNorm
```

**作用**: 捕捉payload内部的局部依赖关系，减少计算复杂度

---

### 串行处理流程图

```
packet_hidden [batch_size, 80, hidden_size]
    |
    +-- 分离 --> header [40] + payload [40]
    |
    v
HeaderAttentionLayer (h-h)
    |
    +-- header_output [40]
    |
    v
CrossAttentionLayer (h-p)
    |
    +-- payload_after_sub2 [40]
    |
    v
PayloadAttentionLayer (p-p)
    |
    +-- payload_final [40]
    |
    v
拼接 --> output [batch_size, 80, hidden_size]
```

---

## 池化层（Pooling Layer）

### 目的
减少序列长度，为流级注意力做准备。

### 处理方式
- **方法**: Mean pooling
- **输入**: 每个数据包的payload (40 tokens)
- **处理**: 将40个tokens分成4组，每组10个tokens，取平均
- **输出**: 4个tokens
- **结果**: 每个数据包从80 tokens → 80 tokens（header 40 + pooled payload 4×10）

---

## 第二级：流级注意力（Flow-Level Attention）

### 目的
对4个数据包之间的跨包依赖关系进行建模。

### 配置参数
- **层数**: 8层
- **处理对象**: 4个数据包
- **注意力头数**: 8

### Dropout应用修复（2025年4月）

#### 修改前（错误）
```python
attention_probs = F.softmax(attention_scores, dim=-1)
context = torch.matmul(self.dropout(attention_probs), values)
```

#### 修改后（正确）
```python
attention_probs = F.softmax(attention_scores, dim=-1)
attention_probs = self.dropout(attention_probs)
context = torch.matmul(attention_probs, values)
```

**原因**: Dropout应该在softmax之后、matmul之前应用，而不是在matmul中应用。这与预训练模型的实现一致。

### 处理流程

#### 1. 数据重组
```
输入: 4个数据包的包级注意力输出（每个80 tokens）
处理:
  - 提取每个数据包的header (40 tokens)
  - 提取每个数据包的payload (40 tokens)
  - 对payload进行mean pooling (40 → 4 tokens)
  - 恢复到原始形状用于后续处理 (4 → 40 tokens)
输出: 4个重组后的数据包 (每个80 tokens)
```

#### 2. 滑动窗口注意力（包括自己和相邻包）
```
对4个数据包应用滑动窗口注意力（包括自己）:

包1: 窗口=[0,1]
  - 可以attend到: 包1, 包2
  - Query: 包1的header/payload
  - Key/Value: 包1和包2的header/payload

包2: 窗口=[0,1,2]
  - 可以attend到: 包1, 包2, 包3
  - Query: 包2的header/payload
  - Key/Value: 包1、包2和包3的header/payload

包3: 窗口=[1,2,3]
  - 可以attend到: 包2, 包3, 包4
  - Query: 包3的header/payload
  - Key/Value: 包2、包3和包4的header/payload

包4: 窗口=[2,3]
  - 可以attend到: 包3, 包4
  - Query: 包4的header/payload
  - Key/Value: 包3和包4的header/payload
```

#### 3. 分离处理Header和Payload
```
对每个数据包分别应用注意力:

Header注意力 (h查h):
  - Query: 当前包的header
  - Key/Value: 窗口内包的header（包括自己）
  - 结果: 更新后的header

Payload注意力 (p查p):
  - Query: 当前包的pooled payload
  - Key/Value: 窗口内包的pooled payload（包括自己）
  - 结果: 更新后的payload
```

---

## 完整架构流程

```
输入序列 [CLS] + p1(80) + [SEP] + p2(80) + [SEP] + p3(80) + [SEP] + p4(80)
    |
    v
Embedding层
    |
    v
【第1-4层：包级注意力堆栈 (PacketAttentionStack)】
    |
    +-- Layer Group 0: 3个子注意力机制串行
    |     |
    |     +-- HeaderAttentionLayer (h-h)
    |     +-- CrossAttentionLayer (h-p)
    |     +-- PayloadAttentionLayer (p-p)
    |
    +-- Layer Group 1: 3个子注意力机制串行
    +-- Layer Group 2: 3个子注意力机制串行
    +-- Layer Group 3: 3个子注意力机制串行
    |
    v
【池化层】
    |
    +-- Mean pooling: payload (40 → 4 tokens)
    |
    v
【第5-12层：流级注意力堆栈（包括自己和相邻包）】
    |
    +-- Layer 0: FlowAttention
    |     |
    |     +-- 包1: attend到[包1, 包2]
    |     +-- 包2: attend到[包1, 包2, 包3]
    |     +-- 包3: attend到[包2, 包3, 包4]
    |     +-- 包4: attend到[包3, 包4]
    |     |
    |     +-- h查h + p查p
    |
    +-- Layer 1-7: FlowAttention (同上)
    |
    v
【分类头】
    |
    +-- 池化 (mean/max/first/last)
    +-- 线性层1 (hidden_size → hidden_size) + tanh
    +-- 线性层2 (hidden_size → labels_num)
    |
    v
输出: logits [batch_size, labels_num]
```

---

## 流级注意力的改进：包括自己的滑动窗口

### 原始设计 vs 改进设计

#### 原始设计（仅相邻包）
```
包1: attend到 [包2]
包2: attend到 [包1, 包3]
包3: attend到 [包2, 包4]
包4: attend到 [包3]

问题:
- 包1和包4无法自注意力，丢失自身特征
- 信息流动不对称
```

#### 改进设计（包括自己）
```
包1: attend到 [包1, 包2]
包2: attend到 [包1, 包2, 包3]
包3: attend到 [包2, 包3, 包4]
包4: attend到 [包3, 包4]

优势:
- 每个包都可以自注意力，保留自身特征
- 信息流动对称且完整
- 更好地融合自身和相邻包的信息
```

### 注意力权重分布

在改进设计中，每个包的注意力权重分布如下：

```
包1的注意力权重:
  - 自注意力 (包1): α₁
  - 相邻注意力 (包2): α₂
  - 总和: α₁ + α₂ = 1.0

包2的注意力权重:
  - 前向注意力 (包1): β₁
  - 自注意力 (包2): β₂
  - 后向注意力 (包3): β₃
  - 总和: β₁ + β₂ + β₃ = 1.0

包3的注意力权重:
  - 前向注意力 (包2): γ₁
  - 自注意力 (包3): γ₂
  - 后向注意力 (包4): γ₃
  - 总和: γ₁ + γ₂ + γ₃ = 1.0

包4的注意力权重:
  - 相邻注意力 (包3): δ₁
  - 自注意力 (包4): δ₂
  - 总和: δ₁ + δ₂ = 1.0
```

### 信息流动图

```
Layer 5 (FlowAttention):
  包1 ←→ 包2
  包1 ←→ 包1 (自注意力)
  
  包1 ←→ 包2 ←→ 包3
  包2 ←→ 包2 (自注意力)
  
  包2 ←→ 包3 ←→ 包4
  包3 ←→ 包3 (自注意力)
  
  包3 ←→ 包4
  包4 ←→ 包4 (自注意力)

Layer 6-12 (FlowAttention):
  重复上述信息流动，逐层加强特征表示
```

### 计算复杂度分析

```
原始设计:
  - 包1: 1个K/V源 (包2)
  - 包2: 2个K/V源 (包1, 包3)
  - 包3: 2个K/V源 (包2, 包4)
  - 包4: 1个K/V源 (包3)
  - 平均: 1.5个K/V源

改进设计:
  - 包1: 2个K/V源 (包1, 包2)
  - 包2: 3个K/V源 (包1, 包2, 包3)
  - 包3: 3个K/V源 (包2, 包3, 包4)
  - 包4: 2个K/V源 (包3, 包4)
  - 平均: 2.5个K/V源

增加的计算量: ~67%（可接受的权衡）
```

---

## 参数结构

### 包级注意力参数

#### HeaderAttentionLayer
```
参数名称:
  - query: Linear(hidden_size, hidden_size)
  - key: Linear(hidden_size, hidden_size)
  - value: Linear(hidden_size, hidden_size)
  - output: Linear(hidden_size, hidden_size)
  - norm: LayerNorm(hidden_size)
  - dropout: Dropout(dropout_rate)

参数路径示例:
  packet_attention_stack.layer_groups.0.0.query.weight
  packet_attention_stack.layer_groups.0.0.query.bias
  packet_attention_stack.layer_groups.0.0.key.weight
  ...
```

#### CrossAttentionLayer
```
参数名称: 同HeaderAttentionLayer
参数路径示例:
  packet_attention_stack.layer_groups.0.1.query.weight
  packet_attention_stack.layer_groups.0.1.query.bias
  ...
```

#### PayloadAttentionLayer
```
参数名称: 同HeaderAttentionLayer + window_size
参数路径示例:
  packet_attention_stack.layer_groups.0.2.query.weight
  packet_attention_stack.layer_groups.0.2.query.bias
  ...
```

#### PacketAttentionStack总参数
```
4层PacketAttentionStack:
  - 每层: 3个子层 × 4个线性层 + 3个LayerNorm = 12个线性层 + 3个LayerNorm
  - 总计: 4 × 12 × (hidden_size × hidden_size) = 48 × hidden_size²
  - LayerNorm: 4 × 3 × hidden_size = 12 × hidden_size
```

### 流级注意力参数

#### FlowAttention
```
参数名称:
  - header_query, header_key, header_value, header_output
  - payload_query, payload_key, payload_value, payload_output
  - header_norm, payload_norm
  - dropout

参数路径示例:
  flow_attention_stack.0.header_query.weight
  flow_attention_stack.0.header_query.bias
  flow_attention_stack.0.payload_query.weight
  ...
```

#### FlowAttentionStack总参数
```
8层FlowAttention:
  - 每层: 2个注意力 × 4个线性层 + 2个LayerNorm = 8个线性层 + 2个LayerNorm
  - 总计: 8 × 8 × (hidden_size × hidden_size) = 64 × hidden_size²
  - LayerNorm: 8 × 2 × hidden_size = 16 × hidden_size
```

---

## 关键设计决策

### 1. 为什么采用串行处理？
- **信息流动**: 每个子注意力机制的输出作为下一个的输入，形成信息的逐步精化
- **依赖建模**: SubAttention2依赖SubAttention1的结果，SubAttention3依赖SubAttention2的结果
- **计算效率**: 相比并行处理，串行处理减少了参数数量

### 2. 为什么分离Header和Payload？
- **结构差异**: Header和Payload在网络数据包中有不同的语义
- **处理方式**: Header使用全局注意力，Payload使用窗口注意力
- **计算复杂度**: 减少了计算量

### 3. 为什么流级注意力包括自己？
- **自注意力**: 每个包可以attend到自己，强化自身特征
- **上下文融合**: 结合自身特征和相邻包特征，形成更丰富的表示
- **信息完整性**: 避免丢失自身重要信息

### 4. 为什么使用滑动窗口注意力？
- **局部性**: 网络流量中相邻数据包更相关
- **计算效率**: 减少了注意力计算的复杂度
- **可解释性**: 更容易理解模型的决策

### 5. 为什么进行池化？
- **维度降低**: 从40个tokens池化到4个tokens，减少流级注意力的计算量
- **信息保留**: Mean pooling保留了payload的平均信息
- **层级过渡**: 为包级到流级的过渡做准备

### 6. 为什么采用3层独立类而不是单一类？
- **参数对齐**: 与预训练模型参数名称完全一致，支持权重转移
- **代码清晰**: 每个层的职责明确，易于理解和维护
- **可扩展性**: 便于后续修改或扩展单个层的功能

---

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 768 | 隐层维度 |
| `num_attention_heads` | 8 | 注意力头数 |
| `packet_size` | 80 | 每个数据包的token数 |
| `header_size` | 40 | Header的token数 |
| `payload_size` | 40 | Payload的token数 |
| `window_size` | 20 | Payload窗口注意力的窗口大小 |
| `dropout` | 0.1 | Dropout比率 |
| `seq_length` | 320 | 输入序列长度 |

---

## 输入输出格式

### 输入
```
src: [batch_size, 320]
  - 格式: [CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80)
  - 总长度: 1 + 80 + 1 + 80 + 1 + 80 + 1 + 80 = 324 (截断到320)

seg: [batch_size, 320]
  - 所有位置都标记为1（表示同一个segment）

tgt: [batch_size]
  - 流量类别标签 (0-K)
```

### 输出
```
logits: [batch_size, labels_num]
  - 分类logits

attention_probs: dict
  - encoder: 流级注意力的概率
  - packet: 包级注意力的概率 (sub1, sub2, sub3)
  - flow: 流级注意力的详细概率 (header, payload)
```

---

## 训练配置

### 损失函数
- **标准交叉熵**: `NLLLoss(LogSoftmax(logits), target)`
- **知识蒸馏** (可选): `α × MSELoss(logits, soft_target) + (1-α) × NLLLoss(...)`

### 优化器
- **AdamW** (推荐)
- **学习率**: 2e-5 (微调)
- **权重衰减**: 0.01

### 学习率调度
- **Warmup**: 前10%的步数进行线性warmup
- **衰减**: 之后线性衰减到0

### 早停
- **Patience**: 3个epoch
- **Min Delta**: 0.0 (任何改进都计数)

---

## 模型变体

### 可配置选项
```python
--use_packet_attention      # 启用包级注意力 (默认: True)
--use_flow_attention        # 启用流级注意力 (默认: True)
--window_size              # 窗口大小 (默认: 20)
--num_attention_heads      # 注意力头数 (默认: 8)
--pooling                  # 池化方式: mean/max/first/last (默认: first)
```

---

## 性能指标

### 评估指标
- **Accuracy**: 整体准确率
- **Macro-Precision**: 宏平均精确率
- **Macro-Recall**: 宏平均召回率
- **Macro-F1**: 宏平均F1分数
- **Confusion Matrix**: 混淆矩阵

### Attention可视化
- 保存包级注意力权重 (h-h, h-p, p-p)
- 保存流级注意力权重 (header, payload)
- 支持BertViz格式

---

## 架构修复历史

### 2025年4月修复
1. **PacketAttention重构**
   - 从单一类改为3个独立层类 + PacketAttentionStack
   - 参数名称从 `sub1_query` 改为 `query`（在各层类中）
   - 参数路径从 `packet_attention_stack.0.sub1_query` 改为 `packet_attention_stack.layer_groups.0.0.query`

2. **Dropout应用顺序修复**
   - 修复FlowAttention中的dropout应用顺序
   - 从 `matmul(dropout(probs), values)` 改为 `matmul(dropout(probs), values)`
   - 确保与预训练模型一致

3. **参数对齐**
   - 微调模型参数名称与预训练模型完全一致
   - 支持直接加载预训练权重（`strict=False`）

---

## 参考文献

- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- 本模型改进: 针对网络流量分类的两级注意力机制
- 架构对齐: 2025年4月与预训练模型完全对齐

---

## 第一级：包级注意力（Packet-Level Attention）

### 目的
对单个数据包内部的header和payload之间的关系进行建模。

### 配置参数
- **层数**: 4层（每个数据包独立处理）
- **数据包大小**: 80 tokens
  - Header: 40 tokens
  - Payload: 40 tokens
- **注意力头数**: 8
- **窗口大小**: 20（用于payload窗口注意力）

### 3个子注意力机制（串行处理）

#### SubAttention1: Header内部全局注意力 (h-h)
```
输入: header_hidden [batch_size, 40, hidden_size]
处理: 
  - Query, Key, Value 都来自header
  - 全局注意力（无mask）
  - 每个header token可以attend到所有其他header tokens
输出: header_output [batch_size, 40, hidden_size]
      + 残差连接 + LayerNorm
```

**作用**: 捕捉header内部的全局依赖关系

---

#### SubAttention2: Header-Payload交叉注意力 (h-p)
```
输入: 
  - payload_hidden [batch_size, 40, hidden_size] (原始payload)
  - header_output [batch_size, 40, hidden_size] (来自SubAttention1)
处理:
  - Query 来自 payload_hidden
  - Key, Value 来自 header_output
  - 交叉注意力（无mask）
  - 每个payload token可以attend到所有header tokens
输出: payload_after_sub2 [batch_size, 40, hidden_size]
      + 残差连接 + LayerNorm
```

**作用**: 让payload学习header中的信息，建立header-payload之间的关系

---

#### SubAttention3: Payload内部窗口注意力 (p-p)
```
输入: payload_after_sub2 [batch_size, 40, hidden_size] (来自SubAttention2)
处理:
  - Query, Key, Value 都来自 payload_after_sub2
  - 窗口注意力（有mask）
  - 每个payload token只能attend到窗口内的tokens
  - 窗口大小: 20 (即 ±10 范围内)
输出: payload_final [batch_size, 40, hidden_size]
      + 残差连接 + LayerNorm
```

**作用**: 捕捉payload内部的局部依赖关系，减少计算复杂度

---

### 串行处理流程图

```
packet_hidden [batch_size, 80, hidden_size]
    |
    +-- 分离 --> header [40] + payload [40]
    |
    v
SubAttention1 (h-h)
    |
    +-- header_output [40]
    |
    v
SubAttention2 (h-p)
    |
    +-- payload_after_sub2 [40]
    |
    v
SubAttention3 (p-p)
    |
    +-- payload_final [40]
    |
    v
拼接 --> output [batch_size, 80, hidden_size]
```

---

## 池化层（Pooling Layer）

### 目的
减少序列长度，为流级注意力做准备。

### 处理方式
- **方法**: Mean pooling
- **输入**: 每个数据包的payload (40 tokens)
- **处理**: 将40个tokens分成4组，每组10个tokens，取平均
- **输出**: 4个tokens
- **结果**: 每个数据包从80 tokens → 80 tokens（header 40 + pooled payload 4×10）

---

## 第二级：流级注意力（Flow-Level Attention）

### 目的
对4个数据包之间的跨包依赖关系进行建模。

### 配置参数
- **层数**: 8层
- **处理对象**: 4个数据包
- **注意力头数**: 8

### 处理流程

#### 1. 数据重组
```
输入: 4个数据包的包级注意力输出（每个80 tokens）
处理:
  - 提取每个数据包的header (40 tokens)
  - 提取每个数据包的payload (40 tokens)
  - 对payload进行mean pooling (40 → 4 tokens)
  - 恢复到原始形状用于后续处理 (4 → 40 tokens)
输出: 4个重组后的数据包 (每个80 tokens)
```

#### 2. 滑动窗口注意力（包括自己和相邻包）
```
对4个数据包应用滑动窗口注意力（包括自己）:

包1: 窗口=[0,1]
  - 可以attend到: 包1, 包2
  - Query: 包1的header/payload
  - Key/Value: 包1和包2的header/payload

包2: 窗口=[0,1,2]
  - 可以attend到: 包1, 包2, 包3
  - Query: 包2的header/payload
  - Key/Value: 包1、包2和包3的header/payload

包3: 窗口=[1,2,3]
  - 可以attend到: 包2, 包3, 包4
  - Query: 包3的header/payload
  - Key/Value: 包2、包3和包4的header/payload

包4: 窗口=[2,3]
  - 可以attend到: 包3, 包4
  - Query: 包4的header/payload
  - Key/Value: 包3和包4的header/payload
```

#### 3. 分离处理Header和Payload
```
对每个数据包分别应用注意力:

Header注意力 (h查h):
  - Query: 当前包的header
  - Key/Value: 窗口内包的header（包括自己）
  - 结果: 更新后的header

Payload注意力 (p查p):
  - Query: 当前包的pooled payload
  - Key/Value: 窗口内包的pooled payload（包括自己）
  - 结果: 更新后的payload
```

---

## 完整架构流程

```
输入序列 [CLS] + p1(80) + [SEP] + p2(80) + [SEP] + p3(80) + [SEP] + p4(80)
    |
    v
Embedding层
    |
    v
【第1-4层：包级注意力堆栈】
    |
    +-- Layer1: PacketAttention (3个子注意力机制串行)
    |     |
    |     +-- SubAttention1 (h-h): header全局注意力
    |     +-- SubAttention2 (h-p): payload交叉注意力
    |     +-- SubAttention3 (p-p): payload窗口注意力
    |
    +-- Layer2: PacketAttention (3个子注意力机制串行)
    +-- Layer3: PacketAttention (3个子注意力机制串行)
    +-- Layer4: PacketAttention (3个子注意力机制串行)
    |
    v
【池化层】
    |
    +-- Mean pooling: payload (40 → 4 tokens)
    |
    v
【第5-12层：流级注意力堆栈（包括自己和相邻包）】
    |
    +-- Layer5: FlowAttention
    |     |
    |     +-- 包1: attend到[包1, 包2]
    |     +-- 包2: attend到[包1, 包2, 包3]
    |     +-- 包3: attend到[包2, 包3, 包4]
    |     +-- 包4: attend到[包3, 包4]
    |     |
    |     +-- h查h + p查p
    |
    +-- Layer6: FlowAttention (同上)
    +-- Layer7: FlowAttention (同上)
    +-- Layer8: FlowAttention (同上)
    +-- Layer9: FlowAttention (同上)
    +-- Layer10: FlowAttention (同上)
    +-- Layer11: FlowAttention (同上)
    +-- Layer12: FlowAttention (同上)
    |
    v
【分类头】
    |
    +-- 池化 (mean/max/first/last)
    +-- 线性层1 (hidden_size → hidden_size) + tanh
    +-- 线性层2 (hidden_size → labels_num)
    |
    v
输出: logits [batch_size, labels_num]
```

---

## 流级注意力的改进：包括自己的滑动窗口

### 原始设计 vs 改进设计

#### 原始设计（仅相邻包）
```
包1: attend到 [包2]
包2: attend到 [包1, 包3]
包3: attend到 [包2, 包4]
包4: attend到 [包3]

问题:
- 包1和包4无法自注意力，丢失自身特征
- 信息流动不对称
```

#### 改进设计（包括自己）
```
包1: attend到 [包1, 包2]
包2: attend到 [包1, 包2, 包3]
包3: attend到 [包2, 包3, 包4]
包4: attend到 [包3, 包4]

优势:
- 每个包都可以自注意力，保留自身特征
- 信息流动对称且完整
- 更好地融合自身和相邻包的信息
```

### 注意力权重分布

在改进设计中，每个包的注意力权重分布如下：

```
包1的注意力权重:
  - 自注意力 (包1): α₁
  - 相邻注意力 (包2): α₂
  - 总和: α₁ + α₂ = 1.0

包2的注意力权重:
  - 前向注意力 (包1): β₁
  - 自注意力 (包2): β₂
  - 后向注意力 (包3): β₃
  - 总和: β₁ + β₂ + β₃ = 1.0

包3的注意力权重:
  - 前向注意力 (包2): γ₁
  - 自注意力 (包3): γ₂
  - 后向注意力 (包4): γ₃
  - 总和: γ₁ + γ₂ + γ₃ = 1.0

包4的注意力权重:
  - 相邻注意力 (包3): δ₁
  - 自注意力 (包4): δ₂
  - 总和: δ₁ + δ₂ = 1.0
```

### 信息流动图

```
Layer 5 (FlowAttention):
  包1 ←→ 包2
  包1 ←→ 包1 (自注意力)
  
  包1 ←→ 包2 ←→ 包3
  包2 ←→ 包2 (自注意力)
  
  包2 ←→ 包3 ←→ 包4
  包3 ←→ 包3 (自注意力)
  
  包3 ←→ 包4
  包4 ←→ 包4 (自注意力)

Layer 6-12 (FlowAttention):
  重复上述信息流动，逐层加强特征表示
```

### 计算复杂度分析

```
原始设计:
  - 包1: 1个K/V源 (包2)
  - 包2: 2个K/V源 (包1, 包3)
  - 包3: 2个K/V源 (包2, 包4)
  - 包4: 1个K/V源 (包3)
  - 平均: 1.5个K/V源

改进设计:
  - 包1: 2个K/V源 (包1, 包2)
  - 包2: 3个K/V源 (包1, 包2, 包3)
  - 包3: 3个K/V源 (包2, 包3, 包4)
  - 包4: 2个K/V源 (包3, 包4)
  - 平均: 2.5个K/V源

增加的计算量: ~67%（可接受的权衡）
```

---

### 包级注意力参数
```
每层PacketAttention:
  - SubAttention1: 3个线性层 (Q, K, V) + 1个输出层 + LayerNorm
  - SubAttention2: 3个线性层 (Q, K, V) + 1个输出层 + LayerNorm
  - SubAttention3: 3个线性层 (Q, K, V) + 1个输出层 + LayerNorm
  - 总计: 12个线性层 + 3个LayerNorm

4层PacketAttention总参数:
  - 线性层: 4 × 12 × (hidden_size × hidden_size) = 48 × hidden_size²
  - LayerNorm: 4 × 3 × hidden_size = 12 × hidden_size
```

### 流级注意力参数
```
每层FlowAttention:
  - Header注意力: 3个线性层 (Q, K, V) + 1个输出层 + LayerNorm
  - Payload注意力: 3个线性层 (Q, K, V) + 1个输出层 + LayerNorm
  - 总计: 8个线性层 + 2个LayerNorm

8层FlowAttention总参数:
  - 线性层: 8 × 8 × (hidden_size × hidden_size) = 64 × hidden_size²
  - LayerNorm: 8 × 2 × hidden_size = 16 × hidden_size
```

---

## 关键设计决策

### 1. 为什么采用串行处理？
- **信息流动**: 每个子注意力机制的输出作为下一个的输入，形成信息的逐步精化
- **依赖建模**: SubAttention2依赖SubAttention1的结果，SubAttention3依赖SubAttention2的结果
- **计算效率**: 相比并行处理，串行处理减少了参数数量

### 2. 为什么分离Header和Payload？
- **结构差异**: Header和Payload在网络数据包中有不同的语义
- **处理方式**: Header使用全局注意力，Payload使用窗口注意力
- **计算复杂度**: 减少了计算量

### 3. 为什么流级注意力包括自己？
- **自注意力**: 每个包可以attend到自己，强化自身特征
- **上下文融合**: 结合自身特征和相邻包特征，形成更丰富的表示
- **信息完整性**: 避免丢失自身重要信息

### 4. 为什么使用滑动窗口注意力？
- **局部性**: 网络流量中相邻数据包更相关
- **计算效率**: 减少了注意力计算的复杂度
- **可解释性**: 更容易理解模型的决策

### 5. 为什么进行池化？
- **维度降低**: 从40个tokens池化到4个tokens，减少流级注意力的计算量
- **信息保留**: Mean pooling保留了payload的平均信息
- **层级过渡**: 为包级到流级的过渡做准备

---

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 768 | 隐层维度 |
| `num_attention_heads` | 8 | 注意力头数 |
| `packet_size` | 80 | 每个数据包的token数 |
| `header_size` | 40 | Header的token数 |
| `payload_size` | 40 | Payload的token数 |
| `window_size` | 20 | Payload窗口注意力的窗口大小 |
| `dropout` | 0.1 | Dropout比率 |
| `seq_length` | 320 | 输入序列长度 |

---

## 输入输出格式

### 输入
```
src: [batch_size, 320]
  - 格式: [CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80)
  - 总长度: 1 + 80 + 1 + 80 + 1 + 80 + 1 + 80 = 324 (截断到320)

seg: [batch_size, 320]
  - 所有位置都标记为1（表示同一个segment）

tgt: [batch_size]
  - 流量类别标签 (0-K)
```

### 输出
```
logits: [batch_size, labels_num]
  - 分类logits

attention_probs: dict
  - encoder: 流级注意力的概率
  - packet: 包级注意力的概率 (sub1, sub2, sub3)
  - flow: 流级注意力的详细概率 (header, payload)
```

---

## 训练配置

### 损失函数
- **标准交叉熵**: `NLLLoss(LogSoftmax(logits), target)`
- **知识蒸馏** (可选): `α × MSELoss(logits, soft_target) + (1-α) × NLLLoss(...)`

### 优化器
- **AdamW** (推荐)
- **学习率**: 2e-5 (微调)
- **权重衰减**: 0.01

### 学习率调度
- **Warmup**: 前10%的步数进行线性warmup
- **衰减**: 之后线性衰减到0

### 早停
- **Patience**: 3个epoch
- **Min Delta**: 0.0 (任何改进都计数)

---

## 模型变体

### 可配置选项
```python
--use_packet_attention      # 启用包级注意力 (默认: True)
--use_flow_attention        # 启用流级注意力 (默认: True)
--window_size              # 窗口大小 (默认: 20)
--num_attention_heads      # 注意力头数 (默认: 8)
--pooling                  # 池化方式: mean/max/first/last (默认: first)
```

---

## 性能指标

### 评估指标
- **Accuracy**: 整体准确率
- **Macro-Precision**: 宏平均精确率
- **Macro-Recall**: 宏平均召回率
- **Macro-F1**: 宏平均F1分数
- **Confusion Matrix**: 混淆矩阵

### Attention可视化
- 保存包级注意力权重 (h-h, h-p, p-p)
- 保存流级注意力权重 (header, payload)
- 支持BertViz格式

---

## 参考文献

- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- 本模型改进: 针对网络流量分类的两级注意力机制
