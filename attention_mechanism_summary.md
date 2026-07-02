# 注意力机制实现总结

## 一、Pre-training（预训练）

### 核心文件

| 文件 | 说明 |
|------|------|
| `pre-training/multi_task_trainer.py` | 自定义流量注意力模块（HeaderAttention、CrossAttention、PayloadAttention、FlowAttention） |
| `pre-training/uer/layers/multi_headed_attn.py` | 标准多头注意力 |
| `pre-training/uer/layers/transformer.py` | TransformerLayer / TransformerDecoderLayer |
| `pre-training/uer/layers/synthesizer.py` | Dense/Random Synthesizer 注意力 |
| `pre-training/uer/layers/relative_position_embedding.py` | T5 相对位置编码 |
| `pre-training/uer/encoders/transformer_encoder.py` | Transformer 编码器堆叠 |

### 注意力结构：两层级分层注意力

```
输入 (330 tokens: [CLS] + 4 packets × 80 tokens + [SEP])
    │
    ▼
Embedding (Word + Position + Segment)
    │
    ▼ 拆分为 4 个 packet（每个 80 tokens = 40 header + 40 payload）
    │
    ▼
╔═══════════════════════════════════════════════════════════╗
║  PacketAttentionStack × 4 层（包内注意力）                  ║
║  每层依次执行：                                             ║
║  ┌───────────────────────────────────────────────────────┐ ║
║  │ 1. HeaderAttentionLayer                               │ ║
║  │    header tokens 之间的多头自注意力                      │ ║
║  │    payload 不参与，直接传递                              │ ║
║  ├───────────────────────────────────────────────────────┤ ║
║  │ 2. CrossAttentionLayer                                │ ║
║  │    payload → header 的交叉注意力                        │ ║
║  │    Q 来自 payload, K/V 来自 header                     │ ║
║  │    让 payload 信息融合 header 的语义                    │ ║
║  ├───────────────────────────────────────────────────────┤ ║
║  │ 3. PayloadAttentionLayer                              │ ║
║  │    payload tokens 的滑动窗口自注意力                     │ ║
║  │    window_size=20，限制局部范围                          │ ║
║  │    header 不参与，直接传递                               │ ║
║  └───────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════╝
    │
    ▼
PoolingLayer: payload 40 tokens → 均值池化 → 4 个摘要向量
    │
    ▼
╔═══════════════════════════════════════════════════════════╗
║  FlowAttentionStack × 8 层（包间注意力）                   ║
║  ┌───────────────────────────────────────────────────────┐ ║
║  │ FlowAttention                                         │ ║
║  │ - 跨 4 个 packet 的滑动窗口注意力（窗口≈3）             │ ║
║  │ - header 和 payload 各有独立的 Q/K/V 投影              │ ║
║  │ - 当前 packet 的 header/payload 注意邻近 packet        │ ║
║  │ - Residual + LayerNorm                                │ ║
║  └───────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════╝
    │
    ▼
重组完整序列 → MLM 预测头（两个任务头）
```

### 各模块功能详解

#### 1. PacketAttentionStack（包内注意力，4层）

每一层包含三个子模块，按顺序处理单个 packet 的 80 个 tokens：

- **HeaderAttentionLayer**：对 header 部分（前 40 tokens）执行标准多头自注意力，捕获协议字段之间的关系。计算 Q/K/V → scaled dot-product attention → 线性投影 → Residual + LayerNorm。Payload tokens 直接跳过不参与计算。

- **CrossAttentionLayer**：payload 到 header 的交叉注意力。Query 来自 payload tokens，Key/Value 来自 header tokens，使 payload 能够感知 header 中的协议结构信息。对 payload 做 Residual + LayerNorm，header 直接传递。

- **PayloadAttentionLayer**：对 payload 部分（后 40 tokens）执行滑动窗口自注意力（window_size=20），每个 token 只能注意到前后各 10 个位置的 token。捕获载荷中的局部模式，避免全局注意力在长 payload 上的计算开销。Header 直接传递。

#### 2. PoolingLayer（维度压缩）

将每个 packet 的 40 个 payload tokens 按每 10 个一组做均值池化，压缩为 4 个摘要向量，降低后续包间注意力的计算量。

#### 3. FlowAttentionStack（包间注意力，8层）

每一层包含一个 FlowAttention 模块，跨 4 个 packet 工作：

- **FlowAttention**：采用滑动窗口机制（窗口大小≈3），当前 packet 注意相邻的 packets。Header 和 payload 分别有独立的 Q/K/V/Output 线性投影。Payload 在注意力前做池化、注意力后做反池化以匹配维度。每个流都有 Residual + LayerNorm。

### 基础框架模块（UER 框架）

| 模块 | 文件 | 功能 |
|------|------|------|
| MultiHeadedAttention | `uer/layers/multi_headed_attn.py` | 标准 scaled dot-product 多头注意力，支持相对位置偏置 |
| TransformerLayer | `uer/layers/transformer.py` | 自注意力 + FFN + Dropout + LayerNorm（支持 pre/post norm） |
| TransformerDecoderLayer | `uer/layers/transformer.py` | 自注意力 + 交叉注意力 + FFN，用于 seq2seq 解码 |
| DenseAttention / DenseSynthesizer | `uer/layers/synthesizer.py` | 用学习的稠密映射 F(X)=Linear₂(ReLU(Linear₁(X))) 替代 Q·K 注意力 |
| RandomAttention / RandomSynthesizer | `uer/layers/synthesizer.py` | 用固定可学习随机矩阵 R 作为注意力权重 |
| RelativePositionEmbedding | `uer/layers/relative_position_embedding.py` | T5 风格的分桶相对位置偏置，加到注意力分数上 |

---

## 二、Fine-tuning（微调）

### 核心文件

| 文件 | 说明 |
|------|------|
| `fine-tuning/Two-task_fine-tuning_two-attention.py` | 自定义流量注意力模块，封装为 TwoLevelAttentionEncoder |
| `fine-tuning/uer/layers/multi_headed_attn.py` | 标准多头注意力 |
| `fine-tuning/uer/layers/transformer.py` | Transformer 层 |
| `fine-tuning/uer/layers/synthesizer.py` | Synthesizer 变体 |
| `fine-tuning/uer/layers/relative_position_embedding.py` | 相对位置编码 |

### 注意力结构：TwoLevelAttentionEncoder

与预训练相同的两层级架构，封装为统一的编码器模块：

```
输入序列 (330 tokens)
    │
    ▼
提取 4 个 packet（位置 1-80, 82-161, 163-242, 244-323）
    │
    ▼
PacketAttentionStack × 4 层（包内注意力）
  ├── HeaderAttentionLayer: header 全局自注意力
  ├── CrossAttentionLayer: payload→header 交叉注意力
  └── PayloadAttentionLayer: payload 滑动窗口自注意力
    │
    ▼
FlowAttention × 8 层（包间注意力）
  └── 跨 packet 滑动窗口注意力（header/payload 独立处理）
    │
    ▼
重组序列 → Pooling → Linear 分类器
```

### 各模块功能

| 层级 | 模块 | 功能 |
|------|------|------|
| 包内 | HeaderAttentionLayer | header 40 tokens 全局自注意力，捕获协议字段间关系 |
| 包内 | CrossAttentionLayer | payload 以 header 为 K/V 做交叉注意力，融合协议上下文 |
| 包内 | PayloadAttentionLayer | payload 滑动窗口(20)自注意力，捕获局部载荷模式 |
| 包间 | FlowAttention | 跨 packet 滑动窗口注意力，建模流级别的时序依赖 |

---

## 三、Pre-training vs Fine-tuning 对比

| 维度 | Pre-training | Fine-tuning |
|------|-------------|-------------|
| 注意力架构 | PacketAttentionStack(4层) + FlowAttentionStack(8层) | TwoLevelAttentionEncoder（相同的两层级结构） |
| 训练目标 | MLM（掩码语言建模，双任务头） | 分类任务（Pooling + Linear） |
| 实现位置 | `multi_task_trainer.py` | `Two-task_fine-tuning_two-attention.py` |
| 框架依赖 | 基于 UER 框架的 Transformer 编码器/解码器 | 同样基于 UER 框架 |
| 输出 | 重组序列 → MLM 预测 | 重组序列 → Pooling → 分类 |
| 关键设计 | Header/Payload 分离处理 + 包间流注意力 | 继承预训练权重，替换预测头为分类头 |

---

## 四、核心设计思想

1. **Header/Payload 分离**：网络流量中 header 和 payload 具有不同的结构特征。Header 是结构化的协议字段，适合全局注意力；payload 是变长的应用数据，适合局部窗口注意力。

2. **交叉注意力桥接**：通过 CrossAttentionLayer 让 payload 感知 header 的协议语义，实现两部分信息的融合。

3. **层级化建模**：先在单个 packet 内部建模（包内注意力），再在 packet 之间建模（包间注意力），从局部到全局逐步捕获流量特征。

4. **滑动窗口机制**：在 payload 内部和 packet 之间都使用滑动窗口，兼顾计算效率和局部依赖建模。

5. **池化降维**：在包内到包间的过渡中使用均值池化压缩 payload 表示，降低包间注意力的计算复杂度。
