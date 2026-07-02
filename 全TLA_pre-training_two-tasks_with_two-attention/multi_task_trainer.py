#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
多任务预训练Trainer - 整合两个预训练任务（带两级注意力机制）

1. NSP_Indicating (Packet Masking) - 50%
   - 使用流级别Packet Masking策略（h-h-h-h-h和p-p-p-p-p关系版本）
   - 输入: 流MFR矩阵（前4个数据包，每个80 tokens）
   - Sequence Length: 330 tokens
   - 两种操作，随机选择：
     - 操作1（h-h-h-h-h关系）：将流的四个数据包MFR矩阵的payload都设置为CNS，随机mask某一个数据包MFR矩阵的header，mask率90%
     - 操作2（p-p-p-p-p关系）：将流的四个数据包MFR矩阵的header都设置为CNS，随机mask某一个数据包MFR矩阵的payload，mask率90%

2. MFR-Masking (Header/Payload Masking) - 50%
   - 对每条流，从4个数据包MFR矩阵中随机选择1个数据包进行处理
   - 每个数据包：80 tokens（Header 40 + Payload 40）
   - 数据包独立处理，不组合成流
   - Sequence Length: 82 tokens (CLS + 80 + SEP)，padding到330以匹配NSP_Indicating
   - 三种关系masking策略（按batch_size内的流划分）：
     - h-h关系 (30%)：Payload用[CNS]替代，对Header随机mask 80%
     - p-p关系 (30%)：Header用[CNS]替代，对Payload随机mask 80%
     - h-p关系 (40%)：对Header随机mask 95%，用Payload预测Header
   
数据共享策略：
   - 每个流产生2个样本：1个NSP_Indicating样本 + 1个MFR-Masking样本
   - 保证两个任务的样本数始终相等

两级注意力机制：
   - 包级注意力（PacketAttention）：对每个数据包内部的header-payload关系建模
   - 流级注意力（FlowAttention）：对4个数据包之间的跨包依赖关系建模

输入数据格式：流MFR矩阵（前4个数据包，每个80 tokens）
"""

import sys
import os
import pickle
import random
import time
import math
from datetime import datetime
from collections import defaultdict

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入UER的trainer模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 从ET-BERT中导入必要的模块
et_bert_path = '/root/我的代码和数据集/ET-BERT-main'
sys.path.insert(0, et_bert_path)

from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer import trainer
from uer.targets import str2target
from uer.trainer import str2trainer, BertTrainer
from uer.layers import str2embedding
from uer.encoders import str2encoder

# 导入masking函数
# 添加路径以导入packet_masking和mfr_masking模块
burst_predicting_dir = "/root/我的代码和数据集/Mine/pre-training/pre-training_Burst-Predicting_Indicating"
mfr_masking_dir = "/root/我的代码和数据集/Mine/pre-training/pre-training_MFR-Masking"
if burst_predicting_dir not in sys.path:
    sys.path.insert(0, burst_predicting_dir)
if mfr_masking_dir not in sys.path:
    sys.path.insert(0, mfr_masking_dir)

try:
    from packet_masking import mask_flow_packets
except ImportError:
    print("Warning: Could not import mask_flow_packets from packet_masking")
    mask_flow_packets = None

try:
    from mfr_masking import mask_single_packet_mfr, HEADER_SIZE, PAYLOAD_SIZE
except ImportError:
    print("Warning: Could not import mask_single_packet_mfr from mfr_masking")
    mask_single_packet_mfr = None


class HeaderAttentionLayer(nn.Module):
    """
    第1层：h-h关系 - header内部全局注意力
    对header进行全局自注意力，payload保持不变并传递到下一层
    """
    def __init__(self, hidden_size, num_attention_heads=8, header_size=40, dropout=0.1):
        super(HeaderAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.header_size = header_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, packet_hidden):
        """对header进行全局自注意力，payload保持不变"""
        header_hidden = packet_hidden[:, :self.header_size, :]
        payload_hidden = packet_hidden[:, self.header_size:, :]

        query = self.transpose_for_scores(self.query(header_hidden))
        key = self.transpose_for_scores(self.key(header_hidden))
        value = self.transpose_for_scores(self.value(header_hidden))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        output = self.output(context)
        output = self.dropout(output)
        header_output = self.norm(header_hidden + output)

        return torch.cat([header_output, payload_hidden], dim=1)


class CrossAttentionLayer(nn.Module):
    """
    第2层：h-p关系 - payload作q，header作k和v的交叉注意力
    payload通过注意力机制从header获取信息，header保持不变并传递到下一层
    """
    def __init__(self, hidden_size, num_attention_heads=8, header_size=40, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.header_size = header_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, packet_hidden):
        """payload作q，header作k和v进行交叉注意力，header保持不变"""
        header_hidden = packet_hidden[:, :self.header_size, :]
        payload_hidden = packet_hidden[:, self.header_size:, :]

        query = self.transpose_for_scores(self.query(payload_hidden))
        key = self.transpose_for_scores(self.key(header_hidden))
        value = self.transpose_for_scores(self.value(header_hidden))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        output = self.output(context)
        output = self.dropout(output)
        payload_output = self.norm(payload_hidden + output)

        return torch.cat([header_hidden, payload_output], dim=1)


class PayloadAttentionLayer(nn.Module):
    """
    第3层：p-p关系 - payload内部窗口注意力
    对payload进行窗口自注意力，header保持不变
    """
    def __init__(self, hidden_size, num_attention_heads=8, header_size=40, window_size=20, dropout=0.1):
        super(PayloadAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.header_size = header_size
        self.window_size = window_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, packet_hidden):
        """对payload进行窗口注意力，header保持不变"""
        header_hidden = packet_hidden[:, :self.header_size, :]
        payload_hidden = packet_hidden[:, self.header_size:, :]

        batch_size, seq_len, hidden_size = payload_hidden.size()

        query = self.transpose_for_scores(self.query(payload_hidden))
        key = self.transpose_for_scores(self.key(payload_hidden))
        value = self.transpose_for_scores(self.value(payload_hidden))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 创建窗口mask
        window_mask = torch.ones((seq_len, seq_len), device=payload_hidden.device) * float('-inf')
        half_window = self.window_size // 2

        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            window_mask[i, start:end] = 0.0

        attention_scores = attention_scores + window_mask.unsqueeze(0).unsqueeze(0)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        output = self.output(context)
        output = self.dropout(output)
        payload_output = self.norm(payload_hidden + output)

        return torch.cat([header_hidden, payload_output], dim=1)


class FlowAttention(nn.Module):
    """
    流级注意力机制模块
    处理流的前4个数据包，使用滑动窗口注意力机制
    """
    def __init__(self, hidden_size, num_attention_heads=8, header_size=40, payload_size=40, dropout=0.1):
        super(FlowAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.header_size = header_size
        self.payload_size = payload_size
        self.pooled_payload_size = 4
        
        self.header_query = nn.Linear(hidden_size, hidden_size)
        self.header_key = nn.Linear(hidden_size, hidden_size)
        self.header_value = nn.Linear(hidden_size, hidden_size)
        self.header_output = nn.Linear(hidden_size, hidden_size)
        
        self.payload_query = nn.Linear(hidden_size, hidden_size)
        self.payload_key = nn.Linear(hidden_size, hidden_size)
        self.payload_value = nn.Linear(hidden_size, hidden_size)
        self.payload_output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.header_norm = nn.LayerNorm(hidden_size)
        self.payload_norm = nn.LayerNorm(hidden_size)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def pool_payload(self, payload_hidden):
        batch_size, seq_len, hidden_size = payload_hidden.size()
        payload_reshaped = payload_hidden.view(batch_size, 4, 10, hidden_size)
        pooled_payload = torch.mean(payload_reshaped, dim=2)
        return pooled_payload
    
    def unpool_payload(self, pooled_payload):
        batch_size, pooled_len, hidden_size = pooled_payload.size()
        unpooled = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)
        unpooled = unpooled.view(batch_size, 40, hidden_size)
        return unpooled
    
    def apply_attention(self, query_hidden, key_value_hiddens, is_header=True):
        if is_header:
            query_layer = self.header_query
            key_layer = self.header_key
            value_layer = self.header_value
            output_layer = self.header_output
        else:
            query_layer = self.payload_query
            key_layer = self.payload_key
            value_layer = self.payload_value
            output_layer = self.payload_output

        query = self.transpose_for_scores(query_layer(query_hidden))

        all_keys = []
        all_values = []
        for kv_hidden in key_value_hiddens:
            key = self.transpose_for_scores(key_layer(kv_hidden))
            value = self.transpose_for_scores(value_layer(kv_hidden))
            all_keys.append(key)
            all_values.append(value)

        keys = torch.cat(all_keys, dim=2)
        values = torch.cat(all_values, dim=2)

        attention_scores = torch.matmul(query, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        output = output_layer(context)
        output = self.dropout(output)

        return output

    def apply_attention_with_self(self, query_hidden, all_key_value_hiddens, query_idx, is_header=True):
        """查询窗口内的所有数据包（包括自己和相邻数据包）"""
        if is_header:
            query_layer = self.header_query
            key_layer = self.header_key
            value_layer = self.header_value
            output_layer = self.header_output
        else:
            query_layer = self.payload_query
            key_layer = self.payload_key
            value_layer = self.payload_value
            output_layer = self.payload_output

        query = self.transpose_for_scores(query_layer(query_hidden))

        all_keys = []
        all_values = []
        for kv_hidden in all_key_value_hiddens:
            key = self.transpose_for_scores(key_layer(kv_hidden))
            value = self.transpose_for_scores(value_layer(kv_hidden))
            all_keys.append(key)
            all_values.append(value)

        keys = torch.cat(all_keys, dim=2)
        values = torch.cat(all_values, dim=2)

        attention_scores = torch.matmul(query, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        output = output_layer(context)
        output = self.dropout(output)

        return output
    
    def forward(self, all_packet_hiddens):
        all_headers = []
        all_payloads = []
        all_pooled_payloads = []

        for packet_hidden in all_packet_hiddens:
            header = packet_hidden[:, :self.header_size, :]
            payload = packet_hidden[:, self.header_size:, :]
            pooled_payload = self.pool_payload(payload)

            all_headers.append(header)
            all_payloads.append(payload)
            all_pooled_payloads.append(pooled_payload)

        output_headers = []
        output_payloads = []

        for i in range(4):
            if i == 0:
                header_out = self.apply_attention_with_self(all_headers[0], [all_headers[0], all_headers[1]], i, is_header=True)
                payload_out = self.apply_attention_with_self(all_pooled_payloads[0], [all_pooled_payloads[0], all_pooled_payloads[1]], i, is_header=False)
            elif i == 1:
                header_out = self.apply_attention_with_self(all_headers[1], [all_headers[0], all_headers[1], all_headers[2]], i, is_header=True)
                payload_out = self.apply_attention_with_self(all_pooled_payloads[1], [all_pooled_payloads[0], all_pooled_payloads[1], all_pooled_payloads[2]], i, is_header=False)
            elif i == 2:
                header_out = self.apply_attention_with_self(all_headers[2], [all_headers[1], all_headers[2], all_headers[3]], i, is_header=True)
                payload_out = self.apply_attention_with_self(all_pooled_payloads[2], [all_pooled_payloads[1], all_pooled_payloads[2], all_pooled_payloads[3]], i, is_header=False)
            else:
                header_out = self.apply_attention_with_self(all_headers[3], [all_headers[2], all_headers[3]], i, is_header=True)
                payload_out = self.apply_attention_with_self(all_pooled_payloads[3], [all_pooled_payloads[2], all_pooled_payloads[3]], i, is_header=False)

            header_final = self.header_norm(all_headers[i] + header_out)
            payload_out_unpooled = self.unpool_payload(payload_out)
            payload_final = self.payload_norm(all_payloads[i] + payload_out_unpooled)

            output_headers.append(header_final)
            output_payloads.append(payload_final)

        return output_headers, output_payloads


class PacketAttentionStack(nn.Module):
    """
    包级注意力堆栈 - 串行3层结构
    每个数据包经过3层串行注意力处理：
    第1层：h-h关系（header内部全局注意力）
    第2层：h-p关系（payload从header获取信息的交叉注意力）
    第3层：p-p关系（payload内部窗口注意力）

    然后重复num_layers次（默认4次），形成4组串行3层结构
    """
    def __init__(self, num_layers, hidden_size, num_attention_heads=8, header_size=40, packet_size=80, window_size=20, dropout=0.1):
        super(PacketAttentionStack, self).__init__()
        self.num_layers = num_layers

        # 每一组包含3层串行注意力
        self.layer_groups = nn.ModuleList()
        for _ in range(num_layers):
            group = nn.ModuleList([
                HeaderAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    header_size=header_size,
                    dropout=dropout
                ),
                CrossAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    header_size=header_size,
                    dropout=dropout
                ),
                PayloadAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    header_size=header_size,
                    window_size=window_size,
                    dropout=dropout
                )
            ])
            self.layer_groups.append(group)

    def forward(self, packet_hidden):
        """
        对单个数据包应用包级注意力堆栈
        每组包含3层串行注意力：h-h → h-p → p-p

        Args:
            packet_hidden: [batch_size, seq_length, hidden_size] 单个数据包的隐层表示

        Returns:
            output: [batch_size, seq_length, hidden_size] 处理后的表示
        """
        output = packet_hidden
        for group in self.layer_groups:
            # 串行执行3层注意力
            output = group[0](output)  # 第1层：h-h关系
            output = group[1](output)  # 第2层：h-p关系
            output = group[2](output)  # 第3层：p-p关系
        return output


class FlowAttentionStack(nn.Module):
    """
    8层流级注意力堆栈
    处理流的前4个数据包，使用滑动窗口注意力机制
    """
    def __init__(self, num_layers, hidden_size, num_attention_heads=8, header_size=40, payload_size=40, dropout=0.1):
        super(FlowAttentionStack, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            FlowAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                header_size=header_size,
                payload_size=payload_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, all_packet_hiddens):
        """
        对4个数据包应用8层流级注意力

        Args:
            all_packet_hiddens: list of 4 tensors, each [batch_size, seq_length, hidden_size]

        Returns:
            output_headers: list of 4 tensors
            output_payloads: list of 4 tensors
        """
        output_packets = all_packet_hiddens
        output_headers = None
        output_payloads = None

        for layer in self.layers:
            output_headers, output_payloads = layer(output_packets)
            # Reconstruct full packets for next layer
            output_packets = [torch.cat([h, p], dim=1) for h, p in zip(output_headers, output_payloads)]

        return output_headers, output_payloads


class PoolingLayer(nn.Module):
    """
    池化层：在PacketAttention之后，FlowAttention之前
    保持原来的池化方式（对payload进行mean pooling）
    """
    def __init__(self, header_size=40, payload_size=40):
        super(PoolingLayer, self).__init__()
        self.header_size = header_size
        self.payload_size = payload_size

    def forward(self, all_packet_hiddens):
        """
        对4个数据包进行池化

        Args:
            all_packet_hiddens: list of 4 tensors, each [batch_size, 80, hidden_size]

        Returns:
            pooled_packets: list of 4 tensors, each [batch_size, 80, hidden_size]（payload部分被池化后恢复）
        """
        pooled_packets = []
        for packet_hidden in all_packet_hiddens:
            header = packet_hidden[:, :self.header_size, :]
            payload = packet_hidden[:, self.header_size:, :]

            # 对payload进行mean pooling（保持原来的方式）
            batch_size, seq_len, hidden_size = payload.size()
            payload_reshaped = payload.view(batch_size, 4, 10, hidden_size)
            pooled_payload = torch.mean(payload_reshaped, dim=2)  # [batch_size, 4, hidden_size]

            # 通过repeat恢复到原始形状 [batch_size, 40, hidden_size]
            unpooled_payload = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)
            unpooled_payload = unpooled_payload.view(batch_size, 40, hidden_size)

            # 拼接header和unpooled payload
            pooled_packet = torch.cat([header, unpooled_payload], dim=1)
            pooled_packets.append(pooled_packet)

        return pooled_packets


class BaseModel(nn.Module):
    """
    基础模型：新架构 = embedding + 4层PacketAttention + 池化 + 8层FlowAttention
    去掉标准自注意力机制（encoder）
    必须在模块级别定义，以便支持多进程训练时的pickle
    """
    def __init__(self, embedding, packet_attention_stack=None, pooling_layer=None, flow_attention_stack=None,
                 packet_size=None, header_size=None):
        super(BaseModel, self).__init__()
        self.embedding = embedding
        self.packet_attention_stack = packet_attention_stack
        self.pooling_layer = pooling_layer
        self.flow_attention_stack = flow_attention_stack
        self.packet_size = packet_size
        self.header_size = header_size

    def forward(self, src, seg):
        """
        前向传播：embedding → 4层PacketAttention → 池化 → 8层FlowAttention

        Args:
            src: [batch_size, seq_length] 输入序列（330 tokens）
            seg: [batch_size, seq_length] segment标记

        Returns:
            output: [batch_size, seq_length, hidden_size] 最终输出
        """
        # Embedding
        emb = self.embedding(src, seg)  # [batch_size, seq_length, hidden_size]

        # 将序列分解为4个数据包（每个80 tokens）
        # 假设seq_length = 330 = 1(CLS) + 320(4*80) + 1(SEP) + padding
        # 实际处理的是320个tokens（4个数据包）
        batch_size, seq_length, hidden_size = emb.size()

        # 提取4个数据包的embedding（跳过CLS token）
        # 假设结构：[CLS] + packet1(80) + packet2(80) + packet3(80) + packet4(80) + [SEP] + padding
        packet_embeddings = []
        for i in range(4):
            start_idx = 1 + i * 80  # 跳过CLS
            end_idx = start_idx + 80

            # 确保不超出序列长度
            if end_idx > seq_length:
                end_idx = seq_length

            packet_emb = emb[:, start_idx:end_idx, :]  # [batch_size, <=80, hidden_size]

            # 如果packet大小不足80，进行padding
            if packet_emb.size(1) < 80:
                padding = torch.zeros(batch_size, 80 - packet_emb.size(1), hidden_size, device=emb.device)
                packet_emb = torch.cat([packet_emb, padding], dim=1)

            packet_embeddings.append(packet_emb)

        # 4层PacketAttention处理每个数据包
        packet_outputs = []
        for packet_emb in packet_embeddings:
            packet_out = self.packet_attention_stack(packet_emb)  # [batch_size, 80, hidden_size]
            packet_outputs.append(packet_out)

        # 池化层
        pooled_packets = self.pooling_layer(packet_outputs)  # list of 4 tensors

        # 8层FlowAttention处理流级依赖
        flow_headers, flow_payloads = self.flow_attention_stack(pooled_packets)  # list of 4 tensors each

        # 拼接所有输出
        # 重新组织为完整的序列：[CLS] + packet1 + packet2 + packet3 + packet4 + [SEP] + padding
        cls_token = emb[:, 0:1, :]  # [batch_size, 1, hidden_size]
        sep_token = emb[:, 321:322, :]  # [batch_size, 1, hidden_size] (假设SEP在321位置)

        # 拼接header和payload
        all_packets = []
        for header, payload in zip(flow_headers, flow_payloads):
            packet = torch.cat([header, payload], dim=1)  # [batch_size, 80, hidden_size]
            all_packets.append(packet)

        # 拼接所有数据包
        packets_concat = torch.cat(all_packets, dim=1)  # [batch_size, 320, hidden_size]

        # 最终输出：[CLS] + packets + [SEP] + padding
        output = torch.cat([cls_token, packets_concat, sep_token], dim=1)  # [batch_size, 322, hidden_size]

        # 填充到原始长度
        if output.size(1) < seq_length:
            padding = torch.zeros(batch_size, seq_length - output.size(1), hidden_size, device=output.device)
            output = torch.cat([output, padding], dim=1)
        elif output.size(1) > seq_length:
            output = output[:, :seq_length, :]

        return output


class MultiTaskModel(nn.Module):
    """
    多任务预训练模型（带两级注意力机制）
    包含两个任务头：
    1. NSP_Indicating: MLM头（预测masked tokens）
    2. MFR-Masking: MLM头（预测masked tokens）
    
    两级注意力机制:
    1. 包级注意力（PacketAttention）: 对每个数据包内部的header-payload关系建模
    2. 流级注意力（FlowAttention）: 对4个数据包之间的跨包依赖关系建模
    """
    def __init__(self, base_model, args, vocab_size):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.args = args

        # 任务权重
        self.nsp_indicating_weight = args.nsp_indicating_weight
        self.mfr_masking_weight = args.mfr_masking_weight

        # NSP_Indicating任务头（MLM）
        self.indicating_mlm_output = nn.Linear(args.hidden_size, vocab_size)

        # MFR-Masking任务头（MLM）
        self.mfr_mlm_output = nn.Linear(args.hidden_size, vocab_size)

        # 损失函数
        self.mlm_loss = nn.CrossEntropyLoss()
    
    def forward(self, src, seg, task_type, tgt_mlm=None):
        """
        前向传播

        Args:
            src: [batch_size, seq_length] 输入序列
            seg: [batch_size, seq_length] segment标记
            task_type: str, 任务类型 ('nsp_indicating', 'mfr_masking')
            tgt_mlm: [batch_size, seq_length] MLM任务的目标

        Returns:
            loss: 任务损失
            correct: 正确预测的数量（用于accuracy计算）
        """
        # 获取base model的输出（已包含两级注意力机制 + 12层encoder）
        output = self.base_model(src, seg)

        if task_type == 'nsp_indicating':
            # NSP_Indicating任务：MLM预测
            mlm_logits = self.indicating_mlm_output(output)  # [batch_size, seq_length, vocab_size]
            
            if tgt_mlm is not None:
                # 只计算被mask的token的loss
                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_mlm_flat = tgt_mlm.view(-1)
                
                # 过滤掉padding（假设padding的label为-1或0）
                mask = tgt_mlm_flat > 0
                if mask.sum() > 0:
                    mlm_logits_masked = mlm_logits_flat[mask]
                    tgt_mlm_masked = tgt_mlm_flat[mask]
                    loss = self.mlm_loss(mlm_logits_masked, tgt_mlm_masked)
                    
                    pred = torch.argmax(mlm_logits_masked, dim=-1)
                    correct = torch.sum(pred == tgt_mlm_masked).float()
                else:
                    loss = torch.tensor(0.0, device=output.device)
                    correct = torch.tensor(0.0, device=output.device)
                
                return loss, correct
            else:
                return mlm_logits
        
        elif task_type == 'mfr_masking':
            # MFR-Masking任务：MLM预测
            mlm_logits = self.mfr_mlm_output(output)  # [batch_size, seq_length, vocab_size]
            
            if tgt_mlm is not None:
                # 只计算被mask的token的loss
                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_mlm_flat = tgt_mlm.view(-1)
                
                # 过滤掉padding（假设padding的label为-1或0）
                mask = tgt_mlm_flat > 0
                if mask.sum() > 0:
                    mlm_logits_masked = mlm_logits_flat[mask]
                    tgt_mlm_masked = tgt_mlm_flat[mask]
                    loss = self.mlm_loss(mlm_logits_masked, tgt_mlm_masked)
                    
                    pred = torch.argmax(mlm_logits_masked, dim=-1)
                    correct = torch.sum(pred == tgt_mlm_masked).float()
                else:
                    loss = torch.tensor(0.0, device=output.device)
                    correct = torch.tensor(0.0, device=output.device)
                
                return loss, correct
            else:
                return mlm_logits
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class MultiTaskDataset(Dataset):
    """
    多任务数据加载器
    两个任务共享同一批流数据，每个流被两个任务分别处理
    
    数据共享策略：
    - 每个step加载N个流样本（N = batch_size，例如8个）
    - 这N个流同时被两个任务处理
    - 任务1 (NSP_Indicating): N个样本（每流1个样本）
    - 任务2 (MFR-Masking): 4N个样本（每流4个数据包，每个数据包1个样本）
    - 总计：每个step使用N个流，产生5N个训练样本
    
    输入数据格式：
    - 原始数据：流MFR矩阵（前5个数据包，每个160 tokens）
    - 处理后：流MFR矩阵（前4个数据包，每个80 tokens）
    - NSP_Indicating: 使用流级别Packet Masking策略（h-h-h-h-h和p-p-p-p-p关系版本），Sequence Length: 330
    - MFR-Masking: 每个数据包独立处理，三种关系masking策略（h-h, p-p, h-p），Sequence Length: 82
    """
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=True):
        self.args = args
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        
        self.seq_length = args.seq_length
        self.tokens_per_packet = args.tokens_per_packet
        
        # 任务权重（用于采样）
        self.task_weights = [
            args.nsp_indicating_weight,
            args.mfr_masking_weight
        ]
        self.task_names = ['nsp_indicating', 'mfr_masking']
        
        # Masking参数
        self.packet_selection_prob = args.packet_selection_prob
        self.packet_mask_ratio = args.packet_mask_ratio
        self.mfr_header_mask_ratio = args.mfr_header_mask_ratio
        self.mfr_payload_mask_ratio = args.mfr_payload_mask_ratio
        
        # MFR-Masking任务参数
        self.hh_ratio = getattr(args, 'hh_ratio', 0.30)  # h-h关系的数据包比例
        self.pp_ratio = getattr(args, 'pp_ratio', 0.30)  # p-p关系的数据包比例
        self.hp_ratio = getattr(args, 'hp_ratio', 0.40)  # h-p关系的数据包比例
        self.hh_mask_ratio = getattr(args, 'hh_mask_ratio', 0.80)  # h-h关系的header mask比例
        self.pp_mask_ratio = getattr(args, 'pp_mask_ratio', 0.80)  # p-p关系的payload mask比例
        self.hp_mask_ratio = getattr(args, 'hp_mask_ratio', 0.95)  # h-p关系的header mask比例
        
        # 特殊token IDs
        self.vocab = args.vocab
        self.mask_id = self.vocab.get('[MASK]', 103)
        self.cls_id = self.vocab.get('[CLS]', 101)
        self.sep_id = self.vocab.get('[SEP]', 102)
        self.pad_id = self.vocab.get('[PAD]', 0)
        
        # ⚠️ 改用缓冲式加载 - 不再一次性加载全部数据！
        # 使用极小的buffer以快速启动训练: batch_size * 3（最快启动速度）
        self.instances_buffer_size = min(getattr(args, 'instances_buffer_size', 25600), batch_size * 50)
        self.buffer = []
        self.buffer_start = 0
        self.buffer_end = 0
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.initial_load = True  # 首次加载标志
        
        if self.proc_id == 0:
            print(f"[Rank {self.proc_id}] MultiTaskDataset初始化完成")
            print(f"[Rank {self.proc_id}] - 首次buffer目标: batch_size × 2 = {batch_size * 2} (快速启动)")
            print(f"[Rank {self.proc_id}] - 后续buffer目标: {self.instances_buffer_size}")
            print(f"[Rank {self.proc_id}] - 策略: 边加载边训练，GPU不等待")
        
        # 用于控制首次信息输出
        self.first_batch_generated = False
    
    def _fill_buffer(self):
        """
        填充buffer - 从数据集中按需加载样本（缓冲式加载，内存友好）
        首次加载更快，后续渐进式填充
        """
        self.buffer = []
        
        # 首次加载：使用极小的buffer快速启动（batch_size * 2，立即开始训练）
        target_size = self.instances_buffer_size
        if self.initial_load:
            target_size = min(self.batch_size * 2, self.instances_buffer_size)
            if self.proc_id == 0:
                print(f"[Rank {self.proc_id}] 首次填充buffer: 目标大小={target_size} (batch_size × 2，快速启动模式)")
        
        loaded = 0
        start_time = time.time()
        while len(self.buffer) < target_size:
            try:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                loaded += 1
                
                # 多进程训练：每个进程只处理自己负责的数据
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                
                # 显示进度
                if self.initial_load and self.proc_id == 0 and loaded % 1000 == 0:
                    elapsed = time.time() - start_time
                    speed = loaded / elapsed if elapsed > 0 else 0
                    print(f"[Rank {self.proc_id}] 已读取 {loaded} 条数据, buffer大小={len(self.buffer)}, 速度={speed:.1f} samples/s")
                    
            except EOFError:
                # 到达文件末尾，重置并继续填充buffer（循环读取）
                self.dataset_reader.seek(0)
                self.read_count = 0
                
                # 如果buffer还不够，继续从头读取
                if len(self.buffer) == 0:
                    break
        
        if self.shuffle:
            random.shuffle(self.buffer)
        
        self.buffer_start = 0
        self.buffer_end = len(self.buffer)
        
        # 首次加载完成
        if self.initial_load:
            elapsed = time.time() - start_time
            if self.proc_id == 0:
                print(f"[Rank {self.proc_id}] 首次buffer填充完成: 读取={loaded}条, buffer={len(self.buffer)}条, 耗时={elapsed:.2f}s")
            self.initial_load = False
    
    def _is_buffer_empty(self):
        """检查buffer是否为空"""
        return self.buffer_start >= self.buffer_end
    
    def __del__(self):
        """析构函数 - 关闭文件"""
        if hasattr(self, 'dataset_reader'):
            self.dataset_reader.close()
    
    def _organize_buffer_data(self, instances):
        """
        从buffer中的实例组织数据（动态构建，不保留历史数据）
        每次填充新buffer时重新组织，减少内存占用
        
        Args:
            instances: 从buffer中获取的实例列表
        
        Returns:
            flows: 解析后的流列表
        """
        start_time = time.time()
        
        flows = []  # 临时存储flows
        raw_packets = []  # 临时存储raw_packets
        
        # 首次运行时打印数据格式信息
        if self.proc_id == 0 and not self.first_batch_generated and len(instances) > 0:
            first_instance = instances[0]
            print(f"[DEBUG] 第一个instance类型: {type(first_instance)}")
            print(f"[DEBUG] 第一个instance长度: {len(first_instance) if hasattr(first_instance, '__len__') else 'N/A'}")
            if isinstance(first_instance, (tuple, list)) and len(first_instance) <= 10:
                print(f"[DEBUG] 第一个instance内容: {first_instance}")
            elif isinstance(first_instance, (tuple, list)):
                print(f"[DEBUG] 第一个instance前10个元素: {first_instance[:10]}")
                print(f"[DEBUG] 第一个instance元素类型: {type(first_instance[0])}")
        
        # 解析buffer中的实例
        parse_error_count = 0
        for idx, instance in enumerate(instances):
            try:
                # ⚠️ 关键修复：先检查原始数据包格式（tuple），然后才处理流MFR矩阵格式
                if isinstance(instance, tuple) and len(instance) == 4:
                    # 原始数据包格式: (flow_id, packet_idx, packet_tokens, dataset_name)
                    raw_packets.append(instance)
                elif isinstance(instance, (list, tuple)):
                    # 流MFR矩阵格式: 解析为整数列表
                    # 原始数据: 5个数据包 × 160 tokens = 800 tokens
                    # 处理后: 4个数据包 × 80 tokens = 320 tokens
                    
                    # 首先将原始数据转换为整数（与All版本一致）
                    raw_flow = []
                    for token in instance:
                        if isinstance(token, int):
                            raw_flow.append(token)
                        elif isinstance(token, str):
                            try:
                                # 尝试十六进制转换
                                val = int(token, 16)
                                raw_flow.append(val + 5)  # +5偏移
                            except (ValueError, TypeError):
                                try:
                                    # 尝试十进制转换
                                    raw_flow.append(int(token))
                                except (ValueError, TypeError):
                                    raw_flow.append(self.pad_id)
                        else:
                            try:
                                raw_flow.append(int(token))
                            except (ValueError, TypeError):
                                raw_flow.append(self.pad_id)
                    
                    # 从5个数据包（每个160 tokens）提取前4个数据包的前80 tokens
                    flow = []
                    for pkt_idx in range(4):  # 只取前4个数据包
                        start = pkt_idx * 160  # 原始每个数据包160 tokens
                        # 每个数据包只取前80 tokens
                        packet_tokens = raw_flow[start:start+80]
                        
                        # 确保每个packet正好80 tokens
                        if len(packet_tokens) < 80:
                            packet_tokens = packet_tokens + [self.pad_id] * (80 - len(packet_tokens))
                        flow.extend(packet_tokens[:80])
                    
                    # 确保最终长度为320 tokens (4 packets × 80 tokens)
                    expected_len = self.tokens_per_packet * 4  # 320
                    if len(flow) < expected_len:
                        flow = flow + [self.pad_id] * (expected_len - len(flow))
                    elif len(flow) > expected_len:
                        flow = flow[:expected_len]
                    
                    flows.append(flow)
            except Exception as e:
                parse_error_count += 1
                if parse_error_count <= 3 and self.proc_id == 0 and not self.first_batch_generated:
                    print(f"[DEBUG] 解析错误 #{parse_error_count}: {type(e).__name__}: {e}")
                    print(f"[DEBUG] 问题instance类型: {type(instance)}, 长度: {len(instance) if hasattr(instance, '__len__') else 'N/A'}")
                continue
        
        # ⚠️ 关键修复：如果有raw_packets，需要重建flows列表
        if len(raw_packets) > 0:
            # 按flow_id分组
            flow_dict = {}
            for flow_id, packet_idx, packet_tokens, dataset_name in raw_packets:
                if flow_id not in flow_dict:
                    flow_dict[flow_id] = []
                flow_dict[flow_id].append((packet_idx, packet_tokens))
            
            # 对每个流，按packet_idx排序，取前4个包，拼接成流MFR矩阵
            for flow_id, packets in flow_dict.items():
                # 按packet_idx排序
                packets.sort(key=lambda x: x[0])
                # 取前4个包
                packets = packets[:4]
                
                # 只有当流有至少1个数据包时才处理
                if len(packets) == 0:
                    continue
                
                # 拼接成320 tokens的流 (4 packets × 80 tokens)
                flow_tokens = []
                for _, packet_tokens in packets:
                    # 原始数据每个packet是160 tokens
                    # 处理：只取前80个tokens
                    packet_int_tokens = []
                    for token_str in packet_tokens[:80]:  # 只取前80个tokens
                        try:
                            val = int(token_str, 16)  # 十六进制转换
                            packet_int_tokens.append(val + 5)  # +5偏移
                        except (ValueError, TypeError):
                            try:
                                packet_int_tokens.append(int(token_str))  # 十进制
                            except (ValueError, TypeError):
                                packet_int_tokens.append(self.pad_id)
                    
                    # 确保每个packet正好80 tokens
                    while len(packet_int_tokens) < 80:
                        packet_int_tokens.append(self.pad_id)
                    flow_tokens.extend(packet_int_tokens[:80])
                
                # 确保长度为320 tokens (4 packets × 80 tokens)
                while len(flow_tokens) < 320:
                    flow_tokens.append(self.pad_id)
                if len(flow_tokens) > 320:
                    flow_tokens = flow_tokens[:320]
                
                flows.append(flow_tokens)
        
        # 组织完成，打印统计
        elapsed = time.time() - start_time
        if self.proc_id == 0 and not self.first_batch_generated:
            print(f"[Rank {self.proc_id}] 数据组织完成: {len(flows)}个流, {len(raw_packets)}个原始包, 耗时{elapsed:.2f}s")
            if parse_error_count > 0:
                print(f"[DEBUG] 解析错误总数: {parse_error_count}/{len(instances)}")
            if len(flows) == 0 and len(raw_packets) == 0:
                print(f"[ERROR] ❌ 没有成功解析任何数据！请检查数据格式。")
        
        # 返回flows用于MLM任务
        return flows
    
    def _apply_packet_masking(self, tokens):
        """
        应用Packet Masking策略（NSP_Indicating任务）
        使用流级别Packet Masking策略（h-h-h-h-h和p-p-p-p-p关系版本）
        
        两种操作，随机选择：
        操作1（h-h-h-h-h关系）：
          - 将流的四个数据包MFR矩阵的payload都设置为CNS（不预测，不用于预测）
          - 随机mask某一个数据包MFR矩阵的header，mask率90%
          - 用流的四个数据包MFR矩阵中剩余没被mask的header预测被mask的header
        
        操作2（p-p-p-p-p关系）：
          - 将流的四个数据包MFR矩阵的header都设置为CNS（不预测，不用于预测）
          - 随机mask某一个数据包MFR矩阵的payload，mask率90%
          - 用流的四个数据包MFR矩阵中剩余没被mask的payload预测被mask的payload
        
        Args:
            tokens: 原始tokens列表（包含4个数据包，每个80 tokens = 320 tokens）
        
        Returns:
            masked_tokens: masked后的tokens
            labels: 原始tokens（用于计算loss，-1表示不需要预测）
        """
        # 确保tokens长度正确
        expected_len = self.tokens_per_packet * 4  # 320
        if len(tokens) < expected_len:
            tokens = tokens + [self.pad_id] * (expected_len - len(tokens))
        elif len(tokens) > expected_len:
            tokens = tokens[:expected_len]
        
        masked_tokens = tokens.copy()
        labels = [-1] * len(tokens)  # -1表示不需要预测
        
        # 获取CNS token ID
        cns_id = self.vocab.get('[CNS]', self.pad_id)
        header_size = 40
        payload_size = 40
        
        # 随机选择操作1（h-h-h-h-h）或操作2（p-p-p-p-p）
        use_hhhh_relation = random.random() < 0.5
        
        if use_hhhh_relation:
            # 操作1：h-h-h-h-h关系
            # 1. 将四个数据包的payload都设置为CNS
            for packet_idx in range(4):
                packet_start = packet_idx * self.tokens_per_packet
                payload_start = packet_start + header_size
                payload_end = packet_start + self.tokens_per_packet
                
                for i in range(payload_start, payload_end):
                    if i < len(tokens):
                        masked_tokens[i] = cns_id
                        labels[i] = -1  # CNS不需要预测
            
            # 2. 随机选择某一个数据包的header进行mask（mask率90%）
            selected_packet_idx = random.randint(0, 3)
            packet_start = selected_packet_idx * self.tokens_per_packet
            header_start = packet_start
            header_end = packet_start + header_size
            
            # 获取该数据包header区域的tokens（排除PAD）
            header_positions = [i for i in range(header_start, header_end) 
                              if i < len(tokens) and tokens[i] != self.pad_id]
            
            if len(header_positions) > 0:
                # 随机选择90%的tokens进行mask
                num_to_mask = max(1, int(len(header_positions) * 0.90))
                mask_positions = random.sample(header_positions, min(num_to_mask, len(header_positions)))
                
                for pos in mask_positions:
                    if pos < len(tokens) and tokens[pos] != self.pad_id:
                        # 直接使用原始token ID作为label（与All版本一致）
                        labels[pos] = tokens[pos]
                        
                        # 80% [MASK], 10% random, 10% unchanged
                        rand = random.random()
                        if rand < 0.8:
                            masked_tokens[pos] = self.mask_id
                        elif rand < 0.9:
                            masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
                        # else: keep unchanged
        else:
            # 操作2：p-p-p-p-p关系
            # 1. 将四个数据包的header都设置为CNS
            for packet_idx in range(4):
                packet_start = packet_idx * self.tokens_per_packet
                header_start = packet_start
                header_end = packet_start + header_size
                
                for i in range(header_start, header_end):
                    if i < len(tokens):
                        masked_tokens[i] = cns_id
                        labels[i] = -1  # CNS不需要预测
            
            # 2. 随机选择某一个数据包的payload进行mask（mask率90%）
            selected_packet_idx = random.randint(0, 3)
            packet_start = selected_packet_idx * self.tokens_per_packet
            payload_start = packet_start + header_size
            payload_end = packet_start + self.tokens_per_packet
            
            # 获取该数据包payload区域的tokens（排除PAD）
            payload_positions = [i for i in range(payload_start, payload_end) 
                               if i < len(tokens) and tokens[i] != self.pad_id]
            
            if len(payload_positions) > 0:
                # 随机选择90%的tokens进行mask
                num_to_mask = max(1, int(len(payload_positions) * 0.90))
                mask_positions = random.sample(payload_positions, min(num_to_mask, len(payload_positions)))
                
                for pos in mask_positions:
                    if pos < len(tokens) and tokens[pos] != self.pad_id:
                        # 直接使用原始token ID作为label（与All版本一致）
                        labels[pos] = tokens[pos]
                        
                        # 80% [MASK], 10% random, 10% unchanged
                        rand = random.random()
                        if rand < 0.8:
                            masked_tokens[pos] = self.mask_id
                        elif rand < 0.9:
                            masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
                        # else: keep unchanged
        
        return masked_tokens, labels
    
    def _apply_mfr_masking_single_packet(self, packet_tokens, relation_type):
        """
        应用MFR Masking策略（MFR-Masking任务）- 单个数据包版本
        每个数据包独立处理，不组合成流
        
        三种关系masking策略：
        1. h-h关系：Payload用[CNS]替代，对Header随机mask 80%
        2. p-p关系：Header用[CNS]替代，对Payload随机mask 80%
        3. h-p关系：对Header随机mask 95%，用Payload预测Header
        
        Args:
            packet_tokens: 单个数据包的tokens列表（80 tokens: Header 40 + Payload 40）
            relation_type: 关系类型 ('hh', 'pp', 'hp')
        
        Returns:
            masked_tokens: masked后的tokens
            labels: 原始tokens（用于计算loss，-1表示不需要预测）
        """
        # 确保tokens长度正确（80 tokens）
        if len(packet_tokens) < self.tokens_per_packet:
            packet_tokens = packet_tokens + [self.pad_id] * (self.tokens_per_packet - len(packet_tokens))
        elif len(packet_tokens) > self.tokens_per_packet:
            packet_tokens = packet_tokens[:self.tokens_per_packet]
        
        masked_tokens = packet_tokens.copy()
        labels = [-1] * len(packet_tokens)  # -1表示不需要预测
        
        # 获取CNS token ID
        cns_id = self.vocab.get('[CNS]', self.pad_id)
        header_size = 40
        payload_size = 40
        
        header_start = 0
        header_end = header_size
        payload_start = header_size
        payload_end = self.tokens_per_packet
        
        if relation_type == 'hh':
            # h-h关系：Payload用[CNS]替代，对Header随机mask 80%
            # 1. 将payload设置为CNS
            for i in range(payload_start, payload_end):
                if i < len(packet_tokens):
                    masked_tokens[i] = cns_id
                    labels[i] = -1  # CNS不需要预测
            
            # 2. 对header随机mask 80%（排除PAD）
            header_positions = [i for i in range(header_start, header_end) 
                              if i < len(packet_tokens) and packet_tokens[i] != self.pad_id]
            num_to_mask = max(1, int(len(header_positions) * 0.80))
            mask_positions = random.sample(header_positions, min(num_to_mask, len(header_positions))) if len(header_positions) > 0 else []
            
            for pos in mask_positions:
                if pos < len(packet_tokens) and packet_tokens[pos] != self.pad_id:
                    # 直接使用原始token ID作为label（与All版本一致）
                    labels[pos] = packet_tokens[pos]
                    
                    # 80% [MASK], 10% random, 10% unchanged
                    rand = random.random()
                    if rand < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif rand < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
                    # else: keep unchanged
        
        elif relation_type == 'pp':
            # p-p关系：Header用[CNS]替代，对Payload随机mask 80%
            # 1. 将header设置为CNS
            for i in range(header_start, header_end):
                if i < len(packet_tokens):
                    masked_tokens[i] = cns_id
                    labels[i] = -1  # CNS不需要预测
            
            # 2. 对payload随机mask 80%（排除PAD）
            payload_positions = [i for i in range(payload_start, payload_end) 
                               if i < len(packet_tokens) and packet_tokens[i] != self.pad_id]
            num_to_mask = max(1, int(len(payload_positions) * 0.80))
            mask_positions = random.sample(payload_positions, min(num_to_mask, len(payload_positions))) if len(payload_positions) > 0 else []
            
            for pos in mask_positions:
                if pos < len(packet_tokens) and packet_tokens[pos] != self.pad_id:
                    # 直接使用原始token ID作为label（与All版本一致）
                    labels[pos] = packet_tokens[pos]
                    
                    # 80% [MASK], 10% random, 10% unchanged
                    rand = random.random()
                    if rand < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif rand < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
                    # else: keep unchanged
        
        elif relation_type == 'hp':
            # h-p关系：对Header随机mask 95%，用Payload预测Header
            # payload保持原样（不替换，用于预测）
            # 对header随机mask 95%（排除PAD）
            header_positions = [i for i in range(header_start, header_end) 
                              if i < len(packet_tokens) and packet_tokens[i] != self.pad_id]
            num_to_mask = max(1, int(len(header_positions) * 0.95))
            mask_positions = random.sample(header_positions, min(num_to_mask, len(header_positions))) if len(header_positions) > 0 else []
            
            for pos in mask_positions:
                if pos < len(packet_tokens) and packet_tokens[pos] != self.pad_id:
                    # 直接使用原始token ID作为label（与All版本一致）
                    labels[pos] = packet_tokens[pos]
                    
                    # 80% [MASK], 10% random, 10% unchanged
                    rand = random.random()
                    if rand < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif rand < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
                    # else: keep unchanged
        
        return masked_tokens, labels
    
    def _create_nsp_indicating_sample(self, flow):
        """
        创建NSP_Indicating任务样本（Packet Masking）
        
        Args:
            flow: 流的数据包tokens（4个数据包×80 tokens = 320 tokens）
        
        Returns:
            src: 输入序列（masked）
            seg: segment标记
            tgt_mlm: MLM标签
        """
        # 确保flow长度正确
        expected_len = self.tokens_per_packet * 4  # 320
        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]
        
        # 应用Packet Masking
        masked_tokens, labels = self._apply_packet_masking(flow)
        
        # 构建输入：[CLS] + masked_tokens
        src = [self.cls_id] + masked_tokens
        seg = [0] * len(src)
        tgt_mlm = [-1] + labels  # 对应地添加CLS的标签
        
        # 截断或填充到seq_length
        if len(src) > self.seq_length:
            src = src[:self.seq_length]
            seg = seg[:self.seq_length]
            tgt_mlm = tgt_mlm[:self.seq_length]
        while len(src) < self.seq_length:
            src.append(self.pad_id)
            seg.append(0)
            tgt_mlm.append(-1)
        
        return src, seg, tgt_mlm
    
    def _create_mfr_masking_sample(self, flow, packet_idx, relation_type):
        """
        创建MFR-Masking任务样本（单个数据包版本）
        对指定数据包进行处理
        
        Args:
            flow: 流的数据包tokens（4个数据包×80 tokens = 320 tokens）
            packet_idx: 数据包索引 (0-3)
            relation_type: 关系类型 ('hh', 'pp', 'hp')
        
        Returns:
            src: 输入序列（masked，单个数据包）
            seg: segment标记
            tgt_mlm: MLM标签
        """
        # 确保flow长度正确
        expected_len = self.tokens_per_packet * 4  # 320
        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]
        
        # 提取指定数据包
        packet_start = packet_idx * self.tokens_per_packet
        packet_end = packet_start + self.tokens_per_packet
        packet_tokens = flow[packet_start:packet_end]
        
        # 应用MFR Masking（单个数据包）
        masked_tokens, labels = self._apply_mfr_masking_single_packet(packet_tokens, relation_type)
        
        # 构建输入：[CLS] + masked_tokens + [SEP]
        # 注意：为了在同一个batch中处理，MFR-Masking样本也需要padding到seq_length（330）
        # 虽然每个数据包只有82 tokens（CLS + 80 + SEP），但需要padding到330以匹配NSP_Indicating
        src = [self.cls_id] + masked_tokens + [self.sep_id]
        seg = [0] * len(src)
        tgt_mlm = [-1] + labels + [-1]  # CLS和SEP不需要预测
        
        # 截断或填充到seq_length（330，与NSP_Indicating保持一致）
        if len(src) > self.seq_length:
            src = src[:self.seq_length]
            seg = seg[:self.seq_length]
            tgt_mlm = tgt_mlm[:self.seq_length]
        while len(src) < self.seq_length:
            src.append(self.pad_id)
            seg.append(0)
            tgt_mlm.append(-1)
        
        return src, seg, tgt_mlm
    
    def __iter__(self):
        """
        迭代器 - 采用缓冲式流式加载（内存优化版本）
        策略：按需填充buffer → 从buffer生成batch → 循环读取
        
        数据共享机制：
        - 每个batch包含N个流样本（N = batch_size）
        - 每个流被两个任务分别处理
        - NSP_Indicating: N个样本（每流1个样本）
        - MFR-Masking: 4N个样本（每流4个数据包，每个数据包1个样本）
        - 返回的batch包含所有任务的样本（按任务顺序：indicating → masking）
        """
        if self.proc_id == 0:
            print(f"[Rank {self.proc_id}] __iter__() 开始，准备加载数据...")
        
        iter_count = 0
        while True:
            # 当buffer为空时，填充新数据
            if self._is_buffer_empty():
                if self.proc_id == 0 and iter_count == 0:
                    print(f"[Rank {self.proc_id}] Buffer为空，开始填充...")
                    
                self._fill_buffer()
                
                if len(self.buffer) == 0:
                    break
                
                # 从buffer中获取一批实例并组织数据
                # 取一部分实例用于生成多个batch
                batch_instances = self.buffer[self.buffer_start:min(self.buffer_start + self.batch_size * 10, self.buffer_end)]
                self.buffer_start += len(batch_instances)
                
                if self.proc_id == 0 and iter_count == 0:
                    print(f"[Rank {self.proc_id}] 开始组织数据: {len(batch_instances)}条实例...")
                
                # 组织数据（动态构建，用完即释放）
                current_flows = self._organize_buffer_data(batch_instances)
                
                if self.proc_id == 0 and iter_count == 0:
                    print(f"[Rank {self.proc_id}] 数据组织完成: 得到{len(current_flows)}个流")
            else:
                # 从当前buffer继续获取实例
                if self.buffer_start >= self.buffer_end:
                    continue
                
                batch_instances = self.buffer[self.buffer_start:min(self.buffer_start + self.batch_size * 10, self.buffer_end)]
                self.buffer_start += len(batch_instances)
                
                # 组织数据
                current_flows = self._organize_buffer_data(batch_instances)
            
            if len(current_flows) == 0:
                continue
            
            # 从current_flows生成batch，两个任务共享同一批流数据
            for batch_start_idx in range(0, len(current_flows), self.batch_size):
                batch_end_idx = min(batch_start_idx + self.batch_size, len(current_flows))
                batch_flows = current_flows[batch_start_idx:batch_end_idx]
                
                if len(batch_flows) == 0:
                    continue
                
                if self.proc_id == 0 and iter_count == 0:
                    print(f"[Rank {self.proc_id}] 开始生成第一个batch: {len(batch_flows)}个流...")
                
                # 为两个任务生成样本列表
                batch_src = []
                batch_seg = []
                batch_tgt_mlm = []
                batch_task_type = []
                
                # === 任务1: NSP_Indicating ===
                # 对每个流生成NSP_Indicating样本
                for flow in batch_flows:
                    src, seg, tgt_mlm = self._create_nsp_indicating_sample(flow)
                    batch_src.append(src)
                    batch_seg.append(seg)
                    batch_tgt_mlm.append(tgt_mlm)
                    batch_task_type.append('nsp_indicating')
                
                # === 任务2: MFR-Masking ===
                # 对每个流的4个数据包分别进行MFR-Masking处理
                # 按所有数据包（batch_size * 4）划分三种关系类型：30% h-h, 30% p-p, 40% h-p
                num_flows = len(batch_flows)
                total_packets = num_flows * 4  # 每个流有4个数据包
                num_hh = max(1, int(total_packets * self.hh_ratio))
                num_pp = max(1, int(total_packets * self.pp_ratio))
                num_hp = total_packets - num_hh - num_pp
                
                # 为所有数据包分配关系类型
                packet_list = [(flow_idx, pkt_idx) for flow_idx in range(num_flows) for pkt_idx in range(4)]
                random.shuffle(packet_list)
                hh_packets = set(packet_list[:num_hh])
                pp_packets = set(packet_list[num_hh:num_hh + num_pp])
                hp_packets = set(packet_list[num_hh + num_pp:])
                
                # 为每个流的每个数据包创建样本
                for flow_idx, flow in enumerate(batch_flows):
                    for packet_idx in range(4):
                        # 确定该数据包的关系类型
                        if (flow_idx, packet_idx) in hh_packets:
                            relation_type = 'hh'
                        elif (flow_idx, packet_idx) in pp_packets:
                            relation_type = 'pp'
                        else:
                            relation_type = 'hp'
                        
                        # 为每个数据包创建MFR-Masking样本
                        src, seg, tgt_mlm = self._create_mfr_masking_sample(flow, packet_idx, relation_type)
                        batch_src.append(src)
                        batch_seg.append(seg)
                        batch_tgt_mlm.append(tgt_mlm)
                        batch_task_type.append('mfr_masking')
                
                # 如果成功生成了样本，yield出去
                if len(batch_src) > 0:
                    # 首次batch生成提示（只打印一次）
                    if not self.first_batch_generated and self.proc_id == 0:
                        num_indicating = len(batch_flows)
                        num_mfr_masking = len(batch_flows) * 4  # 每个流4个数据包
                        print(f"[Rank {self.proc_id}] ✓ 首个batch生成成功: {len(batch_flows)}个流 × 2任务 = {len(batch_src)}个样本")
                        print(f"[Rank {self.proc_id}]   - NSP_Indicating: {num_indicating}个样本（每流1个）")
                        print(f"[Rank {self.proc_id}]   - MFR-Masking: {num_mfr_masking}个样本（每流4个数据包，每个数据包1个样本）")
                        print(f"[Rank {self.proc_id}] 数据加载完成，GPU训练开始！")
                        print("=" * 80)
                        self.first_batch_generated = True
                    
                    iter_count += 1
                    
                    yield (
                        torch.LongTensor(batch_src),
                        torch.LongTensor(batch_seg),
                        torch.LongTensor(batch_tgt_mlm),
                        batch_task_type
                    )


class MultiTaskTrainer:
    """多任务训练器"""
    def __init__(self, args):
        self.args = args
        # 从检查点恢复训练步数（如果有的话）
        checkpoint_step = getattr(args, 'checkpoint_step', 0)
        self.current_step = checkpoint_step + 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.output_model_path = args.output_model_path

        self.start_time = time.time()

        # 两个任务的统计信息
        # 对于MLM任务: num_samples表示样本数（用于计算平均loss），num_tokens表示被mask的token数，correct表示正确预测的token数
        self.task_stats = {
            'nsp_indicating': {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0},
            'mfr_masking': {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0}
        }

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size

        # 损失函数
        self.mlm_loss = nn.CrossEntropyLoss()
    
    def forward_propagation(self, batch, model):
        """
        前向传播 - 分批处理以节省显存

        Args:
            batch: (src, seg, tgt_mlm, task_types)
            model: MultiTaskModel

        Returns:
            loss: 加权损失
        """
        src, seg, tgt_mlm, task_types = batch
        batch_size = src.size(0)

        # 分批处理大小（减少显存占用）
        sub_batch_size = max(1, batch_size // 4)  # 将batch分成4个子batch处理

        # 初始化为None，后续累加
        total_loss = None

        # 分批处理
        for start_idx in range(0, batch_size, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, batch_size)

            src_sub = src[start_idx:end_idx]
            seg_sub = seg[start_idx:end_idx]
            tgt_mlm_sub = tgt_mlm[start_idx:end_idx]
            task_types_sub = task_types[start_idx:end_idx]
            sub_size = end_idx - start_idx

            # 通过base model处理子batch
            output = model.module.base_model(src_sub, seg_sub) if hasattr(model, 'module') else model.base_model(src_sub, seg_sub)

            # 按任务类型分组索引
            task_groups = {'nsp_indicating': [], 'mfr_masking': []}
            for i, task_type in enumerate(task_types_sub):
                task_groups[task_type].append(i)

            # 处理NSP_Indicating任务
            if len(task_groups['nsp_indicating']) > 0:
                indices = task_groups['nsp_indicating']
                indicating_output_layer = model.module.indicating_mlm_output if hasattr(model, 'module') else model.indicating_mlm_output
                mlm_logits = indicating_output_layer(output[indices])  # [n_samples, seq_length, vocab_size]
                tgt = tgt_mlm_sub[indices]

                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_flat = tgt.view(-1)
                mask = tgt_flat > 0

                if mask.sum() > 0:
                    loss = self.mlm_loss(mlm_logits_flat[mask], tgt_flat[mask])
                    pred = torch.argmax(mlm_logits_flat[mask], dim=-1)
                    correct = torch.sum(pred == tgt_flat[mask]).float()
                    num_masked_tokens = mask.sum().float()
                else:
                    loss = torch.tensor(0.0, device=src.device, requires_grad=True)
                    correct = torch.tensor(0.0, device=src.device)
                    num_masked_tokens = torch.tensor(1.0, device=src.device)  # 避免除零

                weighted_loss = loss * self.args.nsp_indicating_weight * len(indices) / batch_size
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

                # NSP_Indicating (MLM): loss按样本数统计，accuracy按token数统计
                self.task_stats['nsp_indicating']['loss'] += loss.item() * len(indices)
                self.task_stats['nsp_indicating']['correct'] += correct.item()
                self.task_stats['nsp_indicating']['num_samples'] += len(indices)
                self.task_stats['nsp_indicating']['num_tokens'] += num_masked_tokens.item()

            # 处理MFR-Masking任务
            if len(task_groups['mfr_masking']) > 0:
                indices = task_groups['mfr_masking']
                mfr_output_layer = model.module.mfr_mlm_output if hasattr(model, 'module') else model.mfr_mlm_output
                mlm_logits = mfr_output_layer(output[indices])  # [n_samples, seq_length, vocab_size]
                tgt = tgt_mlm_sub[indices]

                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_flat = tgt.view(-1)
                mask = tgt_flat > 0

                if mask.sum() > 0:
                    loss = self.mlm_loss(mlm_logits_flat[mask], tgt_flat[mask])
                    pred = torch.argmax(mlm_logits_flat[mask], dim=-1)
                    correct = torch.sum(pred == tgt_flat[mask]).float()
                    num_masked_tokens = mask.sum().float()
                else:
                    loss = torch.tensor(0.0, device=src.device, requires_grad=True)
                    correct = torch.tensor(0.0, device=src.device)
                    num_masked_tokens = torch.tensor(1.0, device=src.device)  # 避免除零

                weighted_loss = loss * self.args.mfr_masking_weight * len(indices) / batch_size
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

                # MFR-Masking (MLM): loss按样本数统计，accuracy按token数统计
                self.task_stats['mfr_masking']['loss'] += loss.item() * len(indices)
                self.task_stats['mfr_masking']['correct'] += correct.item()
                self.task_stats['mfr_masking']['num_samples'] += len(indices)
                self.task_stats['mfr_masking']['num_tokens'] += num_masked_tokens.item()

            # 清理中间变量以节省显存
            del output, src_sub, seg_sub, tgt_mlm_sub
            torch.cuda.empty_cache()

        # 如果没有任何任务样本，返回一个零loss张量（避免backward错误）
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=src.device, requires_grad=True)

        return total_loss / self.accumulation_steps
    
    def report_and_reset_stats(self):
        """报告和重置统计信息"""
        elapsed = time.time() - self.start_time
        
        print(f"| Step: {self.current_step:6d}/{self.total_steps} | Time: {elapsed:6.2f}s |")
        
        # 按任务顺序输出统计信息
        task_order = ['nsp_indicating', 'mfr_masking']
        for task_name in task_order:
            stats = self.task_stats[task_name]
            
            # MLM任务: loss按样本统计，accuracy按token统计
            if stats['num_samples'] > 0:
                avg_loss = stats['loss'] / stats['num_samples']
                if stats['num_tokens'] > 0:
                    accuracy = stats['correct'] / stats['num_tokens'] * 100  # 百分比
                else:
                    accuracy = 0.0  # 没有需要预测的token，Acc为0
                print(f"| {task_name:15s} | Loss: {avg_loss:6.4f} | Acc: {accuracy:6.2f}% | Samples: {int(stats['num_samples']):4d} |")
            else:
                print(f"| {task_name:15s} | No samples |")
        
        print("-" * 80)
        
        # 重置统计信息
        for task_name in self.task_stats:
            self.task_stats[task_name] = {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0}
        
        self.start_time = time.time()
    
    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        """训练循环"""
        model.train()

        # 保存optimizer和scheduler以便在保存checkpoint时使用
        self.optimizer = optimizer
        self.scheduler = scheduler

        if rank == 0:
            print(f"[Rank {rank}] 开始训练循环，准备迭代数据加载器...")
        
        batch_count = 0
        for batch in loader:
            if self.current_step > self.total_steps:
                break
            
            # 首次batch打印信息
            if batch_count == 0 and rank == 0:
                print(f"[Rank {rank}] ✓ 成功获取第一个batch！开始GPU训练...")
            
            # 转移到GPU
            if gpu_id is not None:
                batch = list(batch)
                for i in range(len(batch) - 1):  # 最后一个是task_types（字符串列表），不需要转移
                    if torch.is_tensor(batch[i]):
                        batch[i] = batch[i].cuda(gpu_id, non_blocking=True)
            
            # 前向传播
            loss = self.forward_propagation(batch, model)

            # 反向传播
            loss.backward()

            # 梯度积累
            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # 清理显存
            if gpu_id is not None:
                torch.cuda.empty_cache()

            # 报告统计信息
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
            
            # 保存检查点
            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                checkpoint_path = self.output_model_path + "-" + str(self.current_step)
                # 保存完整的checkpoint（包含model, optimizer, scheduler）
                checkpoint = {
                    'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'step': self.current_step,
                }
                torch.save(checkpoint, checkpoint_path)
                if rank == 0:
                    print(f"✓ Checkpoint saved at step {self.current_step}: {checkpoint_path}")
            
            self.current_step += 1
            batch_count += 1


def train_and_validate_multi_task(args):
    """
    多任务预训练的训练函数
    """
    import time
    
    set_seed(args.seed)
    
    # 加载词汇表
    print("=" * 80)
    print("Loading vocabulary...")
    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "You need to install SentencePiece: "
                "pip install sentencepiece"
            )
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i
                      in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
    
    print(f"Vocabulary size: {len(args.vocab)}")
    
    # 构建base model（新架构：embedding + 4层PacketAttention + 池化 + 8层FlowAttention）
    print("Building base model (embedding + 4-layer PacketAttention + Pooling + 8-layer FlowAttention)...")
    embedding = str2embedding[args.embedding](args, len(args.vocab))

    # 构建新的注意力堆栈
    print("Building attention stacks...")
    packet_size = getattr(args, 'tokens_per_packet', 80)
    header_size = getattr(args, 'header_size', 40)
    payload_size = packet_size - header_size
    window_size = getattr(args, 'window_size', 20)

    # 4层PacketAttention堆栈
    packet_attention_stack = PacketAttentionStack(
        num_layers=4,
        hidden_size=args.hidden_size,
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        header_size=header_size,
        packet_size=packet_size,
        window_size=window_size,
        dropout=getattr(args, 'dropout', 0.1)
    )

    # 池化层
    pooling_layer = PoolingLayer(
        header_size=header_size,
        payload_size=payload_size
    )

    # 8层FlowAttention堆栈
    flow_attention_stack = FlowAttentionStack(
        num_layers=8,
        hidden_size=args.hidden_size,
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        header_size=header_size,
        payload_size=payload_size,
        dropout=getattr(args, 'dropout', 0.1)
    )

    # 构建base model，使用新架构
    base_model = BaseModel(
        embedding,
        packet_attention_stack=packet_attention_stack,
        pooling_layer=pooling_layer,
        flow_attention_stack=flow_attention_stack,
        packet_size=packet_size,
        header_size=header_size
    )

    # 构建多任务模型
    print("Building multi-task model...")
    model = MultiTaskModel(base_model, args, len(args.vocab))

    # 加载或初始化参数
    optimizer_state_dict = None
    scheduler_state_dict = None
    if args.pretrained_model_path is not None:
        print(f"Loading pretrained model from {args.pretrained_model_path}...")
        # 加载预训练的embedding和encoder参数
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')

        # 检查是否是完整的checkpoint（包含model, optimizer, scheduler）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading full checkpoint (model + optimizer + scheduler)...")
            pretrained_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
            scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
        else:
            # 向后兼容：直接是模型参数
            pretrained_dict = checkpoint

        model_dict = model.state_dict()

        # 过滤出模型中存在的参数（支持完整模型和base_model部分加载）
        pretrained_dict_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict_filtered[k] = v
            elif k.startswith('base_model.') and k in model_dict:
                pretrained_dict_filtered[k] = v
            elif 'base_model.' + k in model_dict:
                pretrained_dict_filtered['base_model.' + k] = v

        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict_filtered)} parameters from pretrained model")
    else:
        print("Initializing model with normal distribution...")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    
    print("=" * 80)

    # 进行分布式训练或单机训练
    if args.dist_train:
        # 多进程分布式模式
        print("Starting distributed training...")
        mp.spawn(
            worker_multi_task,
            nprocs=args.ranks_num,
            args=(args.gpu_ranks, args, model, optimizer_state_dict, scheduler_state_dict),
            daemon=False
        )
    elif args.single_gpu:
        # 单GPU模式
        print(f"Starting training on GPU {args.gpu_id}...")
        worker_multi_task(args.gpu_id, None, args, model, optimizer_state_dict, scheduler_state_dict)
    else:
        # CPU模式
        print("Starting training on CPU...")
        worker_multi_task(None, None, args, model, optimizer_state_dict, scheduler_state_dict)


def worker_multi_task(gpu_id, gpu_ranks, args, model, optimizer_state_dict=None, scheduler_state_dict=None):
    """
    工作进程函数
    """
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    rank = 0
    if args.dist_train:
        rank = gpu_ranks.index(gpu_id)
        dist.init_process_group(
            backend=args.backend,
            init_method=args.master_ip,
            world_size=args.world_size,
            rank=rank
        )
        model.cuda(gpu_id)
        # 使用find_unused_parameters=False，因为我们会确保所有参数在每次迭代中都被使用
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=False)
        proc_id = rank
        proc_num = args.world_size
    elif args.single_gpu:
        model.cuda(gpu_id)
        proc_id = 0
        proc_num = 1
    else:
        proc_id = 0
        proc_num = 1

    # 创建数据加载器
    loader = MultiTaskDataset(
        args,
        args.dataset_path,
        args.batch_size,
        proc_id,
        proc_num,
        shuffle=True
    )

    # 构建优化器
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adamw":
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)

    if args.scheduler == "warmup":
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup, args.total_steps)

    # 恢复优化器和调度器状态（如果存在）
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            if proc_id == 0:
                print(f"✓ Restored optimizer state from checkpoint")
        except Exception as e:
            if proc_id == 0:
                print(f"⚠ Warning: Could not restore optimizer state: {e}")

    if scheduler_state_dict is not None:
        try:
            scheduler.load_state_dict(scheduler_state_dict)
            if proc_id == 0:
                print(f"✓ Restored scheduler state from checkpoint")
        except Exception as e:
            if proc_id == 0:
                print(f"⚠ Warning: Could not restore scheduler state: {e}")
    
    # 创建训练器
    trainer = MultiTaskTrainer(args)

    # 执行训练
    if proc_id == 0:
        print("=" * 80)
        print("Starting multi-task training...")
        if hasattr(args, 'checkpoint_step') and args.checkpoint_step > 0:
            print(f"Resuming from checkpoint: step {args.checkpoint_step}")
            print(f"Total steps: {args.total_steps}")
            print(f"Remaining steps: {args.total_steps - args.checkpoint_step}")
        print("=" * 80)
    
    trainer.train(args, gpu_id, rank, loader, model, optimizer, scheduler)

    # 保存最终模型
    if proc_id == 0:
        final_checkpoint = {
            'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'step': trainer.current_step,
        }
        torch.save(final_checkpoint, args.output_model_path)
        print("=" * 80)
        print(f"Training completed! Model saved to {args.output_model_path}")
        print("=" * 80)
