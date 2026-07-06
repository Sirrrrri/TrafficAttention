"""
基于5个数据包的下游任务微调脚本
与预训练任务对齐，使用5个数据包的拼接作为输入

预训练与微调的差别:
- 预训练: 使用NSP Indicating预训练任务训练模型
- 微调: 给定5个数据包，预测流量类别 (Traffic Classification) -> K分类

输入格式:
- seq_length: 410 (保持与预训练显式包边界协议一致)
- 结构: [CLS] + packet1_tokens(80) + [SEP] + packet2_tokens(80) + [SEP] + packet3_tokens(80) + [SEP] + packet4_tokens(80) + [SEP] + packet5_tokens(80) + [SEP]
- 每个数据包80个tokens，总共: 1 + 5 * (80 + 1) = 406，padding到410 tokens
- seg: 所有位置都使用 segment 1
  格式: [CLS]=1, packet1=1, [SEP]=1, ..., packet5=1, [SEP]=1
- 标签: 流量类别ID (如 0=Idle, 1=Backdoor, 等)

数据来源:
- 数据集目录: /root/datasets_new/fine-tuning数据集/
- 预训练模型: /root/models/pre-trained_model_two_tasks/pre-trained_model_two_tasks.bin-5000
"""
import random
import argparse
import pickle
import torch    
import torch.nn as nn
import torch.nn.functional as F
import math
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np

FLOW_PACKET_COUNT = 5


def packet_start_positions(packet_count=FLOW_PACKET_COUNT, packet_size=80):
    return [1 + i * (packet_size + 1) for i in range(packet_count)]


def sep_positions(packet_count=FLOW_PACKET_COUNT, packet_size=80):
    return [start + packet_size for start in packet_start_positions(packet_count, packet_size)]


def required_flow_seq_length(packet_count=FLOW_PACKET_COUNT, packet_size=80):
    return 1 + packet_count * (packet_size + 1)


def build_flow_sequence(packet_ids_list, cls_id, sep_id, pad_id, seq_length, packet_count=FLOW_PACKET_COUNT, packet_size=80):
    src = [cls_id]
    seg = [1]

    for packet_idx in range(packet_count):
        packet_ids = packet_ids_list[packet_idx] if packet_idx < len(packet_ids_list) else []
        packet_ids = packet_ids[:packet_size] + [pad_id] * max(0, packet_size - len(packet_ids))
        src.extend(packet_ids[:packet_size])
        seg.extend([1] * packet_size)
        src.append(sep_id)
        seg.append(1)

    if len(src) > seq_length:
        src = src[:seq_length]
        seg = seg[:seq_length]
    while len(src) < seq_length:
        src.append(pad_id)
        seg.append(0)

    return src, seg


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
            output = group[0](output)
            output = group[1](output)
            output = group[2](output)
        return output


class FlowAttention(nn.Module):
    """
    流级注意力机制模块
    处理流的前5个数据包，使用滑动窗口注意力机制（包括自己和相邻包）

    工作流程:
    1. 接收包级注意力的输出（每个数据包: header 40 tokens + payload 40 tokens）
    2. 按行池化: 将每个数据包payload的40 tokens按每10 tokens平均池化为4 tokens
    3. 数据重组: 前面拼接5个数据包的header，后面拼接5个数据包池化后的payload
    4. 滑动窗口注意力（包括自己）:
       - 包1: 窗口=[0,1]，attend到包1和包2
       - 包2: 窗口=[0,1,2]，attend到包1、包2和包3
       - 包3: 窗口=[1,2,3]，attend到包2、包3和包4
       - 包4: 窗口=[2,3,4]，attend到包3、包4和包5
    5. 分别对header和payload应用注意力（h查h, p查p）
    """
    def __init__(self, hidden_size, num_attention_heads=8, header_size=40, payload_size=40, dropout=0.1):
        super(FlowAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.header_size = header_size
        self.payload_size = payload_size
        self.pooled_payload_size = 4  # 40 tokens -> 4 tokens (每10个tokens池化为1个)
        
        # Header注意力
        self.header_query = nn.Linear(hidden_size, hidden_size)
        self.header_key = nn.Linear(hidden_size, hidden_size)
        self.header_value = nn.Linear(hidden_size, hidden_size)
        self.header_output = nn.Linear(hidden_size, hidden_size)
        
        # Payload注意力
        self.payload_query = nn.Linear(hidden_size, hidden_size)
        self.payload_key = nn.Linear(hidden_size, hidden_size)
        self.payload_value = nn.Linear(hidden_size, hidden_size)
        self.payload_output = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Norm
        self.header_norm = nn.LayerNorm(hidden_size)
        self.payload_norm = nn.LayerNorm(hidden_size)
        
    def transpose_for_scores(self, x):
        """将tensor转换为多头注意力的形状"""
        # x: [batch_size, seq_len, hidden_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # 返回: [batch_size, num_heads, seq_len, head_size]
        return x.permute(0, 2, 1, 3)
    
    def pool_payload(self, payload_hidden):
        """
        将payload的40 tokens按每10个tokens进行平均池化
        Input: [batch_size, 40, hidden_size]
        Output: [batch_size, 4, hidden_size]
        """
        batch_size, seq_len, hidden_size = payload_hidden.size()
        # 重塑为 [batch_size, 4, 10, hidden_size]
        payload_reshaped = payload_hidden.view(batch_size, 4, 10, hidden_size)
        # 按第3维（10个tokens）取平均
        pooled_payload = torch.mean(payload_reshaped, dim=2)  # [batch_size, 4, hidden_size]
        return pooled_payload
    
    def unpool_payload(self, pooled_payload):
        """
        将池化后的payload恢复到原始尺寸（用于与原始输入进行残差连接）
        Input: [batch_size, 4, hidden_size]
        Output: [batch_size, 40, hidden_size]
        """
        batch_size, pooled_len, hidden_size = pooled_payload.size()
        # 每个池化的token复制10次
        unpooled = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)  # [batch_size, 4, 10, hidden_size]
        unpooled = unpooled.view(batch_size, 40, hidden_size)  # [batch_size, 40, hidden_size]
        return unpooled
    
    def apply_attention(self, query_hidden, key_value_hiddens, is_header=True):
        """
        应用注意力机制
        Args:
            query_hidden: [batch_size, seq_len_q, hidden_size] - 查询的hidden states
            key_value_hiddens: list of tensors - 用作key和value的hidden states列表
            is_header: bool - 是否为header注意力
        Returns:
            output: [batch_size, seq_len_q, hidden_size]
            probs: [batch_size, num_heads, seq_len_q, total_seq_len_kv]
        """
        # 选择对应的query/key/value投影层
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

        # 计算Query
        query = self.transpose_for_scores(query_layer(query_hidden))
        # [batch_size, num_heads, seq_len_q, head_size]

        # 计算Key和Value（拼接所有key_value_hiddens）
        all_keys = []
        all_values = []
        for kv_hidden in key_value_hiddens:
            key = self.transpose_for_scores(key_layer(kv_hidden))
            value = self.transpose_for_scores(value_layer(kv_hidden))
            all_keys.append(key)
            all_values.append(value)

        # 拼接所有keys和values
        keys = torch.cat(all_keys, dim=2)  # [batch_size, num_heads, total_seq_len_kv, head_size]
        values = torch.cat(all_values, dim=2)  # [batch_size, num_heads, total_seq_len_kv, head_size]

        # 计算注意力分数
        attention_scores = torch.matmul(query, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 应用注意力
        context = torch.matmul(attention_probs, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        # 输出投影
        output = output_layer(context)
        output = self.dropout(output)

        return output, attention_probs

    def forward(self, all_packet_hiddens):
        """
        对整个流（5个数据包）应用流级注意力
        查询窗口内的所有数据包MFR，包括自己和相邻数据包

        窗口配置:
        - 包1: 窗口=[0,1]，attend到包1和包2
        - 包2: 窗口=[0,1,2]，attend到包1、包2和包3
        - 包3: 窗口=[1,2,3]，attend到包2、包3和包4
        - 包4: 窗口=[2,3,4]，attend到包3、包4和包5

        Args:
            all_packet_hiddens: list of 4 tensors, each [batch_size, 80, hidden_size]
                               每个tensor包含一个数据包的hidden states
        Returns:
            output_headers: list of 4 tensors, each [batch_size, 40, hidden_size]
            output_payloads: list of 4 tensors, each [batch_size, 40, hidden_size]
            probs: dictionary of attention probabilities
        """
        # 分离所有数据包的header和payload
        all_headers = []
        all_payloads = []
        all_pooled_payloads = []

        for packet_hidden in all_packet_hiddens:
            header = packet_hidden[:, :self.header_size, :]  # [batch_size, 40, hidden_size]
            payload = packet_hidden[:, self.header_size:, :]  # [batch_size, 40, hidden_size]
            pooled_payload = self.pool_payload(payload)  # [batch_size, 4, hidden_size]

            all_headers.append(header)
            all_payloads.append(payload)
            all_pooled_payloads.append(pooled_payload)

        # 应用滑动窗口注意力（包括自己和相邻包）
        output_headers = []
        output_payloads = []
        header_probs = []
        payload_probs = []

        packet_count = len(all_packet_hiddens)
        for i in range(packet_count):
            window_start = max(0, i - 1)
            window_end = min(packet_count, i + 2)
            header_window = all_headers[window_start:window_end]
            payload_window = all_pooled_payloads[window_start:window_end]

            header_out, hp = self.apply_attention(all_headers[i], header_window, is_header=True)
            payload_out, pp = self.apply_attention(all_pooled_payloads[i], payload_window, is_header=False)

            # 残差连接 + Layer Norm (header)
            header_final = self.header_norm(all_headers[i] + header_out)

            # 将池化后的payload恢复到原始尺寸，然后应用残差连接 + Layer Norm
            payload_out_unpooled = self.unpool_payload(payload_out)
            payload_final = self.payload_norm(all_payloads[i] + payload_out_unpooled)

            output_headers.append(header_final)
            output_payloads.append(payload_final)
            header_probs.append(hp)
            payload_probs.append(pp)

        probs = {
            "header": header_probs,
            "payload": payload_probs
        }

        return output_headers, output_payloads, probs


class TwoLevelAttentionEncoder(nn.Module):
    """
    两级注意力编码器（新架构）
    第1-4层: 包级注意力堆栈（4层）
    池化层: 对payload进行池化
    第5-12层: 流级注意力堆栈（8层）
    """
    def __init__(self, args):
        super(TwoLevelAttentionEncoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_attention_heads = getattr(args, 'num_attention_heads', 8)
        self.packet_size = getattr(args, 'packet_size', 80)
        self.header_size = getattr(args, 'header_size', 40)
        self.payload_size = self.packet_size - self.header_size
        self.window_size = getattr(args, 'window_size', 20)
        self.packet_count = FLOW_PACKET_COUNT

        # 第1-4层: 包级注意力堆栈（使用PacketAttentionStack）
        self.packet_attention_stack = PacketAttentionStack(
            num_layers=4,
            hidden_size=args.hidden_size,
            num_attention_heads=self.num_attention_heads,
            header_size=self.header_size,
            packet_size=self.packet_size,
            window_size=self.window_size,
            dropout=getattr(args, 'dropout', 0.1)
        )

        # 第5-12层: 流级注意力堆栈（8层）
        self.flow_attention_stack = nn.ModuleList([
            FlowAttention(
                hidden_size=args.hidden_size,
                num_attention_heads=self.num_attention_heads,
                header_size=self.header_size,
                payload_size=self.payload_size,
                dropout=getattr(args, 'dropout', 0.1)
            )
            for _ in range(8)
        ])

        # 存储attention概率用于可视化
        self.packet_attention_probs = None
        self.flow_attention_probs = None

    def forward(self, hidden_states, seg):
        """
        Args:
            hidden_states: [batch_size, seq_length, hidden_size] - 来自embedding层的输入
            seg: [batch_size, seq_length] - segment标记

        Returns:
            output: [batch_size, seq_length, hidden_size]
            attention_probs: tuple of flow attention probabilities
        """
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        # 提取5个数据包的表示，协议：[CLS] + packet(80) + [SEP] * 5 + padding
        all_packet_hiddens = []
        packet_starts = packet_start_positions(self.packet_count, self.packet_size)

        for i in range(self.packet_count):
            start_pos = packet_starts[i]
            end_pos = min(start_pos + self.packet_size, seq_len)

            packet_hidden = hidden_states[:, start_pos:end_pos, :]

            actual_len = end_pos - start_pos
            if actual_len < self.packet_size:
                pad_len = self.packet_size - actual_len
                padding = torch.zeros(
                    packet_hidden.size(0), pad_len, packet_hidden.size(2),
                    dtype=packet_hidden.dtype, device=packet_hidden.device
                )
                packet_hidden = torch.cat([packet_hidden, padding], dim=1)

            all_packet_hiddens.append(packet_hidden)

        # 第1-4层: 应用4层包级注意力堆栈
        packet_outputs = []
        for i in range(self.packet_count):
            packet_output = self.packet_attention_stack(all_packet_hiddens[i])
            packet_outputs.append(packet_output)

        all_packet_hiddens = packet_outputs

        # 池化层：对每个packet的payload进行mean pooling
        pooled_packets = []
        for packet_hidden in all_packet_hiddens:
            header = packet_hidden[:, :self.header_size, :]  # [batch_size, 40, hidden_size]
            payload = packet_hidden[:, self.header_size:, :]  # [batch_size, 40, hidden_size]

            # 对payload进行池化：40 tokens -> 4 tokens
            batch_size_p, seq_len_p, hidden_size_p = payload.size()
            payload_reshaped = payload.view(batch_size_p, 4, 10, hidden_size_p)
            pooled_payload = torch.mean(payload_reshaped, dim=2)  # [batch_size, 4, hidden_size]

            # 恢复到原始形状用于后续处理
            pooled_payload_expanded = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)
            pooled_payload_expanded = pooled_payload_expanded.view(batch_size_p, 40, hidden_size_p)

            pooled_packet = torch.cat([header, pooled_payload_expanded], dim=1)
            pooled_packets.append(pooled_packet)

        # 第5-12层: 应用8层流级注意力堆栈
        flow_probs_list = []
        for layer_idx, flow_attention in enumerate(self.flow_attention_stack):
            output_headers, output_payloads, f_probs = flow_attention(pooled_packets)
            flow_probs_list.append(f_probs)

            # 重新组合为packet格式用于下一层
            pooled_packets = []
            for i in range(self.packet_count):
                packet_output = torch.cat([output_headers[i], output_payloads[i]], dim=1)
                pooled_packets.append(packet_output)

        self.flow_attention_probs = flow_probs_list

        # 重新组合所有数据包
        packet_outputs = pooled_packets

        # 保留[CLS] token和每个packet之后的[SEP] token
        cls_output = hidden_states[:, 0:1, :]
        sep_outputs = []
        for sep_pos in sep_positions(self.packet_count, self.packet_size):
            if sep_pos < seq_len:
                sep_outputs.append(hidden_states[:, sep_pos:sep_pos + 1, :])
            else:
                sep_outputs.append(torch.zeros(batch_size, 1, hidden_states.size(2), dtype=hidden_states.dtype, device=hidden_states.device))

        output_parts = [cls_output]
        for packet_output, sep_output in zip(packet_outputs, sep_outputs):
            output_parts.extend([packet_output, sep_output])

        output = torch.cat(output_parts, dim=1)

        # 返回输出和flow attention概率（作为encoder attention）
        return output, tuple(flow_probs_list)


class Classifier(nn.Module):
    """
    基于5个数据包结构的分类模型（4层包级 + 池化 + 8层流级）
    与NSP Indicating预训练模型结构对齐
    输入: [CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80) + [SEP] + packet5(80) + [SEP]

    架构:
    1. Embedding层
    2. 第1-4层: 包级注意力堆栈（3个子注意力机制串行处理）
       - SubAttention1: header内部全局注意力
       - SubAttention2: payload作q，header作k&v的交叉注意力
       - SubAttention3: payload内部窗口注意力
    3. 池化层: 对payload进行mean pooling
    4. 第5-12层: 流级注意力堆栈（8层）
    5. 分类头
    """
    def __init__(self, args):
        super(Classifier, self).__init__()
        # 创建Embedding Layer
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        # 创建两级注意力 + 标准编码器
        self.encoder = TwoLevelAttentionEncoder(args)
        # 分类任务的类别数量
        self.labels_num = args.labels_num
        # 池化策略
        self.pooling = args.pooling
        # 知识蒸馏标志
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha

        # 分类头 - 保持与预训练时相同的结构
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        前向传播（4层包级 + 池化 + 8层流级）

        包级注意力处理流程（每层）:
        1. SubAttention1: header内部全局注意力
        2. SubAttention2: payload作q，header作k&v的交叉注意力
        3. SubAttention3: payload内部窗口注意力

        Args:
            src: [batch_size x seq_length] - 输入序列
                 格式: [CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80) + [SEP] + packet5(80) + [SEP]
            tgt: [batch_size] - 真实标签（流量类别ID）
            seg: [batch_size x seq_length] - segment标记（所有位置都使用segment 1）
            soft_tgt: [batch_size x labels_num] - 软标签（可选）
        """
        batch_size = src.size(0)

        # Embedding
        emb = self.embedding(src, seg)
        # 编码器: 第1-4层包级 + 池化 + 第5-12层流级
        output, encoder_probs = self.encoder(emb, seg)

        # 池化 - 使用CLS token或其他池化方式
        if self.pooling == "mean":
            pooled_output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            pooled_output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            pooled_output = output[:, -1, :]
        else:
            # 默认: 使用第一个token (CLS)
            pooled_output = output[:, 0, :]

        # 分类头
        classifier_output = torch.tanh(self.output_layer_1(pooled_output))
        logits = self.output_layer_2(classifier_output)

        # 获取attention概率
        packet_attention_probs = self.encoder.packet_attention_probs if hasattr(self.encoder, 'packet_attention_probs') else None
        flow_attention_probs = self.encoder.flow_attention_probs if hasattr(self.encoder, 'flow_attention_probs') else None

        all_attention_probs = {
            "encoder": encoder_probs,
            "packet": packet_attention_probs,
            "flow": flow_attention_probs
        }

        # 损失计算
        if tgt is not None:
            # 知识蒸馏
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            # 标准交叉熵损失
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits, all_attention_probs
        # 推理模式
        else:
            return None, logits, all_attention_probs

    def get_attention_weights(self, src, seg):
        """
        推理模式：获取attention权重用于可视化
        返回格式适配BertViz

        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]

        Returns:
            attention_weights: dict包含encoder、packet、flow的attention权重
        """
        batch_size = src.size(0)

        # Embedding
        emb = self.embedding(src, seg)
        # 编码器: 第1-4层包级 + 池化 + 第5-12层流级
        output, encoder_probs = self.encoder(emb, seg)

        # 获取attention概率
        packet_attention_probs = self.encoder.packet_attention_probs if hasattr(self.encoder, 'packet_attention_probs') else None
        flow_attention_probs = self.encoder.flow_attention_probs if hasattr(self.encoder, 'flow_attention_probs') else None

        attention_weights = {
            "encoder": encoder_probs,
            "packet": packet_attention_probs,
            "flow": flow_attention_probs
        }

        return attention_weights


def count_labels_num(path):
    """统计标签数量"""
    labels_set = set()
    
    # 检查文件扩展名
    if path.endswith('.pt'):
        # 读取.pt文件
        with open(path, 'rb') as f:
            try:
                while True:
                    flow_data = pickle.load(f)
                    if len(flow_data) == 5:
                        flow_id, flow_mfr_tokens, category_name, dataset_name, label_id = flow_data
                        labels_set.add(label_id)
            except EOFError:
                pass
    else:
        # 向后兼容：TSV格式
        columns = {}
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    for i, column_name in enumerate(line.strip().split("\t")):
                        columns[column_name] = i
                    continue
                line = line.strip().split("\t")
                label = int(line[columns["label"]])
                labels_set.add(label)
    
    return max(labels_set) + 1 if labels_set else 0


def load_or_initialize_parameters(args, model):
    """加载预训练参数或随机初始化"""
    if args.pretrained_model_path is not None:
        # 从预训练模型加载
        print(f"Loading pretrained model from {args.pretrained_model_path}")
        
        # 检查CUDA是否可用（只使用GPU）
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ 错误: CUDA不可用！此脚本仅支持GPU训练。\n"
                "   请确保：\n"
                "   1. 已安装CUDA和cuDNN\n"
                "   2. 已安装支持CUDA的PyTorch版本\n"
                "   3. 使用正确的conda环境（如：conda activate Mine）\n"
                f"   当前PyTorch版本: {torch.__version__}\n"
                f"   CUDA可用: {torch.cuda.is_available()}"
            )
        
        # 将其他GPU的模型映射到cuda:0
        map_location = {'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}
        
        model.load_state_dict(
            torch.load(args.pretrained_model_path, map_location=map_location),
            strict=False
        )
        print("✓ Model loaded successfully")
    else:
        # 随机初始化
        print("Initializing model with random parameters")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    """构建优化器和学习率调度器"""
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    """按batch生成数据"""
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    """
    读取数据集并转换为张量格式
    
    **支持.pt文件格式**: 从.pt文件中读取流数据，使用完整的5个数据包
    .pt文件格式: (flow_id, flow_mfr_tokens[400], category_name, dataset_name, label_id)
    
    处理流程:
    1. 读取.pt文件中的流数据（前5个数据包，每个80 tokens，共400 tokens）
    2. 将400个token拆分成5个数据包（每个80个token）
    3. 构建输入序列：使用全部5个数据包
    4. 转换为模型输入格式: [CLS] + packet1 + [SEP] + packet2 + [SEP] + packet3 + [SEP] + packet4 + [SEP] + packet5 + [SEP]
    
    输入结构: 
    - src: [CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80) + [SEP] + packet5(80) + [SEP]
    - seg: 所有有效位置均为1，padding为0
    总长度: 1 + 5 * (80 + 1) = 406，padding到seq_length（默认410）
    """
    dataset = []
    
    # 检查文件扩展名
    if path.endswith('.pt'):
        # 读取.pt文件（pickle格式）
        with open(path, 'rb') as f:
            flows = []
            try:
                while True:
                    flow_data = pickle.load(f)
                    flows.append(flow_data)
            except EOFError:
                pass
        
        # 处理每个流，生成数据包对
        for flow_data in flows:
            if len(flow_data) != 5:
                continue
            
            flow_id, flow_mfr_tokens, category_name, dataset_name, label_id = flow_data
            
            expected_len = FLOW_PACKET_COUNT * args.packet_size
            if not isinstance(flow_mfr_tokens, list):
                continue

            if len(flow_mfr_tokens) < expected_len:
                flow_mfr_tokens = flow_mfr_tokens + [0] * (expected_len - len(flow_mfr_tokens))
            elif len(flow_mfr_tokens) > expected_len:
                flow_mfr_tokens = flow_mfr_tokens[:expected_len]

            packets = []
            for packet_idx in range(FLOW_PACKET_COUNT):
                start_idx = packet_idx * args.packet_size
                end_idx = start_idx + args.packet_size
                packet_tokens = flow_mfr_tokens[start_idx:end_idx]
                if len(packet_tokens) == args.packet_size:
                    packets.append(packet_tokens)

            if len(packets) != FLOW_PACKET_COUNT:
                continue
            
            # 获取特殊token IDs
            cls_id = args.tokenizer.vocab.get(CLS_TOKEN, 2)
            sep_id = args.tokenizer.vocab.get(SEP_TOKEN, 3)
            unk_id = args.tokenizer.vocab.get('[UNK]', 1)
            
            # 转换5个数据包的tokens为IDs
            all_packet_ids = []
            for packet_tokens in packets:
                packet_ids = []
                for hex_token in packet_tokens:
                    token_id = args.tokenizer.vocab.get(str(hex_token), unk_id)
                    packet_ids.append(token_id)
                all_packet_ids.append(packet_ids)
            
            src, seg = build_flow_sequence(
                all_packet_ids, cls_id, sep_id, 0, args.seq_length,
                packet_count=FLOW_PACKET_COUNT, packet_size=args.packet_size
            )
            
            # 添加到数据集
            dataset.append((src, label_id, seg))
    
    else:
        # 向后兼容：支持TSV格式
        columns = {}
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    # 解析header
                    for i, column_name in enumerate(line.strip().split("\t")):
                        columns[column_name] = i
                    continue
                
                line = line[:-1].split("\t")
                tgt = int(line[columns["label"]])
                
                # 处理软标签（如果存在）
                if args.soft_targets and "logits" in columns.keys():
                    soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
                
                # 处理5个数据包格式（TSV格式向后兼容，但主要使用.pt格式）
                if "text_b" not in columns:  
                    # 单数据包格式（向后兼容）
                    text_a = line[columns["text_a"]]
                    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                    seg = [1] * len(src)
                else:  
                    text_a = line[columns["text_a"]]
                    text_b = line[columns["text_b"]]

                    packet_fields = [f"text_{suffix}" for suffix in ["a", "b", "c", "d", "e"]]
                    if all(field in columns for field in packet_fields):
                        packet_ids_list = []
                        for field in packet_fields:
                            tokens = args.tokenizer.tokenize(line[columns[field]])
                            token_ids = args.tokenizer.convert_tokens_to_ids(tokens)
                            packet_ids_list.append(token_ids)
                        src, seg = build_flow_sequence(
                            packet_ids_list,
                            args.tokenizer.vocab.get(CLS_TOKEN, 2),
                            args.tokenizer.vocab.get(SEP_TOKEN, 3),
                            0,
                            args.seq_length,
                            packet_count=FLOW_PACKET_COUNT,
                            packet_size=args.packet_size
                        )
                    else:
                        # 旧格式：只有2个数据包（向后兼容）
                        src_a = args.tokenizer.convert_tokens_to_ids(
                            [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                        )
                        src_b = args.tokenizer.convert_tokens_to_ids(
                            args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                        )
                        src = src_a + src_b
                        # 所有数据包都使用segment 1
                        seg = [1] * (len(src_a) + len(src_b))

                # 截断到seq_length
                if len(src) > args.seq_length:
                    src = src[: args.seq_length]
                    seg = seg[: args.seq_length]
                
                # 填充到seq_length
                while len(src) < args.seq_length:
                    src.append(0)
                    seg.append(0)
                
                # 添加到数据集
                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, soft_tgt))
                else:
                    dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    """训练一个batch"""
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)
    
    # 前向传播
    loss, _, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    
    # 多GPU处理
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    # 反向传播
    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # 优化器步进
    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False, save_attention_weights=False, dataset_name=None):
    """在验证/测试集上评估，返回loss和准确率"""
    import gc
    gc.collect()  # 清理CPU内存

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    # 评估时使用较小的batch_size以节省GPU内存
    batch_size = max(1, args.batch_size // 2)

    correct = 0
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    # 用于保存attention权重
    all_attention_weights = {
        "encoder": [],
        "packet": [],
        "flow": []
    }

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)

        with torch.no_grad():
            loss, logits, attention_probs = args.model(src_batch, tgt_batch, seg_batch)

            # 清理GPU缓存以防止内存碎片化
            torch.cuda.empty_cache()

            # 计算loss（多GPU时需要平均）
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item() * src_batch.size(0)
            total_samples += src_batch.size(0)

            # 保存attention权重
            if save_attention_weights:
                if attention_probs["encoder"] is not None:
                    encoder_attn = attention_probs["encoder"]
                    # encoder_attn是tuple，每个元素是一层的attention: [batch_size x heads x seq_len x seq_len]
                    if isinstance(encoder_attn, tuple):
                        # 只在第一个batch时初始化层级列表
                        if i == 0:
                            all_attention_weights["encoder"] = [[] for _ in range(len(encoder_attn))]
                        # 为每一层追加该batch的attention权重
                        for layer_idx, layer_attn in enumerate(encoder_attn):
                            if isinstance(layer_attn, torch.Tensor):
                                all_attention_weights["encoder"][layer_idx].append(layer_attn.cpu())
                    elif isinstance(encoder_attn, torch.Tensor):
                        all_attention_weights["encoder"].append(encoder_attn.cpu())

                if attention_probs["packet"] is not None:
                    packet_attn = attention_probs["packet"]
                    # packet_attention_probs是list of dicts，每个dict对应一个数据包
                    # 例如: [{"hh": tensor, "hp": tensor, "pp": tensor}, ...]
                    # 其中每个tensor的形状为 [batch, num_heads, seq_len, seq_len]
                    if isinstance(packet_attn, list) and len(packet_attn) > 0:
                        if isinstance(packet_attn[0], dict):
                            # 分别保存hh、hp、pp三种attention类型
                            packet_attn_hh = []
                            packet_attn_hp = []
                            packet_attn_pp = []
                            for attn_dict in packet_attn:
                                if "hh" in attn_dict and isinstance(attn_dict["hh"], torch.Tensor):
                                    packet_attn_hh.append(attn_dict["hh"])
                                if "hp" in attn_dict and isinstance(attn_dict["hp"], torch.Tensor):
                                    packet_attn_hp.append(attn_dict["hp"])
                                if "pp" in attn_dict and isinstance(attn_dict["pp"], torch.Tensor):
                                    packet_attn_pp.append(attn_dict["pp"])

                            # 堆叠所有数据包的attention: [num_packets, batch, num_heads, seq_len, seq_len]
                            if packet_attn_hh:
                                all_attention_weights["packet"].append({
                                    "hh": torch.stack(packet_attn_hh).cpu(),
                                    "hp": torch.stack(packet_attn_hp).cpu() if packet_attn_hp else None,
                                    "pp": torch.stack(packet_attn_pp).cpu() if packet_attn_pp else None
                                })
                        else:
                            # 原有逻辑：列表中是张量
                            packet_attn = torch.stack(packet_attn)
                            if isinstance(packet_attn, torch.Tensor):
                                all_attention_weights["packet"].append(packet_attn.cpu())

                if attention_probs["flow"] is not None:
                    flow_attn = attention_probs["flow"]
                    # flow_attn 是字典: {"header": [list of tensors], "payload": [list of tensors]}
                    if isinstance(flow_attn, dict):
                        # 分别处理 header 和 payload attention
                        flow_dict = {}
                        if "header" in flow_attn and flow_attn["header"]:
                            # header_probs 是 list of 4 tensors，每个 tensor 形状 [batch, num_heads, seq_len_q, seq_len_kv]
                            flow_dict["header"] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in flow_attn["header"]]
                        if "payload" in flow_attn and flow_attn["payload"]:
                            # payload_probs 是 list of 4 tensors
                            flow_dict["payload"] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in flow_attn["payload"]]
                        if flow_dict:
                            all_attention_weights["flow"].append(flow_dict)
                    elif isinstance(flow_attn, torch.Tensor):
                        all_attention_weights["flow"].append(flow_attn.cpu())

        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch

        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

        # 清理GPU内存中的中间变量
        del src_batch, tgt_batch, seg_batch, loss, logits, attention_probs, pred, gold
        torch.cuda.empty_cache()

        # 定期清理CPU内存（每10个batch清理一次）
        if (i + 1) % 10 == 0:
            gc.collect()

    # 计算平均loss和准确率
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0

    # 计算Macro-Precision, Macro-Recall, Macro-F1
    eps = 1e-9
    precisions = []
    recalls = []
    f1_scores = []
    correct_counts = []  # 每个类别预测正确的样本数量

    for i in range(confusion.size()[0]):
        # Precision = TP / (TP + FP)
        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        precision = tp / (tp + fp + eps)

        # Recall = TP / (TP + FN)
        fn = confusion[:, i].sum().item() - tp
        recall = tp / (tp + fn + eps)

        # F1 = 2 * (precision * recall) / (precision + recall)
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        correct_counts.append(tp)  # 保存每个类别预测正确的样本数量

    # Macro平均（所有类别的平均值）
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        import os
        output_dir = os.path.dirname(args.output_model_path)
        confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.txt")
        with open(confusion_matrix_path, 'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')
        print(f"Confusion matrix saved to: {confusion_matrix_path}")
        print("Report precision, recall, and f1:")
        for i in range(confusion.size()[0]):
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, precisions[i], recalls[i], f1_scores[i]))

    print("Loss: {:.4f}, Acc. (Correct/Total): {:.4f} ({}/{}) ".format(
        avg_loss, accuracy, correct, len(dataset)))
    print("Macro-Precision: {:.4f}, Macro-Recall: {:.4f}, Macro-F1: {:.4f}".format(
        macro_precision, macro_recall, macro_f1))

    # 输出每个类别预测正确的样本数量
    print("Correct predictions per class:")
    for i in range(confusion.size()[0]):
        total_true = confusion[:, i].sum().item()  # 该类别真实样本总数
        print("Label {}: {}/{} (correct/total)".format(i, correct_counts[i], total_true))

    # 保存attention权重到文件
    if save_attention_weights:
        import os
        output_dir = os.path.dirname(args.output_model_path)
        os.makedirs(output_dir, exist_ok=True)

        # 处理encoder attention：将所有batch的attention权重堆叠成单个张量
        if all_attention_weights["encoder"] and isinstance(all_attention_weights["encoder"][0], list):
            # 新格式：all_attention_weights["encoder"] = [[layer0_batch0, layer0_batch1, ...], [layer1_batch0, ...], ...]
            processed_encoder = []
            for layer_attns in all_attention_weights["encoder"]:
                if layer_attns:
                    # 只保留第一个样本以确保所有层大小一致
                    # 或者可以堆叠所有batch：torch.cat(layer_attns, dim=0)
                    # 这里选择只保留第一个样本
                    processed_encoder.append(layer_attns[0])
            all_attention_weights["encoder"] = processed_encoder

        # 处理packet attention - 分别保存hh、hp、pp
        processed_packet = []
        packet_hh_list = []
        packet_hp_list = []
        packet_pp_list = []

        if all_attention_weights["packet"]:
            for packet_item in all_attention_weights["packet"]:
                if isinstance(packet_item, dict):
                    # 新格式：dict包含hh、hp、pp
                    if "hh" in packet_item and packet_item["hh"] is not None:
                        packet_hh_list.append(packet_item["hh"])
                    if "hp" in packet_item and packet_item["hp"] is not None:
                        packet_hp_list.append(packet_item["hp"])
                    if "pp" in packet_item and packet_item["pp"] is not None:
                        packet_pp_list.append(packet_item["pp"])
                elif isinstance(packet_item, torch.Tensor):
                    # 旧格式：直接是张量
                    processed_packet.append(packet_item)

        # 处理flow attention
        if all_attention_weights["flow"] and isinstance(all_attention_weights["flow"][0], list):
            processed_flow = []
            for flow_attns in all_attention_weights["flow"]:
                if flow_attns:
                    processed_flow.append(flow_attns[0])
            all_attention_weights["flow"] = processed_flow

        # 保存标准encoder attention
        if all_attention_weights["encoder"]:
            # 验证Encoder attention的统计信息（只验证第一个，因为大小可能不一致）
            encoder_tensor = all_attention_weights["encoder"][0]
            print(f"\n📊 Encoder Attention Statistics (Layer 1, First batch):")
            print(f"  Shape: {encoder_tensor.shape}")
            print(f"  Min value: {encoder_tensor.min().item():.6f}")
            print(f"  Max value: {encoder_tensor.max().item():.6f}")
            print(f"  Mean value: {encoder_tensor.mean().item():.6f}")
            print(f"  Std value: {encoder_tensor.std().item():.6f}")
            print(f"  Contains negative: {(encoder_tensor < 0).any().item()}")
            print(f"  All values in [0, 1]: {((encoder_tensor >= 0) & (encoder_tensor <= 1)).all().item()}")

        encoder_attention_path = os.path.join(output_dir, "attention_encoder.pt")
        torch.save(all_attention_weights["encoder"], encoder_attention_path)
        print(f"\n✓ Encoder attention saved to: {encoder_attention_path}")
        print(f"  Encoder layers: {len(all_attention_weights['encoder'])}")
        if all_attention_weights["encoder"]:
            print(f"    First layer shape: {all_attention_weights['encoder'][0].shape}")

        # 保存flow attention
        if all_attention_weights["flow"]:
            # 验证Flow attention的统计信息（只验证第一个batch）
            if all_attention_weights["flow"] and isinstance(all_attention_weights["flow"][0], dict):
                first_flow = all_attention_weights["flow"][0]

                if "header" in first_flow and first_flow["header"]:
                    header_tensor = first_flow["header"][0]
                    print(f"\n📊 Flow Header Attention Statistics (First batch):")
                    print(f"  Shape: {header_tensor.shape}")
                    print(f"  Min value: {header_tensor.min().item():.6f}")
                    print(f"  Max value: {header_tensor.max().item():.6f}")
                    print(f"  Mean value: {header_tensor.mean().item():.6f}")
                    print(f"  Std value: {header_tensor.std().item():.6f}")
                    print(f"  Contains negative: {(header_tensor < 0).any().item()}")
                    print(f"  All values in [0, 1]: {((header_tensor >= 0) & (header_tensor <= 1)).all().item()}")

                if "payload" in first_flow and first_flow["payload"]:
                    payload_tensor = first_flow["payload"][0]
                    print(f"\n📊 Flow Payload Attention Statistics (First batch):")
                    print(f"  Shape: {payload_tensor.shape}")
                    print(f"  Min value: {payload_tensor.min().item():.6f}")
                    print(f"  Max value: {payload_tensor.max().item():.6f}")
                    print(f"  Mean value: {payload_tensor.mean().item():.6f}")
                    print(f"  Std value: {payload_tensor.std().item():.6f}")
                    print(f"  Contains negative: {(payload_tensor < 0).any().item()}")
                    print(f"  All values in [0, 1]: {((payload_tensor >= 0) & (payload_tensor <= 1)).all().item()}")

            flow_attention_path = os.path.join(output_dir, "attention_flow.pt")
            torch.save(all_attention_weights["flow"], flow_attention_path)
            print(f"\n✓ Flow attention saved to: {flow_attention_path}")

            # 打印详细信息
            if all_attention_weights["flow"] and isinstance(all_attention_weights["flow"][0], dict):
                first_flow = all_attention_weights["flow"][0]
                if "header" in first_flow:
                    print(f"  Header attention: {len(first_flow['header'])} packets")
                    if first_flow["header"]:
                        print(f"    First packet shape: {first_flow['header'][0].shape}")
                if "payload" in first_flow:
                    print(f"  Payload attention: {len(first_flow['payload'])} packets")
                    if first_flow["payload"]:
                        print(f"    First packet shape: {first_flow['payload'][0].shape}")
        else:
            print("⚠ No flow attention data to save")

        # 分别保存packet attention的三种类型
        if packet_hh_list:
            # 验证H-H attention是否都是正数（只验证第一个，因为大小可能不一致）
            hh_tensor = packet_hh_list[0] if isinstance(packet_hh_list[0], torch.Tensor) else packet_hh_list[0]
            print(f"\n📊 Packet H-H Attention Statistics (First batch):")
            print(f"  Shape: {hh_tensor.shape}")
            print(f"  Min value: {hh_tensor.min().item():.6f}")
            print(f"  Max value: {hh_tensor.max().item():.6f}")
            print(f"  Mean value: {hh_tensor.mean().item():.6f}")
            print(f"  Std value: {hh_tensor.std().item():.6f}")
            print(f"  Contains negative: {(hh_tensor < 0).any().item()}")
            print(f"  All values in [0, 1]: {((hh_tensor >= 0) & (hh_tensor <= 1)).all().item()}")

            packet_hh_path = os.path.join(output_dir, "attention_packet_hh.pt")
            torch.save(packet_hh_list, packet_hh_path)
            print(f"✓ Packet attention (h-h) saved to: {packet_hh_path}")
            print(f"  Number of batches: {len(packet_hh_list)}")
            if packet_hh_list:
                print(f"    First batch shape: {packet_hh_list[0].shape}")

        if packet_hp_list:
            # 验证H-P attention是否都是正数（只验证第一个，因为大小可能不一致）
            hp_tensor = packet_hp_list[0] if isinstance(packet_hp_list[0], torch.Tensor) else packet_hp_list[0]
            print(f"\n📊 Packet H-P Attention Statistics (First batch):")
            print(f"  Shape: {hp_tensor.shape}")
            print(f"  Min value: {hp_tensor.min().item():.6f}")
            print(f"  Max value: {hp_tensor.max().item():.6f}")
            print(f"  Mean value: {hp_tensor.mean().item():.6f}")
            print(f"  Std value: {hp_tensor.std().item():.6f}")
            print(f"  Contains negative: {(hp_tensor < 0).any().item()}")
            print(f"  All values in [0, 1]: {((hp_tensor >= 0) & (hp_tensor <= 1)).all().item()}")

            packet_hp_path = os.path.join(output_dir, "attention_packet_hp.pt")
            torch.save(packet_hp_list, packet_hp_path)
            print(f"✓ Packet attention (h-p) saved to: {packet_hp_path}")
            print(f"  Number of batches: {len(packet_hp_list)}")

        if packet_pp_list:
            # 验证P-P attention是否都是正数（只验证第一个，因为大小可能不一致）
            pp_tensor = packet_pp_list[0] if isinstance(packet_pp_list[0], torch.Tensor) else packet_pp_list[0]
            print(f"\n📊 Packet P-P Attention Statistics (First batch):")
            print(f"  Shape: {pp_tensor.shape}")
            print(f"  Min value: {pp_tensor.min().item():.6f}")
            print(f"  Max value: {pp_tensor.max().item():.6f}")
            print(f"  Mean value: {pp_tensor.mean().item():.6f}")
            print(f"  Std value: {pp_tensor.std().item():.6f}")
            print(f"  Contains negative: {(pp_tensor < 0).any().item()}")
            print(f"  All values in [0, 1]: {((pp_tensor >= 0) & (pp_tensor <= 1)).all().item()}")

            packet_pp_path = os.path.join(output_dir, "attention_packet_pp.pt")
            torch.save(packet_pp_list, packet_pp_path)
            print(f"✓ Packet attention (p-p) saved to: {packet_pp_path}")
            print(f"  Number of batches: {len(packet_pp_list)}")


    # 清理GPU和CPU内存
    torch.cuda.empty_cache()
    gc.collect()

    return accuracy, avg_loss, confusion, macro_precision, macro_recall, macro_f1


def main():
    """主函数"""
    import gc
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    # 池化策略
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Tokenizer类型
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")

    # 知识蒸馏
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    
    # 早停
    parser.add_argument("--early_stopping", action='store_true',
                        help="Enable early stopping mechanism.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="Minimum change in the monitored quantity to qualify as an improvement.")
    
    # **新增**: 5个数据包输入格式
    parser.add_argument("--five_packet_format", action='store_true', default=True,
                        help="Use 5-packet format for input ([CLS]+p1+[SEP]+...+p5+[SEP], each packet 80 tokens).")
    
    # **包级注意力机制参数**
    parser.add_argument("--use_packet_attention", action='store_true', default=True,
                        help="Enable packet-level attention mechanism.")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Window size for payload-payload window attention.")
    
    # **流级注意力机制参数**
    parser.add_argument("--use_flow_attention", action='store_true', default=True,
                        help="Enable flow-level attention mechanism.")
    
    # **通用参数**
    parser.add_argument("--packet_size", type=int, default=80,
                        help="Number of tokens per packet.")
    parser.add_argument("--header_size", type=int, default=40,
                        help="Number of tokens in packet header.")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads for both packet and flow attention.")
    
    args = parser.parse_args()

    # 检查CUDA是否可用（只支持GPU训练）
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("❌ 错误: CUDA不可用！")
        print("="*80)
        print("此脚本仅支持GPU训练。请确保：")
        print("  1. 已安装CUDA和cuDNN")
        print("  2. 已安装支持CUDA的PyTorch版本")
        print("  3. 使用正确的conda环境（如：conda activate Mine）")
        print(f"\n当前环境信息：")
        print(f"  - PyTorch版本: {torch.__version__}")
        print(f"  - CUDA可用: {torch.cuda.is_available()}")
        print("="*80 + "\n")
        raise RuntimeError("CUDA不可用，无法继续训练")
    
    print(f"✓ CUDA可用: {torch.cuda.device_count()} 个GPU")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # 从配置文件加载超参数
    args = load_hyperparam(args)

    # 设置随机种子
    set_seed(args.seed)

    # 统计标签数量
    args.labels_num = count_labels_num(args.train_path)

    # 构建tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    required_seq_length = required_flow_seq_length(FLOW_PACKET_COUNT, args.packet_size)
    if args.seq_length < required_seq_length:
        raise ValueError(
            f"seq_length={args.seq_length}会截断显式包边界协议；"
            f"{FLOW_PACKET_COUNT}包输入至少需要{required_seq_length}=1+{FLOW_PACKET_COUNT}*({args.packet_size}+1)。"
        )

    # 构建分类模型
    model = Classifier(args)

    # 加载或初始化参数
    load_or_initialize_parameters(args, model)

    # 设置设备（只使用GPU）
    args.device = torch.device("cuda:0")
    model = model.to(args.device)
    print(f"✓ Model moved to device: {args.device}")

    # 准备训练数据
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("=" * 80)
    print("Finetuning with 5-Packet Format and Two-Level Attention")
    print("=" * 80)
    print(f"Input format:")
    print(f"  src: [CLS] + packet1(80) + [SEP] + packet2(80) + [SEP] + packet3(80) + [SEP] + packet4(80) + [SEP] + packet5(80) + [SEP]")
    print(f"  seg: 所有位置都使用 segment 1")
    print(f"       ([CLS], packet1, [SEP], packet2, [SEP], packet3, [SEP], packet4, [SEP], packet5, [SEP] 全部标记为1)")
    print(f"Input length: 1 + 5 * (80 + 1) = 406，padding到410 tokens (padded to {args.seq_length})")
    print(f"Sequence length: {args.seq_length}")
    print(f"Batch size: {batch_size}")
    print(f"Training instances: {instances_num}")
    print(f"Number of labels: {args.labels_num}")
    print(f"Data source: .pt files from /root/datasets_new/fine-tuning数据集/")
    print("")
    print("=" * 80)
    print("Two-Level Attention Mechanism Configuration")
    print("=" * 80)
    print(f"Packet size: {args.packet_size} tokens (Header: {args.header_size}, Payload: {args.packet_size - args.header_size})")
    print(f"Number of attention heads: {args.num_attention_heads}")
    print("")

    # 包级注意力配置
    print("【Level 1: Packet-Level Attention (4 layers)】")
    print(f"  Status: {'✓ Enabled' if args.use_packet_attention else '✗ Disabled'}")
    if args.use_packet_attention:
        print(f"  Purpose: 对每个数据包内部的header-payload关系建模")
        print(f"  3 Sub-Attention Mechanisms (Serial Processing):")
        print(f"    (1) SubAttention1 (h-h): Header内部全局注意力 ({args.header_size} tokens)")
        print(f"    (2) SubAttention2 (h-p): Payload作q，Header作k&v的交叉注意力（使用SubAttention1输出）")
        print(f"    (3) SubAttention3 (p-p): Payload内部窗口注意力 (window={args.window_size})（使用SubAttention2输出）")
        print(f"  Layers: 4 layers (each packet processed independently)")
    print("")

    # 池化层
    print("【Pooling Layer】")
    print(f"  Purpose: 减少序列长度，为流级注意力做准备")
    print(f"  Method: Mean pooling on payload (40 tokens -> 4 tokens per packet)")
    print("")

    # 流级注意力配置
    print("【Level 2: Flow-Level Attention (8 layers)】")
    print(f"  Status: {'✓ Enabled' if args.use_flow_attention else '✗ Disabled'}")
    if args.use_flow_attention:
        print(f"  Purpose: 对5个数据包之间的跨包依赖关系建模")
        print(f"  Input: 包级注意力的输出（池化后）")
        print(f"  Mechanisms:")
        print(f"    (1) 数据重组: 拼接5个数据包的header和池化后的payload")
        print(f"    (2) 滑动窗口注意力（包括自己和相邻包）:")
        print(f"        - 包1: 窗口=[0,1]，attend到包1和包2")
        print(f"        - 包2: 窗口=[0,1,2]，attend到包1、包2和包3")
        print(f"        - 包3: 窗口=[1,2,3]，attend到包2、包3和包4")
        print(f"        - 包4: 窗口=[2,3,4]，attend到包3、包4和包5")
        print(f"        - 包5: 窗口=[3,4]，attend到包4和包5")
        print(f"    (3) h查h: Header的tokens作q，窗口内包的header作k&v")
        print(f"    (4) p查p: Payload的tokens作q，窗口内包的payload作k&v")
        print(f"  Layers: 8 layers (stacked flow attention)")
    print("=" * 80)

    # 构建优化器和调度器
    optimizer, scheduler = build_optimizer(args, model)

    # 混合精度
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    # 多GPU
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    # 早停
    if args.early_stopping:
        patience_counter = 0
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    else:
        patience_counter = None

    print("Start training.")

    # 训练循环
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()

        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                avg_loss = total_loss / args.report_steps
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.4f}".format(epoch, i + 1, avg_loss))
                total_loss = 0.0

                # 定期清理CPU内存
                if (i + 1) % (args.report_steps * 5) == 0:
                    gc.collect()

        # 验证
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} - Validation")
        print(f"{'='*80}")
        result = evaluate(args, read_dataset(args, args.dev_path))
        current_acc = result[0]
        current_loss = result[1]
        
        # 保存最佳模型
        if current_acc > best_result + args.min_delta:
            best_result = current_acc
            save_model(model, args.output_model_path)
            if args.early_stopping:
                patience_counter = 0
                print(f"✓ Epoch {epoch}: Validation improved! Loss: {current_loss:.4f}, Acc: {current_acc:.4f} (Best: {best_result:.4f}). Model saved.")
        else:
            if args.early_stopping:
                patience_counter += 1
                print(f"✗ Epoch {epoch}: Validation did not improve. Loss: {current_loss:.4f}, Acc: {current_acc:.4f} (Best: {best_result:.4f}). Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered! No improvement for {args.patience} consecutive epochs.")
                    print(f"Best validation accuracy: {best_result:.4f}")
                    print(f"Training stopped at epoch {epoch}/{args.epochs_num}")
                    break
        print(f"{'='*80}\n")

    # 测试
    if args.test_path is not None:
        print(f"\n{'='*80}")
        print("Test set evaluation")
        print(f"{'='*80}")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))

        # 提取数据集名称
        import os
        test_path = args.test_path
        dataset_name = os.path.basename(os.path.dirname(test_path))

        test_result = evaluate(args, read_dataset(args, args.test_path),
                             print_confusion_matrix=True,
                             save_attention_weights=True,
                             dataset_name=dataset_name)
        test_acc = test_result[0]
        test_loss = test_result[1]
        test_macro_precision = test_result[3]
        test_macro_recall = test_result[4]
        test_macro_f1 = test_result[5]
        print(f"\n{'='*80}")
        print(f"Final Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Macro-Precision: {test_macro_precision:.4f}")
        print(f"  Macro-Recall: {test_macro_recall:.4f}")
        print(f"  Macro-F1: {test_macro_f1:.4f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

