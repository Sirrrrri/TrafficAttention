#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import os
import pickle
import random
import time
import math
from datetime import datetime
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

et_bert_path = ''
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

burst_predicting_dir = "/root/我的代码和数据集/Mine/pre-training/pre-training_Burst-Predicting_Indicating"
mfr_masking_dir = "/root/我的代码和数据集/Mine/pre-training/pre-training_MFR-Masking"
if burst_predicting_dir not in sys.path:
    sys.path.insert(0, burst_predicting_dir)
if mfr_masking_dir not in sys.path:
    sys.path.insert(0, mfr_masking_dir)

try:
    from packet_masking import mask_flow_packets
except ImportError:
    mask_flow_packets = None

try:
    from mfr_masking import mask_single_packet_mfr, HEADER_SIZE, PAYLOAD_SIZE
except ImportError:
    mask_single_packet_mfr = None


class HeaderAttentionLayer(nn.Module):
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


class FlowAttention(nn.Module):
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
        return torch.mean(payload_reshaped, dim=2)

    def unpool_payload(self, pooled_payload):
        batch_size, pooled_len, hidden_size = pooled_payload.size()
        unpooled = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)
        return unpooled.view(batch_size, 40, hidden_size)

    def apply_attention_with_self(self, query_hidden, all_key_value_hiddens, query_idx, is_header=True):
        if is_header:
            query_layer, key_layer, value_layer, output_layer = self.header_query, self.header_key, self.header_value, self.header_output
        else:
            query_layer, key_layer, value_layer, output_layer = self.payload_query, self.payload_key, self.payload_value, self.payload_output
        query = self.transpose_for_scores(query_layer(query_hidden))
        all_keys = [self.transpose_for_scores(key_layer(kv)) for kv in all_key_value_hiddens]
        all_values = [self.transpose_for_scores(value_layer(kv)) for kv in all_key_value_hiddens]
        keys = torch.cat(all_keys, dim=2)
        values = torch.cat(all_values, dim=2)
        attention_scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = self.dropout(F.softmax(attention_scores, dim=-1))
        context = torch.matmul(attention_probs, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.hidden_size,))
        return self.dropout(output_layer(context))

    def forward(self, all_packet_hiddens):
        all_headers = [p[:, :self.header_size, :] for p in all_packet_hiddens]
        all_payloads = [p[:, self.header_size:, :] for p in all_packet_hiddens]
        all_pooled_payloads = [self.pool_payload(p) for p in all_payloads]
        output_headers, output_payloads = [], []
        windows = [
            ([all_headers[0], all_headers[1]], [all_pooled_payloads[0], all_pooled_payloads[1]]),
            ([all_headers[0], all_headers[1], all_headers[2]], [all_pooled_payloads[0], all_pooled_payloads[1], all_pooled_payloads[2]]),
            ([all_headers[1], all_headers[2], all_headers[3]], [all_pooled_payloads[1], all_pooled_payloads[2], all_pooled_payloads[3]]),
            ([all_headers[2], all_headers[3]], [all_pooled_payloads[2], all_pooled_payloads[3]]),
        ]
        for i in range(4):
            h_win, p_win = windows[i]
            header_out = self.apply_attention_with_self(all_headers[i], h_win, i, is_header=True)
            payload_out = self.apply_attention_with_self(all_pooled_payloads[i], p_win, i, is_header=False)
            output_headers.append(self.header_norm(all_headers[i] + header_out))
            output_payloads.append(self.payload_norm(all_payloads[i] + self.unpool_payload(payload_out)))
        return output_headers, output_payloads


class PacketAttentionStack(nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads=8, header_size=40, packet_size=80, window_size=20, dropout=0.1):
        super(PacketAttentionStack, self).__init__()
        self.num_layers = num_layers
        self.layer_groups = nn.ModuleList()
        for _ in range(num_layers):
            group = nn.ModuleList([
                HeaderAttentionLayer(hidden_size, num_attention_heads, header_size, dropout),
                CrossAttentionLayer(hidden_size, num_attention_heads, header_size, dropout),
                PayloadAttentionLayer(hidden_size, num_attention_heads, header_size, window_size, dropout)
            ])
            self.layer_groups.append(group)

    def forward(self, packet_hidden):
        output = packet_hidden
        for group in self.layer_groups:
            output = group[0](output)
            output = group[1](output)
            output = group[2](output)
        return output


class FlowAttentionStack(nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads=8, header_size=40, payload_size=40, dropout=0.1):
        super(FlowAttentionStack, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            FlowAttention(hidden_size, num_attention_heads, header_size, payload_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, all_packet_hiddens):
        output_packets = all_packet_hiddens
        output_headers = output_payloads = None
        for layer in self.layers:
            output_headers, output_payloads = layer(output_packets)
            output_packets = [torch.cat([h, p], dim=1) for h, p in zip(output_headers, output_payloads)]
        return output_headers, output_payloads


class PoolingLayer(nn.Module):
    def __init__(self, header_size=40, payload_size=40):
        super(PoolingLayer, self).__init__()
        self.header_size = header_size
        self.payload_size = payload_size

    def forward(self, all_packet_hiddens):
        pooled_packets = []
        for packet_hidden in all_packet_hiddens:
            header = packet_hidden[:, :self.header_size, :]
            payload = packet_hidden[:, self.header_size:, :]
            batch_size, seq_len, hidden_size = payload.size()
            pooled_payload = torch.mean(payload.view(batch_size, 4, 10, hidden_size), dim=2)
            unpooled_payload = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1).view(batch_size, 40, hidden_size)
            pooled_packets.append(torch.cat([header, unpooled_payload], dim=1))
        return pooled_packets


class BaseModel(nn.Module):
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
        emb = self.embedding(src, seg)
        batch_size, seq_length, hidden_size = emb.size()
        packet_embeddings = []
        for i in range(4):
            start_idx = 1 + i * 80
            end_idx = start_idx + 80
            if end_idx > seq_length:
                end_idx = seq_length
            packet_emb = emb[:, start_idx:end_idx, :]
            if packet_emb.size(1) < 80:
                padding = torch.zeros(batch_size, 80 - packet_emb.size(1), hidden_size, device=emb.device)
                packet_emb = torch.cat([packet_emb, padding], dim=1)
            packet_embeddings.append(packet_emb)
        packet_outputs = [self.packet_attention_stack(p) for p in packet_embeddings]
        pooled_packets = self.pooling_layer(packet_outputs)
        flow_headers, flow_payloads = self.flow_attention_stack(pooled_packets)
        cls_token = emb[:, 0:1, :]
        sep_token = emb[:, 321:322, :]
        all_packets = [torch.cat([h, p], dim=1) for h, p in zip(flow_headers, flow_payloads)]
        packets_concat = torch.cat(all_packets, dim=1)
        output = torch.cat([cls_token, packets_concat, sep_token], dim=1)
        if output.size(1) < seq_length:
            padding = torch.zeros(batch_size, seq_length - output.size(1), hidden_size, device=output.device)
            output = torch.cat([output, padding], dim=1)
        elif output.size(1) > seq_length:
            output = output[:, :seq_length, :]
        return output


class MultiTaskModel(nn.Module):
    def __init__(self, base_model, args, vocab_size):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.args = args
        self.nsp_indicating_weight = args.nsp_indicating_weight
        self.mfr_masking_weight = args.mfr_masking_weight
        self.indicating_mlm_output = nn.Linear(args.hidden_size, vocab_size)
        self.mfr_mlm_output = nn.Linear(args.hidden_size, vocab_size)
        self.mlm_loss = nn.CrossEntropyLoss()

    def forward(self, src, seg, task_type, tgt_mlm=None):
        output = self.base_model(src, seg)
        if task_type == 'nsp_indicating':
            mlm_logits = self.indicating_mlm_output(output)
            if tgt_mlm is not None:
                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_mlm_flat = tgt_mlm.view(-1)
                mask = tgt_mlm_flat > 0
                if mask.sum() > 0:
                    loss = self.mlm_loss(mlm_logits_flat[mask], tgt_mlm_flat[mask])
                    pred = torch.argmax(mlm_logits_flat[mask], dim=-1)
                    correct = torch.sum(pred == tgt_mlm_flat[mask]).float()
                else:
                    loss = torch.tensor(0.0, device=output.device)
                    correct = torch.tensor(0.0, device=output.device)
                return loss, correct
            else:
                return mlm_logits
        elif task_type == 'mfr_masking':
            mlm_logits = self.mfr_mlm_output(output)
            if tgt_mlm is not None:
                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_mlm_flat = tgt_mlm.view(-1)
                mask = tgt_mlm_flat > 0
                if mask.sum() > 0:
                    loss = self.mlm_loss(mlm_logits_flat[mask], tgt_mlm_flat[mask])
                    pred = torch.argmax(mlm_logits_flat[mask], dim=-1)
                    correct = torch.sum(pred == tgt_mlm_flat[mask]).float()
                else:
                    loss = torch.tensor(0.0, device=output.device)
                    correct = torch.tensor(0.0, device=output.device)
                return loss, correct
            else:
                return mlm_logits
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class MultiTaskDataset(Dataset):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=True):
        self.args = args
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.seq_length = args.seq_length
        self.tokens_per_packet = args.tokens_per_packet
        self.task_weights = [args.nsp_indicating_weight, args.mfr_masking_weight]
        self.task_names = ['nsp_indicating', 'mfr_masking']
        self.packet_selection_prob = args.packet_selection_prob
        self.packet_mask_ratio = args.packet_mask_ratio
        self.mfr_header_mask_ratio = args.mfr_header_mask_ratio
        self.mfr_payload_mask_ratio = args.mfr_payload_mask_ratio
        self.hh_ratio = getattr(args, 'hh_ratio', 0.30)
        self.pp_ratio = getattr(args, 'pp_ratio', 0.30)
        self.hp_ratio = getattr(args, 'hp_ratio', 0.40)
        self.hh_mask_ratio = getattr(args, 'hh_mask_ratio', 0.80)
        self.pp_mask_ratio = getattr(args, 'pp_mask_ratio', 0.80)
        self.hp_mask_ratio = getattr(args, 'hp_mask_ratio', 0.95)
        self.vocab = args.vocab
        self.mask_id = self.vocab.get('[MASK]', 103)
        self.cls_id = self.vocab.get('[CLS]', 101)
        self.sep_id = self.vocab.get('[SEP]', 102)
        self.pad_id = self.vocab.get('[PAD]', 0)
        self.instances_buffer_size = min(getattr(args, 'instances_buffer_size', 25600), batch_size * 50)
        self.buffer = []
        self.buffer_start = 0
        self.buffer_end = 0
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.initial_load = True
        self.first_batch_generated = False

    def _fill_buffer(self):
        self.buffer = []
        target_size = self.instances_buffer_size
        if self.initial_load:
            target_size = min(self.batch_size * 2, self.instances_buffer_size)
        loaded = 0
        while len(self.buffer) < target_size:
            try:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                loaded += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
            except EOFError:
                self.dataset_reader.seek(0)
                self.read_count = 0
                if len(self.buffer) == 0:
                    break
        if self.shuffle:
            random.shuffle(self.buffer)
        self.buffer_start = 0
        self.buffer_end = len(self.buffer)
        self.initial_load = False

    def _is_buffer_empty(self):
        return self.buffer_start >= self.buffer_end

    def __del__(self):
        if hasattr(self, 'dataset_reader'):
            self.dataset_reader.close()

    def _organize_buffer_data(self, instances):
        flows = []
        raw_packets = []
        for idx, instance in enumerate(instances):
            try:
                if isinstance(instance, tuple) and len(instance) == 4:
                    raw_packets.append(instance)
                elif isinstance(instance, (list, tuple)):
                    raw_flow = []
                    for token in instance:
                        if isinstance(token, int):
                            raw_flow.append(token)
                        elif isinstance(token, str):
                            try:
                                val = int(token, 16)
                                raw_flow.append(val + 5)
                            except (ValueError, TypeError):
                                try:
                                    raw_flow.append(int(token))
                                except (ValueError, TypeError):
                                    raw_flow.append(self.pad_id)
                        else:
                            try:
                                raw_flow.append(int(token))
                            except (ValueError, TypeError):
                                raw_flow.append(self.pad_id)
                    flow = []
                    for pkt_idx in range(4):
                        start = pkt_idx * 160
                        packet_tokens = raw_flow[start:start+80]
                        if len(packet_tokens) < 80:
                            packet_tokens = packet_tokens + [self.pad_id] * (80 - len(packet_tokens))
                        flow.extend(packet_tokens[:80])
                    expected_len = self.tokens_per_packet * 4
                    if len(flow) < expected_len:
                        flow = flow + [self.pad_id] * (expected_len - len(flow))
                    elif len(flow) > expected_len:
                        flow = flow[:expected_len]
                    flows.append(flow)
            except Exception:
                continue
        if len(raw_packets) > 0:
            flow_dict = {}
            for flow_id, packet_idx, packet_tokens, dataset_name in raw_packets:
                if flow_id not in flow_dict:
                    flow_dict[flow_id] = []
                flow_dict[flow_id].append((packet_idx, packet_tokens))
            for flow_id, packets in flow_dict.items():
                packets.sort(key=lambda x: x[0])
                packets = packets[:4]
                if len(packets) == 0:
                    continue
                flow_tokens = []
                for _, packet_tokens in packets:
                    packet_int_tokens = []
                    for token_str in packet_tokens[:80]:
                        try:
                            val = int(token_str, 16)
                            packet_int_tokens.append(val + 5)
                        except (ValueError, TypeError):
                            try:
                                packet_int_tokens.append(int(token_str))
                            except (ValueError, TypeError):
                                packet_int_tokens.append(self.pad_id)
                    while len(packet_int_tokens) < 80:
                        packet_int_tokens.append(self.pad_id)
                    flow_tokens.extend(packet_int_tokens[:80])
                while len(flow_tokens) < 320:
                    flow_tokens.append(self.pad_id)
                if len(flow_tokens) > 320:
                    flow_tokens = flow_tokens[:320]
                flows.append(flow_tokens)
        return flows

    def _apply_packet_masking(self, tokens):
        expected_len = self.tokens_per_packet * 4
        if len(tokens) < expected_len:
            tokens = tokens + [self.pad_id] * (expected_len - len(tokens))
        elif len(tokens) > expected_len:
            tokens = tokens[:expected_len]
        masked_tokens = tokens.copy()
        labels = [-1] * len(tokens)
        cns_id = self.vocab.get('[CNS]', self.pad_id)
        header_size = 40
        use_hhhh_relation = random.random() < 0.5
        if use_hhhh_relation:
            for packet_idx in range(4):
                packet_start = packet_idx * self.tokens_per_packet
                for i in range(packet_start + header_size, packet_start + self.tokens_per_packet):
                    if i < len(tokens):
                        masked_tokens[i] = cns_id
            selected_packet_idx = random.randint(0, 3)
            packet_start = selected_packet_idx * self.tokens_per_packet
            header_positions = [i for i in range(packet_start, packet_start + header_size)
                                 if i < len(tokens) and tokens[i] != self.pad_id]
            if header_positions:
                num_to_mask = max(1, int(len(header_positions) * 0.90))
                for pos in random.sample(header_positions, min(num_to_mask, len(header_positions))):
                    labels[pos] = tokens[pos]
                    r = random.random()
                    if r < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif r < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
        else:
            for packet_idx in range(4):
                packet_start = packet_idx * self.tokens_per_packet
                for i in range(packet_start, packet_start + header_size):
                    if i < len(tokens):
                        masked_tokens[i] = cns_id
            selected_packet_idx = random.randint(0, 3)
            packet_start = selected_packet_idx * self.tokens_per_packet
            payload_positions = [i for i in range(packet_start + header_size, packet_start + self.tokens_per_packet)
                                  if i < len(tokens) and tokens[i] != self.pad_id]
            if payload_positions:
                num_to_mask = max(1, int(len(payload_positions) * 0.90))
                for pos in random.sample(payload_positions, min(num_to_mask, len(payload_positions))):
                    labels[pos] = tokens[pos]
                    r = random.random()
                    if r < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif r < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
        return masked_tokens, labels

    def _apply_mfr_masking_single_packet(self, packet_tokens, relation_type):
        if len(packet_tokens) < self.tokens_per_packet:
            packet_tokens = packet_tokens + [self.pad_id] * (self.tokens_per_packet - len(packet_tokens))
        elif len(packet_tokens) > self.tokens_per_packet:
            packet_tokens = packet_tokens[:self.tokens_per_packet]
        masked_tokens = packet_tokens.copy()
        labels = [-1] * len(packet_tokens)
        cns_id = self.vocab.get('[CNS]', self.pad_id)
        header_size = 40
        if relation_type == 'hh':
            for i in range(header_size, self.tokens_per_packet):
                masked_tokens[i] = cns_id
            positions = [i for i in range(header_size) if packet_tokens[i] != self.pad_id]
            mask_ratio = 0.80
        elif relation_type == 'pp':
            for i in range(header_size):
                masked_tokens[i] = cns_id
            positions = [i for i in range(header_size, self.tokens_per_packet) if packet_tokens[i] != self.pad_id]
            mask_ratio = 0.80
        else:  # hp
            positions = [i for i in range(header_size) if packet_tokens[i] != self.pad_id]
            mask_ratio = 0.95
        if positions:
            num_to_mask = max(1, int(len(positions) * mask_ratio))
            for pos in random.sample(positions, min(num_to_mask, len(positions))):
                labels[pos] = packet_tokens[pos]
                r = random.random()
                if r < 0.8:
                    masked_tokens[pos] = self.mask_id
                elif r < 0.9:
                    masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)
        return masked_tokens, labels

    def _create_nsp_indicating_sample(self, flow):
        expected_len = self.tokens_per_packet * 4
        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]
        masked_tokens, labels = self._apply_packet_masking(flow)
        src = [self.cls_id] + masked_tokens
        seg = [0] * len(src)
        tgt_mlm = [-1] + labels
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
        expected_len = self.tokens_per_packet * 4
        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]
        packet_start = packet_idx * self.tokens_per_packet
        packet_tokens = flow[packet_start:packet_start + self.tokens_per_packet]
        masked_tokens, labels = self._apply_mfr_masking_single_packet(packet_tokens, relation_type)
        src = [self.cls_id] + masked_tokens + [self.sep_id]
        seg = [0] * len(src)
        tgt_mlm = [-1] + labels + [-1]
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
        iter_count = 0
        while True:
            if self._is_buffer_empty():
                self._fill_buffer()
                if len(self.buffer) == 0:
                    break
                batch_instances = self.buffer[self.buffer_start:min(self.buffer_start + self.batch_size * 10, self.buffer_end)]
                self.buffer_start += len(batch_instances)
                current_flows = self._organize_buffer_data(batch_instances)
            else:
                if self.buffer_start >= self.buffer_end:
                    continue
                batch_instances = self.buffer[self.buffer_start:min(self.buffer_start + self.batch_size * 10, self.buffer_end)]
                self.buffer_start += len(batch_instances)
                current_flows = self._organize_buffer_data(batch_instances)

            if len(current_flows) == 0:
                continue

            for batch_start_idx in range(0, len(current_flows), self.batch_size):
                batch_flows = current_flows[batch_start_idx:min(batch_start_idx + self.batch_size, len(current_flows))]
                if len(batch_flows) == 0:
                    continue

                batch_src, batch_seg, batch_tgt_mlm, batch_task_type = [], [], [], []

                for flow in batch_flows:
                    src, seg, tgt_mlm = self._create_nsp_indicating_sample(flow)
                    batch_src.append(src)
                    batch_seg.append(seg)
                    batch_tgt_mlm.append(tgt_mlm)
                    batch_task_type.append('nsp_indicating')

                num_flows = len(batch_flows)
                total_packets = num_flows * 4
                num_hh = max(1, int(total_packets * self.hh_ratio))
                num_pp = max(1, int(total_packets * self.pp_ratio))
                packet_list = [(fi, pi) for fi in range(num_flows) for pi in range(4)]
                random.shuffle(packet_list)
                hh_packets = set(packet_list[:num_hh])
                pp_packets = set(packet_list[num_hh:num_hh + num_pp])

                for flow_idx, flow in enumerate(batch_flows):
                    for packet_idx in range(4):
                        if (flow_idx, packet_idx) in hh_packets:
                            relation_type = 'hh'
                        elif (flow_idx, packet_idx) in pp_packets:
                            relation_type = 'pp'
                        else:
                            relation_type = 'hp'
                        src, seg, tgt_mlm = self._create_mfr_masking_sample(flow, packet_idx, relation_type)
                        batch_src.append(src)
                        batch_seg.append(seg)
                        batch_tgt_mlm.append(tgt_mlm)
                        batch_task_type.append('mfr_masking')

                if batch_src:
                    self.first_batch_generated = True
                    iter_count += 1
                    yield (
                        torch.LongTensor(batch_src),
                        torch.LongTensor(batch_seg),
                        torch.LongTensor(batch_tgt_mlm),
                        batch_task_type
                    )


class MultiTaskTrainer:
    def __init__(self, args):
        self.args = args
        checkpoint_step = getattr(args, 'checkpoint_step', 0)
        self.current_step = checkpoint_step + 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.output_model_path = args.output_model_path
        self.start_time = time.time()
        self.task_stats = {
            'nsp_indicating': {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0},
            'mfr_masking': {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0}
        }
        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.mlm_loss = nn.CrossEntropyLoss()

    def forward_propagation(self, batch, model):
        src, seg, tgt_mlm, task_types = batch
        batch_size = src.size(0)
        sub_batch_size = max(1, batch_size // 4)
        total_loss = None
        for start_idx in range(0, batch_size, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, batch_size)
            src_sub = src[start_idx:end_idx]
            seg_sub = seg[start_idx:end_idx]
            tgt_mlm_sub = tgt_mlm[start_idx:end_idx]
            task_types_sub = task_types[start_idx:end_idx]
            output = model.module.base_model(src_sub, seg_sub) if hasattr(model, 'module') else model.base_model(src_sub, seg_sub)
            task_groups = {'nsp_indicating': [], 'mfr_masking': []}
            for i, task_type in enumerate(task_types_sub):
                task_groups[task_type].append(i)
            if task_groups['nsp_indicating']:
                indices = task_groups['nsp_indicating']
                indicating_output_layer = model.module.indicating_mlm_output if hasattr(model, 'module') else model.indicating_mlm_output
                mlm_logits = indicating_output_layer(output[indices])
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
                    num_masked_tokens = torch.tensor(1.0, device=src.device)
                weighted_loss = loss * self.args.nsp_indicating_weight * len(indices) / batch_size
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                self.task_stats['nsp_indicating']['loss'] += loss.item() * len(indices)
                self.task_stats['nsp_indicating']['correct'] += correct.item()
                self.task_stats['nsp_indicating']['num_samples'] += len(indices)
                self.task_stats['nsp_indicating']['num_tokens'] += num_masked_tokens.item()
            if task_groups['mfr_masking']:
                indices = task_groups['mfr_masking']
                mfr_output_layer = model.module.mfr_mlm_output if hasattr(model, 'module') else model.mfr_mlm_output
                mlm_logits = mfr_output_layer(output[indices])
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
                    num_masked_tokens = torch.tensor(1.0, device=src.device)
                weighted_loss = loss * self.args.mfr_masking_weight * len(indices) / batch_size
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                self.task_stats['mfr_masking']['loss'] += loss.item() * len(indices)
                self.task_stats['mfr_masking']['correct'] += correct.item()
                self.task_stats['mfr_masking']['num_samples'] += len(indices)
                self.task_stats['mfr_masking']['num_tokens'] += num_masked_tokens.item()
            del output, src_sub, seg_sub, tgt_mlm_sub
            torch.cuda.empty_cache()
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=src.device, requires_grad=True)
        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        elapsed = time.time() - self.start_time
        print(f"| Step: {self.current_step:6d}/{self.total_steps} | Time: {elapsed:6.2f}s |")
        for task_name in ['nsp_indicating', 'mfr_masking']:
            stats = self.task_stats[task_name]
            if stats['num_samples'] > 0:
                avg_loss = stats['loss'] / stats['num_samples']
                accuracy = stats['correct'] / stats['num_tokens'] * 100 if stats['num_tokens'] > 0 else 0.0
                print(f"| {task_name:15s} | Loss: {avg_loss:6.4f} | Acc: {accuracy:6.2f}% | Samples: {int(stats['num_samples']):4d} |")
            else:
                print(f"| {task_name:15s} | No samples |")
        print("-" * 80)
        for task_name in self.task_stats:
            self.task_stats[task_name] = {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0}
        self.start_time = time.time()

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        self.optimizer = optimizer
        self.scheduler = scheduler
        for batch in loader:
            if self.current_step > self.total_steps:
                break
            if gpu_id is not None:
                batch = list(batch)
                for i in range(len(batch) - 1):
                    if torch.is_tensor(batch[i]):
                        batch[i] = batch[i].cuda(gpu_id, non_blocking=True)
            loss = self.forward_propagation(batch, model)
            loss.backward()
            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if gpu_id is not None:
                torch.cuda.empty_cache()
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                checkpoint_path = self.output_model_path + "-" + str(self.current_step)
                checkpoint = {
                    'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'step': self.current_step,
                }
                torch.save(checkpoint, checkpoint_path)
            self.current_step += 1


def train_and_validate_multi_task(args):
    set_seed(args.seed)
    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("You need to install SentencePiece: pip install sentencepiece")
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab

    embedding = str2embedding[args.embedding](args, len(args.vocab))
    packet_size = getattr(args, 'tokens_per_packet', 80)
    header_size = getattr(args, 'header_size', 40)
    payload_size = packet_size - header_size
    window_size = getattr(args, 'window_size', 20)

    packet_attention_stack = PacketAttentionStack(
        num_layers=4, hidden_size=args.hidden_size,
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        header_size=header_size, packet_size=packet_size,
        window_size=window_size, dropout=getattr(args, 'dropout', 0.1)
    )
    pooling_layer = PoolingLayer(header_size=header_size, payload_size=payload_size)
    flow_attention_stack = FlowAttentionStack(
        num_layers=8, hidden_size=args.hidden_size,
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        header_size=header_size, payload_size=payload_size,
        dropout=getattr(args, 'dropout', 0.1)
    )
    base_model = BaseModel(
        embedding, packet_attention_stack=packet_attention_stack,
        pooling_layer=pooling_layer, flow_attention_stack=flow_attention_stack,
        packet_size=packet_size, header_size=header_size
    )
    model = MultiTaskModel(base_model, args, len(args.vocab))

    optimizer_state_dict = None
    scheduler_state_dict = None
    if args.pretrained_model_path is not None:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
            scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
        else:
            pretrained_dict = checkpoint
        model_dict = model.state_dict()
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
    else:
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        mp.spawn(worker_multi_task, nprocs=args.ranks_num,
                 args=(args.gpu_ranks, args, model, optimizer_state_dict, scheduler_state_dict), daemon=False)
    elif args.single_gpu:
        worker_multi_task(args.gpu_id, None, args, model, optimizer_state_dict, scheduler_state_dict)
    else:
        worker_multi_task(None, None, args, model, optimizer_state_dict, scheduler_state_dict)


def worker_multi_task(gpu_id, gpu_ranks, args, model, optimizer_state_dict=None, scheduler_state_dict=None):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    rank = 0
    if args.dist_train:
        rank = gpu_ranks.index(gpu_id)
        dist.init_process_group(backend=args.backend, init_method=args.master_ip,
                                world_size=args.world_size, rank=rank)
        model.cuda(gpu_id)
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

    loader = MultiTaskDataset(args, args.dataset_path, args.batch_size, proc_id, proc_num, shuffle=True)

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

    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
        except Exception:
            pass
    if scheduler_state_dict is not None:
        try:
            scheduler.load_state_dict(scheduler_state_dict)
        except Exception:
            pass

    trainer = MultiTaskTrainer(args)
    trainer.train(args, gpu_id, rank, loader, model, optimizer, scheduler)

    if proc_id == 0:
        final_checkpoint = {
            'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'step': trainer.current_step,
        }
        torch.save(final_checkpoint, args.output_model_path)
