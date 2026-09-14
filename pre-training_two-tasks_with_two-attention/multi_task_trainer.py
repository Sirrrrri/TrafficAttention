import sys
import os
import pickle
import random
import time
import math
import traceback
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

FLOW_PACKET_COUNT = 5


def packet_start_positions(packet_count=FLOW_PACKET_COUNT, packet_size=80):
    return [1 + i * (packet_size + 1) for i in range(packet_count)]


def sep_positions(packet_count=FLOW_PACKET_COUNT, packet_size=80):
    return [start + packet_size for start in packet_start_positions(packet_count, packet_size)]


def required_flow_seq_length(packet_count=FLOW_PACKET_COUNT, packet_size=80):
    return 1 + packet_count * (packet_size + 1)


def build_flow_sequence(packets, cls_id, sep_id, pad_id, seq_length, packet_count=FLOW_PACKET_COUNT, packet_size=80):
    src = [cls_id]
    seg = [1]

    for packet_idx in range(packet_count):
        packet = packets[packet_idx] if packet_idx < len(packets) else []
        packet = packet[:packet_size] + [pad_id] * max(0, packet_size - len(packet))
        src.extend(packet[:packet_size])
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


def build_flow_labels(packet_labels, seq_length, packet_count=FLOW_PACKET_COUNT, packet_size=80):
    labels = [-1]

    for packet_idx in range(packet_count):
        packet = packet_labels[packet_idx] if packet_idx < len(packet_labels) else []
        packet = packet[:packet_size] + [-1] * max(0, packet_size - len(packet))
        labels.extend(packet[:packet_size])
        labels.append(-1)

    if len(labels) > seq_length:
        labels = labels[:seq_length]
    while len(labels) < seq_length:
        labels.append(-1)

    return labels



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

    def __init__(self, hidden_size, num_attention_heads=8, header_size=40, packet_size=80, window_size=20, dropout=0.1):
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

        payload_size = packet_size - header_size
        half_window = window_size // 2
        window_mask = torch.full((payload_size, payload_size), float('-inf'))
        for i in range(payload_size):
            start = max(0, i - half_window)
            end = min(payload_size, i + half_window + 1)
            window_mask[i, start:end] = 0.0
        self.register_buffer("window_mask", window_mask)

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

        window_mask = self.window_mask[:seq_len, :seq_len].to(dtype=attention_scores.dtype)
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
        self.packet_count = 5
        
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

        flow_window_mask = torch.full((self.packet_count, self.packet_count), float('-inf'))
        for packet_idx in range(self.packet_count):
            window_start = max(0, packet_idx - 1)
            window_end = min(self.packet_count, packet_idx + 2)
            flow_window_mask[packet_idx, window_start:window_end] = 0.0
        self.register_buffer("flow_window_mask", flow_window_mask)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def unpool_payload(self, pooled_payload):
        batch_size, pooled_len, hidden_size = pooled_payload.size()
        unpooled = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)
        unpooled = unpooled.view(batch_size, 40, hidden_size)
        return unpooled

    def transpose_flow_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def apply_window_attention(self, hidden, is_header=True):
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

        batch_size, packet_count, seq_len, _ = hidden.size()
        query = self.transpose_flow_for_scores(query_layer(hidden))
        key = self.transpose_flow_for_scores(key_layer(hidden))
        value = self.transpose_flow_for_scores(value_layer(hidden))

        key = key.reshape(batch_size, self.num_attention_heads, packet_count * seq_len, self.attention_head_size)
        value = value.reshape(batch_size, self.num_attention_heads, packet_count * seq_len, self.attention_head_size)

        attention_scores = torch.matmul(query, key.transpose(-1, -2).unsqueeze(2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        packet_mask = self.flow_window_mask[:packet_count, :packet_count].to(
            dtype=attention_scores.dtype,
            device=attention_scores.device
        )
        token_mask = packet_mask.repeat_interleave(seq_len, dim=-1)
        attention_scores = attention_scores + token_mask.view(1, 1, packet_count, 1, packet_count * seq_len)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value.unsqueeze(2))
        context = context.permute(0, 2, 3, 1, 4).contiguous()
        context = context.view(batch_size, packet_count, seq_len, self.hidden_size)

        output = output_layer(context)
        output = self.dropout(output)
        return output
    
    def forward(self, all_packet_hiddens):
        packet_count = len(all_packet_hiddens)
        if packet_count != self.packet_count:
            raise ValueError(f"FlowAttention expects {self.packet_count} packets, got {packet_count}.")

        packets = torch.stack(all_packet_hiddens, dim=1)
        headers = packets[:, :, :self.header_size, :]
        pooled_payloads = packets[:, :, self.header_size:, :]

        header_out = self.apply_window_attention(headers, is_header=True)
        payload_out = self.apply_window_attention(pooled_payloads, is_header=False)

        header_final = self.header_norm(headers + header_out)
        payload_final = self.payload_norm(pooled_payloads + payload_out)

        output_headers = [header_final[:, i] for i in range(packet_count)]
        output_payloads = [payload_final[:, i] for i in range(packet_count)]

        return output_headers, output_payloads


class PacketAttentionStack(nn.Module):

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
                    packet_size=packet_size,
                    window_size=window_size,
                    dropout=dropout
                )
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

        output_packets = all_packet_hiddens
        output_headers = None
        output_payloads = None

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
            payload_reshaped = payload.view(batch_size, 4, 10, hidden_size)
            pooled_payload = torch.mean(payload_reshaped, dim=2)  


            pooled_packet = torch.cat([header, pooled_payload], dim=1)
            pooled_packets.append(pooled_packet)

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

        starts = packet_start_positions(FLOW_PACKET_COUNT, self.packet_size)
        seps = sep_positions(FLOW_PACKET_COUNT, self.packet_size)

        packet_embeddings = []
        for start_idx in starts:
            end_idx = start_idx + self.packet_size


            if end_idx > seq_length:
                end_idx = seq_length

            packet_emb = emb[:, start_idx:end_idx, :]  


            if packet_emb.size(1) < self.packet_size:
                padding = torch.zeros(batch_size, self.packet_size - packet_emb.size(1), hidden_size, device=emb.device)
                packet_emb = torch.cat([packet_emb, padding], dim=1)

            packet_embeddings.append(packet_emb)


        packet_outputs = []
        for packet_emb in packet_embeddings:
            packet_out = self.packet_attention_stack(packet_emb)  
            packet_outputs.append(packet_out)


        pooled_packets = self.pooling_layer(packet_outputs)  


        flow_headers, flow_payloads = self.flow_attention_stack(pooled_packets)  


        cls_token = emb[:, 0:1, :]  
        sep_tokens = []
        for sep_idx in seps:
            if sep_idx < seq_length:
                sep_tokens.append(emb[:, sep_idx:sep_idx + 1, :])
            else:
                sep_tokens.append(torch.zeros(batch_size, 1, hidden_size, device=emb.device, dtype=emb.dtype))


        all_packets = []
        for header, payload in zip(flow_headers, flow_payloads):
            payload = payload.unsqueeze(2).repeat(1, 1, 10, 1).view(batch_size, self.packet_size - self.header_size, hidden_size)
            packet = torch.cat([header, payload], dim=1)  
            all_packets.append(packet)

        output_parts = [cls_token]
        for packet, sep_token in zip(all_packets, sep_tokens):
            output_parts.extend([packet, sep_token])
        output = torch.cat(output_parts, dim=1)


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

            mlm_logits = self.mfr_mlm_output(output)  
            
            if tgt_mlm is not None:

                mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                tgt_mlm_flat = tgt_mlm.view(-1)
                

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

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=True):
        self.args = args
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        
        self.seq_length = args.seq_length
        self.tokens_per_packet = args.tokens_per_packet
        self.packet_count = FLOW_PACKET_COUNT
        required_seq_length = required_flow_seq_length(self.packet_count, self.tokens_per_packet)
        if self.seq_length < required_seq_length:
            raise ValueError(
                f"seq_length={self.seq_length}会截断显式包边界协议；"
                f"{self.packet_count}包输入至少需要{required_seq_length}=1+{self.packet_count}*({self.tokens_per_packet}+1)。"
            )
        

        self.task_weights = [
            args.nsp_indicating_weight,
            args.mfr_masking_weight
        ]
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
        self.unk_id = self.vocab.get('[UNK]', self.pad_id)
        self.invalid_token_count = 0
        


        self.instances_buffer_size = min(getattr(args, 'instances_buffer_size', 25600), batch_size * 50)
        self.buffer = []
        self.buffer_start = 0
        self.buffer_end = 0
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.initial_load = True  
        
        if self.proc_id == 0:
            pass
        

        self.first_batch_generated = False

    def _token_to_id(self, token):
        token_text = str(token).strip().lower()
        token_id = self.vocab.get(token_text)
        if token_id is not None:
            return token_id

        self.invalid_token_count += 1
        if self.proc_id == 0 and self.invalid_token_count <= 5:
            pass
        return self.unk_id

    def _tokens_to_ids(self, tokens):
        return [self._token_to_id(token) for token in tokens]

    def _normalize_flow_tokens(self, raw_tokens):
        raw_flow = self._tokens_to_ids(raw_tokens)

        expected_len = self.tokens_per_packet * self.packet_count
        source_stride = self.tokens_per_packet
        if len(raw_flow) >= self.packet_count * self.tokens_per_packet * 2:
            source_stride = self.tokens_per_packet * 2

        flow = []
        for pkt_idx in range(self.packet_count):
            start = pkt_idx * source_stride
            packet_tokens = raw_flow[start:start + self.tokens_per_packet]
            if len(packet_tokens) < self.tokens_per_packet:
                packet_tokens = packet_tokens + [self.pad_id] * (self.tokens_per_packet - len(packet_tokens))
            flow.extend(packet_tokens[:self.tokens_per_packet])

        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]

        return flow
    
    def _fill_buffer(self):

        self.buffer = []
        target_size = self.instances_buffer_size
        if self.initial_load:
            target_size = min(self.batch_size * 2, self.instances_buffer_size)
            if self.proc_id == 0:
                pass
        
        loaded = 0
        start_time = time.time()
        while len(self.buffer) < target_size:
            try:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                loaded += 1
                

                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                

                if self.initial_load and self.proc_id == 0 and loaded % 1000 == 0:
                    elapsed = time.time() - start_time
                    speed = loaded / elapsed if elapsed > 0 else 0
                    pass
                    
            except EOFError:

                self.dataset_reader.seek(0)
                self.read_count = 0
                

                if len(self.buffer) == 0:
                    break
        
        if self.shuffle:
            random.shuffle(self.buffer)
        
        self.buffer_start = 0
        self.buffer_end = len(self.buffer)
        

        if self.initial_load:
            elapsed = time.time() - start_time
            if self.proc_id == 0:
                pass
            self.initial_load = False
    
    def _is_buffer_empty(self):

        return self.buffer_start >= self.buffer_end
    
    def __del__(self):

        if hasattr(self, 'dataset_reader'):
            self.dataset_reader.close()
    
    def _organize_buffer_data(self, instances):

        start_time = time.time()
        
        flows = []  
        raw_packets = []  
        

        if self.proc_id == 0 and not self.first_batch_generated and len(instances) > 0:
            first_instance = instances[0]
            pass
            if isinstance(first_instance, (tuple, list)) and len(first_instance) <= 10:
                pass
            elif isinstance(first_instance, (tuple, list)):
                pass
        

        parse_error_count = 0
        for idx, instance in enumerate(instances):
            try:
                if isinstance(instance, tuple) and len(instance) == 3:

                    _, flow_mfr_tokens, _ = instance
                    flows.append(self._normalize_flow_tokens(flow_mfr_tokens))
                elif isinstance(instance, tuple) and len(instance) == 4:

                    raw_packets.append(instance)
                elif isinstance(instance, list):
                    flows.append(self._normalize_flow_tokens(instance))
                elif isinstance(instance, tuple):

                    flows.append(self._normalize_flow_tokens(instance))
            except Exception as e:
                parse_error_count += 1
                if parse_error_count <= 3 and self.proc_id == 0 and not self.first_batch_generated:
                    pass
                continue


        if len(raw_packets) > 0:
            flow_dict = {}
            for flow_id, packet_idx, packet_tokens, _ in raw_packets:
                if flow_id not in flow_dict:
                    flow_dict[flow_id] = []
                flow_dict[flow_id].append((packet_idx, packet_tokens))

            for flow_id, packets in flow_dict.items():
                packets.sort(key=lambda x: x[0])
                packets = packets[:self.packet_count]

                if len(packets) == 0:
                    continue

                flow_tokens = []
                for _, packet_tokens in packets:
                    packet_ids = self._tokens_to_ids(packet_tokens[:self.tokens_per_packet])
                    if len(packet_ids) < self.tokens_per_packet:
                        packet_ids = packet_ids + [self.pad_id] * (self.tokens_per_packet - len(packet_ids))
                    flow_tokens.extend(packet_ids[:self.tokens_per_packet])

                expected_len = self.tokens_per_packet * self.packet_count
                if len(flow_tokens) < expected_len:
                    flow_tokens = flow_tokens + [self.pad_id] * (expected_len - len(flow_tokens))
                elif len(flow_tokens) > expected_len:
                    flow_tokens = flow_tokens[:expected_len]

                flows.append(flow_tokens)


        elapsed = time.time() - start_time
        if self.proc_id == 0 and not self.first_batch_generated:
            pass
            if parse_error_count > 0:
                pass
            if len(flows) == 0 and len(raw_packets) == 0:
                pass
        

        return flows
    
    def _apply_packet_masking(self, tokens):

        expected_len = self.tokens_per_packet * self.packet_count  
        if len(tokens) < expected_len:
            tokens = tokens + [self.pad_id] * (expected_len - len(tokens))
        elif len(tokens) > expected_len:
            tokens = tokens[:expected_len]
        
        masked_tokens = tokens.copy()
        labels = [-1] * len(tokens)  
        

        cns_id = self.vocab.get('[CNS]', self.pad_id)
        header_size = 40
        payload_size = 40
        

        use_hhhh_relation = random.random() < 0.5
        
        if use_hhhh_relation:


            for packet_idx in range(self.packet_count):
                packet_start = packet_idx * self.tokens_per_packet
                payload_start = packet_start + header_size
                payload_end = packet_start + self.tokens_per_packet
                
                for i in range(payload_start, payload_end):
                    if i < len(tokens):
                        masked_tokens[i] = cns_id
                        labels[i] = -1  
            

            selected_packet_idx = random.randint(0, self.packet_count - 1)
            packet_start = selected_packet_idx * self.tokens_per_packet
            header_start = packet_start
            header_end = packet_start + header_size
            

            header_positions = [i for i in range(header_start, header_end) 
                              if i < len(tokens) and tokens[i] != self.pad_id]
            
            if len(header_positions) > 0:

                num_to_mask = max(1, int(len(header_positions) * 0.90))
                mask_positions = random.sample(header_positions, min(num_to_mask, len(header_positions)))
                
                for pos in mask_positions:
                    if pos < len(tokens) and tokens[pos] != self.pad_id:

                        labels[pos] = tokens[pos]
                        

                        rand = random.random()
                        if rand < 0.8:
                            masked_tokens[pos] = self.mask_id
                        elif rand < 0.9:
                            masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)

        else:


            for packet_idx in range(self.packet_count):
                packet_start = packet_idx * self.tokens_per_packet
                header_start = packet_start
                header_end = packet_start + header_size
                
                for i in range(header_start, header_end):
                    if i < len(tokens):
                        masked_tokens[i] = cns_id
                        labels[i] = -1  
            

            selected_packet_idx = random.randint(0, self.packet_count - 1)
            packet_start = selected_packet_idx * self.tokens_per_packet
            payload_start = packet_start + header_size
            payload_end = packet_start + self.tokens_per_packet
            

            payload_positions = [i for i in range(payload_start, payload_end) 
                               if i < len(tokens) and tokens[i] != self.pad_id]
            
            if len(payload_positions) > 0:

                num_to_mask = max(1, int(len(payload_positions) * 0.90))
                mask_positions = random.sample(payload_positions, min(num_to_mask, len(payload_positions)))
                
                for pos in mask_positions:
                    if pos < len(tokens) and tokens[pos] != self.pad_id:

                        labels[pos] = tokens[pos]
                        

                        rand = random.random()
                        if rand < 0.8:
                            masked_tokens[pos] = self.mask_id
                        elif rand < 0.9:
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
        payload_size = 40
        
        header_start = 0
        header_end = header_size
        payload_start = header_size
        payload_end = self.tokens_per_packet
        
        if relation_type == 'hh':


            for i in range(payload_start, payload_end):
                if i < len(packet_tokens):
                    masked_tokens[i] = cns_id
                    labels[i] = -1  
            

            header_positions = [i for i in range(header_start, header_end) 
                              if i < len(packet_tokens) and packet_tokens[i] != self.pad_id]
            num_to_mask = max(1, int(len(header_positions) * 0.80))
            mask_positions = random.sample(header_positions, min(num_to_mask, len(header_positions))) if len(header_positions) > 0 else []
            
            for pos in mask_positions:
                if pos < len(packet_tokens) and packet_tokens[pos] != self.pad_id:

                    labels[pos] = packet_tokens[pos]
                    

                    rand = random.random()
                    if rand < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif rand < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)

        
        elif relation_type == 'pp':


            for i in range(header_start, header_end):
                if i < len(packet_tokens):
                    masked_tokens[i] = cns_id
                    labels[i] = -1  
            

            payload_positions = [i for i in range(payload_start, payload_end) 
                               if i < len(packet_tokens) and packet_tokens[i] != self.pad_id]
            num_to_mask = max(1, int(len(payload_positions) * 0.80))
            mask_positions = random.sample(payload_positions, min(num_to_mask, len(payload_positions))) if len(payload_positions) > 0 else []
            
            for pos in mask_positions:
                if pos < len(packet_tokens) and packet_tokens[pos] != self.pad_id:

                    labels[pos] = packet_tokens[pos]
                    

                    rand = random.random()
                    if rand < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif rand < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)

        
        elif relation_type == 'hp':



            header_positions = [i for i in range(header_start, header_end) 
                              if i < len(packet_tokens) and packet_tokens[i] != self.pad_id]
            num_to_mask = max(1, int(len(header_positions) * 0.95))
            mask_positions = random.sample(header_positions, min(num_to_mask, len(header_positions))) if len(header_positions) > 0 else []
            
            for pos in mask_positions:
                if pos < len(packet_tokens) and packet_tokens[pos] != self.pad_id:

                    labels[pos] = packet_tokens[pos]
                    

                    rand = random.random()
                    if rand < 0.8:
                        masked_tokens[pos] = self.mask_id
                    elif rand < 0.9:
                        masked_tokens[pos] = random.randint(1, len(self.vocab) - 1)

        
        return masked_tokens, labels
    
    def _create_nsp_indicating_sample(self, flow):

        expected_len = self.tokens_per_packet * self.packet_count  
        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]
        

        masked_tokens, labels = self._apply_packet_masking(flow)
        
        packets = [
            masked_tokens[i * self.tokens_per_packet:(i + 1) * self.tokens_per_packet]
            for i in range(self.packet_count)
        ]
        label_packets = [
            labels[i * self.tokens_per_packet:(i + 1) * self.tokens_per_packet]
            for i in range(self.packet_count)
        ]
        src, seg = build_flow_sequence(
            packets, self.cls_id, self.sep_id, self.pad_id, self.seq_length,
            packet_count=self.packet_count, packet_size=self.tokens_per_packet
        )
        tgt_mlm = build_flow_labels(
            label_packets, self.seq_length,
            packet_count=self.packet_count, packet_size=self.tokens_per_packet
        )
        
        return src, seg, tgt_mlm
    
    def _create_mfr_masking_sample(self, flow, packet_idx, relation_type):

        expected_len = self.tokens_per_packet * self.packet_count  
        if len(flow) < expected_len:
            flow = flow + [self.pad_id] * (expected_len - len(flow))
        elif len(flow) > expected_len:
            flow = flow[:expected_len]
        

        packet_start = packet_idx * self.tokens_per_packet
        packet_end = packet_start + self.tokens_per_packet
        packet_tokens = flow[packet_start:packet_end]
        

        masked_tokens, labels = self._apply_mfr_masking_single_packet(packet_tokens, relation_type)
        
        packets = [[self.pad_id] * self.tokens_per_packet for _ in range(self.packet_count)]
        label_packets = [[-1] * self.tokens_per_packet for _ in range(self.packet_count)]
        packets[packet_idx] = masked_tokens
        label_packets[packet_idx] = labels

        src, seg = build_flow_sequence(
            packets, self.cls_id, self.sep_id, self.pad_id, self.seq_length,
            packet_count=self.packet_count, packet_size=self.tokens_per_packet
        )
        tgt_mlm = build_flow_labels(
            label_packets, self.seq_length,
            packet_count=self.packet_count, packet_size=self.tokens_per_packet
        )
        
        return src, seg, tgt_mlm
    
    def __iter__(self):

        if self.proc_id == 0:
            pass
        
        iter_count = 0
        while True:

            if self._is_buffer_empty():
                if self.proc_id == 0 and iter_count == 0:
                    pass
                    
                self._fill_buffer()
                
                if len(self.buffer) == 0:
                    break
                


                batch_instances = self.buffer[self.buffer_start:min(self.buffer_start + self.batch_size * 10, self.buffer_end)]
                self.buffer_start += len(batch_instances)
                
                if self.proc_id == 0 and iter_count == 0:
                    pass
                

                current_flows = self._organize_buffer_data(batch_instances)
                
                if self.proc_id == 0 and iter_count == 0:
                    pass
            else:

                if self.buffer_start >= self.buffer_end:
                    continue
                
                batch_instances = self.buffer[self.buffer_start:min(self.buffer_start + self.batch_size * 10, self.buffer_end)]
                self.buffer_start += len(batch_instances)
                

                current_flows = self._organize_buffer_data(batch_instances)
            
            if len(current_flows) == 0:
                continue
            

            for batch_start_idx in range(0, len(current_flows), self.batch_size):
                batch_end_idx = min(batch_start_idx + self.batch_size, len(current_flows))
                batch_flows = current_flows[batch_start_idx:batch_end_idx]
                
                if len(batch_flows) == 0:
                    continue
                
                if self.proc_id == 0 and iter_count == 0:
                    pass
                

                batch_src = []
                batch_seg = []
                batch_tgt_mlm = []
                batch_task_type = []
                


                for flow in batch_flows:
                    src, seg, tgt_mlm = self._create_nsp_indicating_sample(flow)
                    batch_src.append(src)
                    batch_seg.append(seg)
                    batch_tgt_mlm.append(tgt_mlm)
                    batch_task_type.append('nsp_indicating')
                



                num_flows = len(batch_flows)
                total_packets = num_flows * self.packet_count
                num_hh = max(1, int(total_packets * self.hh_ratio))
                num_pp = max(1, int(total_packets * self.pp_ratio))
                num_hp = total_packets - num_hh - num_pp
                

                packet_list = [(flow_idx, pkt_idx) for flow_idx in range(num_flows) for pkt_idx in range(self.packet_count)]
                random.shuffle(packet_list)
                hh_packets = set(packet_list[:num_hh])
                pp_packets = set(packet_list[num_hh:num_hh + num_pp])
                hp_packets = set(packet_list[num_hh + num_pp:])
                

                for flow_idx, flow in enumerate(batch_flows):
                    for packet_idx in range(self.packet_count):

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
                

                if len(batch_src) > 0:

                    if not self.first_batch_generated and self.proc_id == 0:
                        num_indicating = len(batch_flows)
                        num_mfr_masking = len(batch_flows) * self.packet_count
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
            sub_size = end_idx - start_idx


            output = model.module.base_model(src_sub, seg_sub) if hasattr(model, 'module') else model.base_model(src_sub, seg_sub)


            task_groups = {'nsp_indicating': [], 'mfr_masking': []}
            for i, task_type in enumerate(task_types_sub):
                task_groups[task_type].append(i)


            if len(task_groups['nsp_indicating']) > 0:
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
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss


                self.task_stats['nsp_indicating']['loss'] += loss.item() * len(indices)
                self.task_stats['nsp_indicating']['correct'] += correct.item()
                self.task_stats['nsp_indicating']['num_samples'] += len(indices)
                self.task_stats['nsp_indicating']['num_tokens'] += num_masked_tokens.item()


            if len(task_groups['mfr_masking']) > 0:
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
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss


                self.task_stats['mfr_masking']['loss'] += loss.item() * len(indices)
                self.task_stats['mfr_masking']['correct'] += correct.item()
                self.task_stats['mfr_masking']['num_samples'] += len(indices)
                self.task_stats['mfr_masking']['num_tokens'] += num_masked_tokens.item()


            del output, src_sub, seg_sub, tgt_mlm_sub
            torch.cuda.empty_cache()


        if total_loss is None:
            total_loss = torch.tensor(0.0, device=src.device, requires_grad=True)

        return total_loss / self.accumulation_steps
    
    def _get_global_task_stats(self, device):

        task_order = ['nsp_indicating', 'mfr_masking']
        local_values = []
        for task_name in task_order:
            stats = self.task_stats[task_name]
            local_values.extend([
                stats['loss'],
                stats['correct'],
                stats['num_samples'],
                stats['num_tokens'],
            ])

        stats_tensor = torch.tensor(local_values, dtype=torch.float64, device=device)
        if self.dist_train:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        global_stats = {}
        offset = 0
        for task_name in task_order:
            global_stats[task_name] = {
                'loss': stats_tensor[offset].item(),
                'correct': stats_tensor[offset + 1].item(),
                'num_samples': stats_tensor[offset + 2].item(),
                'num_tokens': stats_tensor[offset + 3].item(),
            }
            offset += 4

        return global_stats

    def report_and_reset_stats(self, rank=None, device=None):

        elapsed = time.time() - self.start_time
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        global_stats = self._get_global_task_stats(device)

        if not self.dist_train or rank == 0:
            mode = f"global_avg({self.world_size} GPUs)" if self.dist_train else "single"
            pass


            task_order = ['nsp_indicating', 'mfr_masking']
            for task_name in task_order:
                stats = global_stats[task_name]


                if stats['num_samples'] > 0:
                    avg_loss = stats['loss'] / stats['num_samples']
                    if stats['num_tokens'] > 0:
                        accuracy = stats['correct'] / stats['num_tokens'] * 100  
                    else:
                        accuracy = 0.0  
                    pass
                else:
                    pass

            pass


        for task_name in self.task_stats:
            self.task_stats[task_name] = {'loss': 0.0, 'correct': 0.0, 'num_samples': 0.0, 'num_tokens': 0.0}

        self.start_time = time.time()
    
    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):

        model.train()


        self.optimizer = optimizer
        self.scheduler = scheduler

        if rank == 0:
            pass
        
        batch_count = 0
        for batch in loader:
            if self.current_step > self.total_steps:
                break
            

            if batch_count == 0 and rank == 0:
                pass
            

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


            if self.current_step % self.report_steps == 0:
                report_device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
                self.report_and_reset_stats(rank=rank, device=report_device)
            

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
                if rank == 0:
                    pass
            
            self.current_step += 1
            batch_count += 1



def build_multi_task_model(args, log_prefix=""):

    def log(message):
        pass


    log("Building base model (embedding + 4-layer PacketAttention + Pooling + 8-layer FlowAttention)...")
    embedding = str2embedding[args.embedding](args, len(args.vocab))


    log("Building attention stacks...")
    packet_size = getattr(args, 'tokens_per_packet', 80)
    header_size = getattr(args, 'header_size', 40)
    payload_size = packet_size - header_size
    window_size = getattr(args, 'window_size', 20)


    packet_attention_stack = PacketAttentionStack(
        num_layers=4,
        hidden_size=args.hidden_size,
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        header_size=header_size,
        packet_size=packet_size,
        window_size=window_size,
        dropout=getattr(args, 'dropout', 0.1)
    )


    pooling_layer = PoolingLayer(
        header_size=header_size,
        payload_size=payload_size
    )


    flow_attention_stack = FlowAttentionStack(
        num_layers=8,
        hidden_size=args.hidden_size,
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        header_size=header_size,
        payload_size=payload_size,
        dropout=getattr(args, 'dropout', 0.1)
    )


    base_model = BaseModel(
        embedding,
        packet_attention_stack=packet_attention_stack,
        pooling_layer=pooling_layer,
        flow_attention_stack=flow_attention_stack,
        packet_size=packet_size,
        header_size=header_size
    )


    log("Building multi-task model...")
    model = MultiTaskModel(base_model, args, len(args.vocab))


    optimizer_state_dict = None
    scheduler_state_dict = None
    if args.pretrained_model_path is not None:
        log(f"Loading pretrained model from {args.pretrained_model_path}...")

        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')


        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            log("Loading full checkpoint (model + optimizer + scheduler)...")
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
        log(f"Loaded {len(pretrained_dict_filtered)} parameters from pretrained model")
    else:
        log("Initializing model with normal distribution...")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    return model, optimizer_state_dict, scheduler_state_dict

def train_and_validate_multi_task(args):

    import time
    
    set_seed(args.seed)
    
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
    
    pass

    model = None
    optimizer_state_dict = None
    scheduler_state_dict = None
    if args.dist_train:
        pass
    else:
        model, optimizer_state_dict, scheduler_state_dict = build_multi_task_model(args)

    pass


    if args.dist_train:

        pass
        mp.spawn(
            worker_multi_task,
            nprocs=args.ranks_num,
            args=(args.gpu_ranks, args, model, optimizer_state_dict, scheduler_state_dict),
            daemon=False
        )
    elif args.single_gpu:

        pass
        worker_multi_task(args.gpu_id, None, args, model, optimizer_state_dict, scheduler_state_dict)
    else:

        pass
        worker_multi_task(None, None, args, model, optimizer_state_dict, scheduler_state_dict)


def worker_multi_task(local_rank, gpu_ranks, args, model, optimizer_state_dict=None, scheduler_state_dict=None):
    try:
        return _worker_multi_task_impl(local_rank, gpu_ranks, args, model, optimizer_state_dict, scheduler_state_dict)
    except BaseException as exc:
        pass
        raise


def _worker_multi_task_impl(local_rank, gpu_ranks, args, model, optimizer_state_dict=None, scheduler_state_dict=None):



    gpu_id = local_rank
    rank = 0
    if args.dist_train:
        if gpu_ranks is None or local_rank >= len(gpu_ranks):
            raise ValueError(
                f"Invalid distributed local_rank={local_rank}; gpu_ranks={gpu_ranks}"
            )
        gpu_id = gpu_ranks[local_rank]
        rank = local_rank

    pass

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        pass

    if model is None:
        model, optimizer_state_dict, scheduler_state_dict = build_multi_task_model(
            args, log_prefix=f"[Rank {rank}] "
        )

    if args.dist_train:
        pass
        dist.init_process_group(
            backend=args.backend,
            init_method=args.master_ip,
            world_size=args.world_size,
            rank=rank
        )
        pass
        model.cuda(gpu_id)

        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=False)
        pass
        proc_id = rank
        proc_num = args.world_size
    elif args.single_gpu:
        pass
        model.cuda(gpu_id)
        proc_id = 0
        proc_num = 1
    else:
        proc_id = 0
        proc_num = 1


    loader = MultiTaskDataset(
        args,
        args.dataset_path,
        args.batch_size,
        proc_id,
        proc_num,
        shuffle=True
    )


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
            if proc_id == 0:
                pass
        except Exception as e:
            if proc_id == 0:
                pass

    if scheduler_state_dict is not None:
        try:
            scheduler.load_state_dict(scheduler_state_dict)
            if proc_id == 0:
                pass
        except Exception as e:
            if proc_id == 0:
                pass
    

    trainer = MultiTaskTrainer(args)


    if proc_id == 0:
        pass
        if hasattr(args, 'checkpoint_step') and args.checkpoint_step > 0:s
        pass
    
    trainer.train(args, gpu_id, rank, loader, model, optimizer, scheduler)


    if proc_id == 0:
        final_checkpoint = {
            'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'step': trainer.current_step,
        }
        torch.save(final_checkpoint, args.output_model_path)
