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

        return output, attention_probs

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
        header_probs = []
        payload_probs = []

        for i in range(4):
            if i == 0:
                header_out, hp = self.apply_attention(
                    all_headers[0], [all_headers[0], all_headers[1]], is_header=True
                )
                payload_out, pp = self.apply_attention(
                    all_pooled_payloads[0], [all_pooled_payloads[0], all_pooled_payloads[1]], is_header=False
                )
            elif i == 1:
                header_out, hp = self.apply_attention(
                    all_headers[1], [all_headers[0], all_headers[1], all_headers[2]], is_header=True
                )
                payload_out, pp = self.apply_attention(
                    all_pooled_payloads[1], [all_pooled_payloads[0], all_pooled_payloads[1], all_pooled_payloads[2]], is_header=False
                )
            elif i == 2:
                header_out, hp = self.apply_attention(
                    all_headers[2], [all_headers[1], all_headers[2], all_headers[3]], is_header=True
                )
                payload_out, pp = self.apply_attention(
                    all_pooled_payloads[2], [all_pooled_payloads[1], all_pooled_payloads[2], all_pooled_payloads[3]], is_header=False
                )
            else:
                header_out, hp = self.apply_attention(
                    all_headers[3], [all_headers[2], all_headers[3]], is_header=True
                )
                payload_out, pp = self.apply_attention(
                    all_pooled_payloads[3], [all_pooled_payloads[2], all_pooled_payloads[3]], is_header=False
                )


            header_final = self.header_norm(all_headers[i] + header_out)


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
    def __init__(self, args):
        super(TwoLevelAttentionEncoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_attention_heads = getattr(args, 'num_attention_heads', 8)
        self.packet_size = getattr(args, 'packet_size', 80)
        self.header_size = getattr(args, 'header_size', 40)
        self.payload_size = self.packet_size - self.header_size
        self.window_size = getattr(args, 'window_size', 20)


        self.packet_attention_stack = PacketAttentionStack(
            num_layers=4,
            hidden_size=args.hidden_size,
            num_attention_heads=self.num_attention_heads,
            header_size=self.header_size,
            packet_size=self.packet_size,
            window_size=self.window_size,
            dropout=getattr(args, 'dropout', 0.1)
        )


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


        self.packet_attention_probs = None
        self.flow_attention_probs = None

    def forward(self, hidden_states, seg):
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)


        all_packet_hiddens = []
        packet_start_positions = [1, 82, 163, 244]

        for i in range(4):
            start_pos = packet_start_positions[i]
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


        packet_outputs = []
        for i in range(4):
            packet_output = self.packet_attention_stack(all_packet_hiddens[i])
            packet_outputs.append(packet_output)

        all_packet_hiddens = packet_outputs


        pooled_packets = []
        for packet_hidden in all_packet_hiddens:
            header = packet_hidden[:, :self.header_size, :]
            payload = packet_hidden[:, self.header_size:, :]


            batch_size_p, seq_len_p, hidden_size_p = payload.size()
            payload_reshaped = payload.view(batch_size_p, 4, 10, hidden_size_p)
            pooled_payload = torch.mean(payload_reshaped, dim=2)


            pooled_payload_expanded = pooled_payload.unsqueeze(2).repeat(1, 1, 10, 1)
            pooled_payload_expanded = pooled_payload_expanded.view(batch_size_p, 40, hidden_size_p)

            pooled_packet = torch.cat([header, pooled_payload_expanded], dim=1)
            pooled_packets.append(pooled_packet)


        flow_probs_list = []
        for layer_idx, flow_attention in enumerate(self.flow_attention_stack):
            output_headers, output_payloads, f_probs = flow_attention(pooled_packets)
            flow_probs_list.append(f_probs)


            pooled_packets = []
            for i in range(4):
                packet_output = torch.cat([output_headers[i], output_payloads[i]], dim=1)
                pooled_packets.append(packet_output)

        self.flow_attention_probs = flow_probs_list


        packet_outputs = pooled_packets


        cls_output = hidden_states[:, 0:1, :]
        sep_outputs = [
            hidden_states[:, 81:82, :],
            hidden_states[:, 162:163, :],
            hidden_states[:, 243:244, :],
        ]


        output_parts = [cls_output, packet_outputs[0], sep_outputs[0],
                       packet_outputs[1], sep_outputs[1],
                       packet_outputs[2], sep_outputs[2],
                       packet_outputs[3]]

        output = torch.cat(output_parts, dim=1)


        return output, tuple(flow_probs_list)


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))

        self.encoder = TwoLevelAttentionEncoder(args)

        self.labels_num = args.labels_num

        self.pooling = args.pooling

        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha


        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        batch_size = src.size(0)


        emb = self.embedding(src, seg)

        output, encoder_probs = self.encoder(emb, seg)


        if self.pooling == "mean":
            pooled_output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            pooled_output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            pooled_output = output[:, -1, :]
        else:

            pooled_output = output[:, 0, :]


        classifier_output = torch.tanh(self.output_layer_1(pooled_output))
        logits = self.output_layer_2(classifier_output)


        packet_attention_probs = self.encoder.packet_attention_probs if hasattr(self.encoder, 'packet_attention_probs') else None
        flow_attention_probs = self.encoder.flow_attention_probs if hasattr(self.encoder, 'flow_attention_probs') else None

        all_attention_probs = {
            "encoder": encoder_probs,
            "packet": packet_attention_probs,
            "flow": flow_attention_probs
        }


        if tgt is not None:

            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))

            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits, all_attention_probs

        else:
            return None, logits, all_attention_probs

    def get_attention_weights(self, src, seg):
        batch_size = src.size(0)


        emb = self.embedding(src, seg)

        output, encoder_probs = self.encoder(emb, seg)


        packet_attention_probs = self.encoder.packet_attention_probs if hasattr(self.encoder, 'packet_attention_probs') else None
        flow_attention_probs = self.encoder.flow_attention_probs if hasattr(self.encoder, 'flow_attention_probs') else None

        attention_weights = {
            "encoder": encoder_probs,
            "packet": packet_attention_probs,
            "flow": flow_attention_probs
        }

        return attention_weights


def count_labels_num(path):
    labels_set = set()


    if path.endswith('.pt'):

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
    if args.pretrained_model_path is not None:


        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires GPU training.")


        map_location = {'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}

        model.load_state_dict(
            torch.load(args.pretrained_model_path, map_location=map_location),
            strict=False
        )
    else:

        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
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
    dataset = []


    if path.endswith('.pt'):

        with open(path, 'rb') as f:
            flows = []
            try:
                while True:
                    flow_data = pickle.load(f)
                    flows.append(flow_data)
            except EOFError:
                pass


        for flow_data in flows:
            if len(flow_data) != 5:
                continue

            flow_id, flow_mfr_tokens, category_name, dataset_name, label_id = flow_data


            if not isinstance(flow_mfr_tokens, list) or len(flow_mfr_tokens) != 320:
                continue


            packets = []
            for packet_idx in range(4):
                start_idx = packet_idx * 80
                end_idx = (packet_idx + 1) * 80
                packet_tokens = flow_mfr_tokens[start_idx:end_idx]
                if len(packet_tokens) == 80:
                    packets.append(packet_tokens)


            if len(packets) != 4:
                continue


            cls_id = args.tokenizer.vocab.get(CLS_TOKEN, 2)
            sep_id = args.tokenizer.vocab.get(SEP_TOKEN, 3)
            unk_id = args.tokenizer.vocab.get('[UNK]', 1)


            all_packet_ids = []
            for packet_tokens in packets:
                packet_ids = []
                for hex_token in packet_tokens:
                    token_id = args.tokenizer.vocab.get(str(hex_token), unk_id)
                    packet_ids.append(token_id)
                all_packet_ids.append(packet_ids)


            src = [cls_id]
            seg = [1]


            for packet_idx, packet_ids in enumerate(all_packet_ids):
                src.extend(packet_ids)
                seg.extend([1] * len(packet_ids))
                src.append(sep_id)
                seg.append(1)


            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]


            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)


            dataset.append((src, label_id, seg))

    else:

        columns = {}
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:

                    for i, column_name in enumerate(line.strip().split("\t")):
                        columns[column_name] = i
                    continue

                line = line[:-1].split("\t")
                tgt = int(line[columns["label"]])


                if args.soft_targets and "logits" in columns.keys():
                    soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]


                if "text_b" not in columns:

                    text_a = line[columns["text_a"]]
                    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                    seg = [1] * len(src)
                else:


                    text_a = line[columns["text_a"]]
                    text_b = line[columns["text_b"]]


                    if "text_c" in columns and "text_d" in columns:
                        text_c = line[columns["text_c"]]
                        text_d = line[columns["text_d"]]


                        src = [args.tokenizer.vocab.get(CLS_TOKEN, 2)]
                        seg = [1]


                        for packet_idx, text in enumerate([text_a, text_b, text_c, text_d]):
                            tokens = args.tokenizer.tokenize(text)
                            token_ids = args.tokenizer.convert_tokens_to_ids(tokens)
                            src.extend(token_ids)
                            seg.extend([1] * len(token_ids))
                            src.append(args.tokenizer.vocab.get(SEP_TOKEN, 3))
                            seg.append(1)
                    else:

                        src_a = args.tokenizer.convert_tokens_to_ids(
                            [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                        )
                        src_b = args.tokenizer.convert_tokens_to_ids(
                            args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                        )
                        src = src_a + src_b

                        seg = [1] * (len(src_a) + len(src_b))


                if len(src) > args.seq_length:
                    src = src[: args.seq_length]
                    seg = seg[: args.seq_length]


                while len(src) < args.seq_length:
                    src.append(0)
                    seg.append(0)


                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, soft_tgt))
                else:
                    dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)


    loss, _, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)


    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)


    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False, save_attention_weights=False, dataset_name=None):
    import gc
    gc.collect()

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])


    batch_size = max(1, args.batch_size // 2)

    correct = 0
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)


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


            torch.cuda.empty_cache()


            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item() * src_batch.size(0)
            total_samples += src_batch.size(0)


            if save_attention_weights:
                if attention_probs["encoder"] is not None:
                    encoder_attn = attention_probs["encoder"]

                    if isinstance(encoder_attn, tuple):

                        if i == 0:
                            all_attention_weights["encoder"] = [[] for _ in range(len(encoder_attn))]

                        for layer_idx, layer_attn in enumerate(encoder_attn):
                            if isinstance(layer_attn, torch.Tensor):
                                all_attention_weights["encoder"][layer_idx].append(layer_attn.cpu())
                    elif isinstance(encoder_attn, torch.Tensor):
                        all_attention_weights["encoder"].append(encoder_attn.cpu())

                if attention_probs["packet"] is not None:
                    packet_attn = attention_probs["packet"]


                    if isinstance(packet_attn, list) and len(packet_attn) > 0:
                        if isinstance(packet_attn[0], dict):

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


                            if packet_attn_hh:
                                all_attention_weights["packet"].append({
                                    "hh": torch.stack(packet_attn_hh).cpu(),
                                    "hp": torch.stack(packet_attn_hp).cpu() if packet_attn_hp else None,
                                    "pp": torch.stack(packet_attn_pp).cpu() if packet_attn_pp else None
                                })
                        else:

                            packet_attn = torch.stack(packet_attn)
                            if isinstance(packet_attn, torch.Tensor):
                                all_attention_weights["packet"].append(packet_attn.cpu())

                if attention_probs["flow"] is not None:
                    flow_attn = attention_probs["flow"]

                    if isinstance(flow_attn, dict):

                        flow_dict = {}
                        if "header" in flow_attn and flow_attn["header"]:

                            flow_dict["header"] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in flow_attn["header"]]
                        if "payload" in flow_attn and flow_attn["payload"]:

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


        del src_batch, tgt_batch, seg_batch, loss, logits, attention_probs, pred, gold
        torch.cuda.empty_cache()


        if (i + 1) % 10 == 0:
            gc.collect()


    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0


    eps = 1e-9
    precisions = []
    recalls = []
    f1_scores = []
    correct_counts = []

    for i in range(confusion.size()[0]):

        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        precision = tp / (tp + fp + eps)


        fn = confusion[:, i].sum().item() - tp
        recall = tp / (tp + fn + eps)


        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        correct_counts.append(tp)


    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    if print_confusion_matrix:
        cf_array = confusion.numpy()
        import os
        output_dir = os.path.dirname(args.output_model_path)
        confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.txt")
        with open(confusion_matrix_path, 'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')
        for i in range(confusion.size()[0]):
            pass


    for i in range(confusion.size()[0]):
        total_true = confusion[:, i].sum().item()


    if save_attention_weights:
        import os
        output_dir = os.path.dirname(args.output_model_path)
        os.makedirs(output_dir, exist_ok=True)


        if all_attention_weights["encoder"] and isinstance(all_attention_weights["encoder"][0], list):

            processed_encoder = []
            for layer_attns in all_attention_weights["encoder"]:
                if layer_attns:


                    processed_encoder.append(layer_attns[0])
            all_attention_weights["encoder"] = processed_encoder


        processed_packet = []
        packet_hh_list = []
        packet_hp_list = []
        packet_pp_list = []

        if all_attention_weights["packet"]:
            for packet_item in all_attention_weights["packet"]:
                if isinstance(packet_item, dict):

                    if "hh" in packet_item and packet_item["hh"] is not None:
                        packet_hh_list.append(packet_item["hh"])
                    if "hp" in packet_item and packet_item["hp"] is not None:
                        packet_hp_list.append(packet_item["hp"])
                    if "pp" in packet_item and packet_item["pp"] is not None:
                        packet_pp_list.append(packet_item["pp"])
                elif isinstance(packet_item, torch.Tensor):

                    processed_packet.append(packet_item)


        if all_attention_weights["flow"] and isinstance(all_attention_weights["flow"][0], list):
            processed_flow = []
            for flow_attns in all_attention_weights["flow"]:
                if flow_attns:
                    processed_flow.append(flow_attns[0])
            all_attention_weights["flow"] = processed_flow


        if all_attention_weights["encoder"]:

            encoder_tensor = all_attention_weights["encoder"][0]

        encoder_attention_path = os.path.join(output_dir, "attention_encoder.pt")
        torch.save(all_attention_weights["encoder"], encoder_attention_path)
        if all_attention_weights["encoder"]:
            pass


        if all_attention_weights["flow"]:

            if all_attention_weights["flow"] and isinstance(all_attention_weights["flow"][0], dict):
                first_flow = all_attention_weights["flow"][0]

                if "header" in first_flow and first_flow["header"]:
                    header_tensor = first_flow["header"][0]

                if "payload" in first_flow and first_flow["payload"]:
                    payload_tensor = first_flow["payload"][0]

            flow_attention_path = os.path.join(output_dir, "attention_flow.pt")
            torch.save(all_attention_weights["flow"], flow_attention_path)


            if all_attention_weights["flow"] and isinstance(all_attention_weights["flow"][0], dict):
                first_flow = all_attention_weights["flow"][0]
                if "header" in first_flow:
                    if first_flow["header"]:
                        pass
                if "payload" in first_flow:
                    if first_flow["payload"]:
                        pass
        else:
            pass


        if packet_hh_list:

            hh_tensor = packet_hh_list[0] if isinstance(packet_hh_list[0], torch.Tensor) else packet_hh_list[0]

            packet_hh_path = os.path.join(output_dir, "attention_packet_hh.pt")
            torch.save(packet_hh_list, packet_hh_path)
            if packet_hh_list:
                pass

        if packet_hp_list:

            hp_tensor = packet_hp_list[0] if isinstance(packet_hp_list[0], torch.Tensor) else packet_hp_list[0]

            packet_hp_path = os.path.join(output_dir, "attention_packet_hp.pt")
            torch.save(packet_hp_list, packet_hp_path)

        if packet_pp_list:

            pp_tensor = packet_pp_list[0] if isinstance(packet_pp_list[0], torch.Tensor) else packet_pp_list[0]

            packet_pp_path = os.path.join(output_dir, "attention_packet_pp.pt")
            torch.save(packet_pp_list, packet_pp_path)


    torch.cuda.empty_cache()
    gc.collect()

    return accuracy, avg_loss, confusion, macro_precision, macro_recall, macro_f1


def main():
    import gc
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)


    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")


    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")


    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")


    parser.add_argument("--early_stopping", action='store_true',
                        help="Enable early stopping mechanism.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="Minimum change in the monitored quantity to qualify as an improvement.")


    parser.add_argument("--four_packet_format", action='store_true', default=True,
                        help="Use 4-packet format for input ([CLS]+p1+[SEP]+p2+[SEP]+p3+[SEP]+p4, each packet 80 tokens).")


    parser.add_argument("--use_packet_attention", action='store_true', default=True,
                        help="Enable packet-level attention mechanism.")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Window size for payload-payload window attention.")


    parser.add_argument("--use_flow_attention", action='store_true', default=True,
                        help="Enable flow-level attention mechanism.")


    parser.add_argument("--packet_size", type=int, default=80,
                        help="Number of tokens per packet.")
    parser.add_argument("--header_size", type=int, default=40,
                        help="Number of tokens in packet header.")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads for both packet and flow attention.")

    args = parser.parse_args()


    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU training.")

    for i in range(torch.cuda.device_count()):
        pass


    args = load_hyperparam(args)


    set_seed(args.seed)


    args.labels_num = count_labels_num(args.train_path)


    args.tokenizer = str2tokenizer[args.tokenizer](args)


    model = Classifier(args)


    load_or_initialize_parameters(args, model)


    args.device = torch.device("cuda:0")
    model = model.to(args.device)


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


    if args.use_packet_attention:
        pass


    if args.use_flow_attention:
        pass


    optimizer, scheduler = build_optimizer(args, model)


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp


    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0


    if args.early_stopping:
        patience_counter = 0
    else:
        patience_counter = None


    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()

        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                avg_loss = total_loss / args.report_steps
                total_loss = 0.0


                if (i + 1) % (args.report_steps * 5) == 0:
                    gc.collect()


        result = evaluate(args, read_dataset(args, args.dev_path))
        current_acc = result[0]
        current_loss = result[1]


        if current_acc > best_result + args.min_delta:
            best_result = current_acc
            save_model(model, args.output_model_path)
            if args.early_stopping:
                patience_counter = 0
        else:
            if args.early_stopping:
                patience_counter += 1

                if patience_counter >= args.patience:
                    break


    if args.test_path is not None:
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))


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


if __name__ == "__main__":
    main()

