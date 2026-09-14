import os
import random
import pickle
import torch
from collections import defaultdict
from uer.utils.constants import *

CLS_ID = 2
SEP_ID = 3
PAD_ID = 0


class PacketPairDataLoaderV3(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.diff_flow_ratio = 0.2
        self.seq_length = args.seq_length if hasattr(args, 'seq_length') else 320
        self.dataset_flow_packets = defaultdict(lambda: defaultdict(list))
        self.dataset_all_packets = defaultdict(list)
        self.dataset_flow_adjacent_map = defaultdict(lambda: defaultdict(dict))
        self.datasets = set()

    def _load_vocab(self, vocab_path):
        token2id = {}
        if vocab_path and os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        token = line.strip()
                        if token:
                            token2id[token] = idx
            except Exception:
                token2id = self._create_default_vocab()
        else:
            token2id = self._create_default_vocab()
        return token2id

    def _create_default_vocab(self):
        token2id = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        for i in range(65536):
            token2id[f'{i:04x}'] = i + 5
        return token2id

    def _load_and_organize_packets(self):
        dataset_flow_packets = defaultdict(lambda: defaultdict(list))
        dataset_all_packets = defaultdict(list)
        dataset_flow_adjacent_map = defaultdict(lambda: defaultdict(dict))
        datasets = set()
        packet_count = 0

        try:
            with open(self.dataset_path, 'rb') as f:
                while True:
                    try:
                        instance = pickle.load(f)
                        if len(instance) == 4:
                            flow_id, packet_idx, packet_tokens, _ = instance
                        elif len(instance) == 3:
                            flow_id, packet_idx, packet_tokens = instance
                        else:
                            continue
                        dataset_name = ""

                        if not isinstance(flow_id, str):
                            flow_id = str(flow_id)
                        if not isinstance(packet_idx, int):
                            packet_idx = int(packet_idx)
                        if not isinstance(packet_tokens, list):
                            packet_tokens = list(packet_tokens)
                        packet_tokens = [str(x) for x in packet_tokens]

                        dataset_flow_packets[dataset_name][flow_id].append(
                            (packet_idx, packet_tokens)
                        )
                        dataset_all_packets[dataset_name].append(
                            (flow_id, packet_idx, packet_tokens)
                        )
                        datasets.add(dataset_name)
                        packet_count += 1
                    except EOFError:
                        break
                    except Exception:
                        continue
        except Exception:
            return False

        for dataset_name in dataset_flow_packets:
            for flow_id in dataset_flow_packets[dataset_name]:
                flow_packets = dataset_flow_packets[dataset_name][flow_id]
                for idx, tokens in flow_packets:
                    for adj_idx, adj_tokens in flow_packets:
                        if abs(adj_idx - idx) == 1:
                            dataset_flow_adjacent_map[dataset_name][flow_id][idx] = (
                                adj_idx,
                                adj_tokens
                            )
                            break

        self.dataset_flow_packets = dataset_flow_packets
        self.dataset_all_packets = dataset_all_packets
        self.dataset_flow_adjacent_map = dataset_flow_adjacent_map
        self.datasets = datasets
        return len(datasets) > 0 and packet_count > 0

    def _get_adjacent_packets(self, flow_packets, packet_idx):
        for idx, tokens in flow_packets:
            if abs(idx - packet_idx) == 1:
                return idx, tokens
        return None

    def _create_packet_pair(self):
        if len(self.datasets) == 0:
            return None

        dataset_name = random.choice(list(self.datasets))
        if dataset_name not in self.dataset_all_packets or \
                len(self.dataset_all_packets[dataset_name]) < 2:
            return None

        if random.random() < self.diff_flow_ratio:
            label = 0
            dataset_flows = self.dataset_flow_packets[dataset_name]
            adjacent_map = self.dataset_flow_adjacent_map[dataset_name]

            if len(dataset_flows) < 2:
                return self._create_packet_pair()

            flows_with_neighbors = [
                flow_id for flow_id in dataset_flows
                if len(adjacent_map[flow_id]) > 0
            ]
            if len(flows_with_neighbors) < 2:
                return self._create_packet_pair()

            for _ in range(5):
                flow_id1, flow_id2 = random.sample(flows_with_neighbors, 2)
                packets_with_neighbors1 = list(adjacent_map[flow_id1].keys())
                if not packets_with_neighbors1:
                    continue
                packet_idx1 = random.choice(packets_with_neighbors1)
                _, packet_tokens1 = adjacent_map[flow_id1][packet_idx1]

                packets_with_neighbors2 = list(adjacent_map[flow_id2].keys())
                if not packets_with_neighbors2:
                    continue
                packet_idx2 = random.choice(packets_with_neighbors2)
                _, packet_tokens2 = adjacent_map[flow_id2][packet_idx2]
                break
            else:
                return self._create_packet_pair()
        else:
            dataset_flows = self.dataset_flow_packets[dataset_name]
            adjacent_map = self.dataset_flow_adjacent_map[dataset_name]
            flows_with_neighbors = [
                flow_id for flow_id in dataset_flows
                if len(adjacent_map[flow_id]) > 0
            ]
            if not flows_with_neighbors:
                return self._create_packet_pair()

            for _ in range(5):
                flow_id1 = random.choice(flows_with_neighbors)
                packets_with_neighbors = list(adjacent_map[flow_id1].keys())
                if not packets_with_neighbors:
                    continue

                packet_idx1 = random.choice(packets_with_neighbors)
                packet_idx2, packet_tokens2 = adjacent_map[flow_id1][packet_idx1]
                packet_tokens1 = None
                for idx, tokens in dataset_flows[flow_id1]:
                    if idx == packet_idx1:
                        packet_tokens1 = tokens
                        break
                if packet_tokens1 is not None:
                    break
            else:
                return self._create_packet_pair()

            label = 1 if packet_idx1 < packet_idx2 else 2

        max_len_per_packet = (self.seq_length - 4) // 2
        packet_tokens1 = packet_tokens1[:max_len_per_packet]
        packet_tokens2 = packet_tokens2[:max_len_per_packet]

        src = [CLS_ID]
        for hex_token in packet_tokens1:
            try:
                token_id = int(hex_token, 16) + 5
            except (ValueError, TypeError):
                token_id = 1
            src.append(token_id)

        src.append(SEP_ID)
        for hex_token in packet_tokens2:
            try:
                token_id = int(hex_token, 16) + 5
            except (ValueError, TypeError):
                token_id = 1
            src.append(token_id)

        src.append(SEP_ID)
        src.extend([PAD_ID] * max(0, self.seq_length - len(src)))
        return src[:self.seq_length], label

    def __iter__(self):
        batch_load_size = 50000

        try:
            with open(self.dataset_path, 'rb') as f:
                while True:
                    self.dataset_flow_packets.clear()
                    self.dataset_all_packets.clear()
                    self.datasets.clear()

                    packet_count = 0
                    batch_has_data = False

                    while packet_count < batch_load_size:
                        try:
                            instance = pickle.load(f)
                            batch_has_data = True

                            if len(instance) == 4:
                                flow_id, packet_idx, packet_tokens, _ = instance
                            elif len(instance) == 3:
                                flow_id, packet_idx, packet_tokens = instance
                            else:
                                continue
                            dataset_name = ""

                            if not isinstance(flow_id, str):
                                flow_id = str(flow_id)
                            if not isinstance(packet_idx, int):
                                packet_idx = int(packet_idx)
                            if not isinstance(packet_tokens, list):
                                packet_tokens = list(packet_tokens)
                            packet_tokens = [str(x) for x in packet_tokens]

                            self.dataset_flow_packets[dataset_name][flow_id].append(
                                (packet_idx, packet_tokens)
                            )
                            self.dataset_all_packets[dataset_name].append(
                                (flow_id, packet_idx, packet_tokens)
                            )
                            self.datasets.add(dataset_name)
                            packet_count += 1
                        except EOFError:
                            break
                        except Exception:
                            continue

                    if not batch_has_data or packet_count == 0:
                        break

                    for dataset_name in self.dataset_flow_packets:
                        for flow_id in self.dataset_flow_packets[dataset_name]:
                            flow_packets = self.dataset_flow_packets[dataset_name][flow_id]
                            self.dataset_flow_adjacent_map[dataset_name][flow_id].clear()
                            for idx, tokens in flow_packets:
                                for adj_idx, adj_tokens in flow_packets:
                                    if abs(adj_idx - idx) == 1:
                                        self.dataset_flow_adjacent_map[dataset_name][flow_id][idx] = (
                                            adj_idx,
                                            adj_tokens
                                        )
                                        break

                    batch_src = []
                    batch_label = []
                    target_pairs_in_batch = batch_load_size * 10
                    pairs_generated_in_batch = 0

                    while pairs_generated_in_batch < target_pairs_in_batch:
                        pair = self._create_packet_pair()
                        if pair is None:
                            continue

                        src, label = pair
                        try:
                            src = [int(x) for x in src]
                        except (ValueError, TypeError):
                            continue

                        batch_src.append(src)
                        batch_label.append(label)
                        pairs_generated_in_batch += 1

                        if len(batch_src) >= self.batch_size:
                            yield (
                                torch.tensor(batch_src, dtype=torch.long),
                                torch.tensor(batch_label, dtype=torch.long)
                            )
                            batch_src = []
                            batch_label = []

                    if batch_src:
                        yield (
                            torch.tensor(batch_src, dtype=torch.long),
                            torch.tensor(batch_label, dtype=torch.long)
                        )
        except Exception:
            return


PacketPairDataLoaderV2 = PacketPairDataLoaderV3
PacketPairDataLoader = PacketPairDataLoaderV3
PacketPairDataset = None
