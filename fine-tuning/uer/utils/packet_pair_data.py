"""
数据包对预测任务的数据加载器 (V3版本 - 支持数据集分组)
支持从 (flow_id, packet_index, packet_tokens, dataset_name) 格式的数据动态生成数据包对
确保数据包对只能在同一个数据集内生成（不跨数据集）

数据包对生成规则:
- 20%: 来自不同流（但同一数据集）-> label = 0
- 80%: 来自同一流（必然同一数据集）
    - 50% packet1在前 (packet_index1 < packet_index2) -> label = 1
    - 50% packet1在后 (packet_index1 > packet_index2) -> label = 2
"""
import os
import random
import pickle
import torch
from collections import defaultdict
from uer.utils.constants import *

# 确保CLS_ID, SEP_ID, PAD_ID有默认值
CLS_ID = 2
SEP_ID = 3
PAD_ID = 0


class PacketPairDataLoaderV3(object):
    """
    数据包对数据加载器 V3
    支持从原始数据格式 (flow_id, packet_index, packet_tokens, dataset_name) 动态生成数据包对
    确保数据包对只能在同一数据集内生成
    """
    
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
        
        # 数据包对生成参数
        self.diff_flow_ratio = 0.2  # 20% 来自不同流
        self.seq_length = args.seq_length if hasattr(args, 'seq_length') else 320
        
        # 用于生成数据包对的缓存（按数据集和流ID组织）
        # 结构: {dataset_name: {flow_id: [(packet_index, packet_tokens), ...]}}
        self.dataset_flow_packets = defaultdict(lambda: defaultdict(list))
        # 用于快速访问数据集内的所有数据包
        # 结构: {dataset_name: [(flow_id, packet_index, packet_tokens), ...]}
        self.dataset_all_packets = defaultdict(list)
        self.datasets = set()  # 所有数据集名称
        
        # 不需要加载token2id字典，因为使用直接计算而不是字典查询
        # Token转换采用数学计算: int(hex_string, 16) + 5
        # 这样避免了大型字典查询的性能开销
    
    def _load_vocab(self, vocab_path):
        """加载词汇表，创建token到ID的映射"""
        token2id = {}
        if vocab_path and os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        token = line.strip()
                        if token:
                            token2id[token] = idx
                print(f"[PacketPairDataLoaderV3] 加载词汇表: {len(token2id)} 个tokens")
            except Exception as e:
                print(f"[PacketPairDataLoaderV3] 警告: 无法加载词汇表 {vocab_path}: {e}")
                # 创建默认映射（特殊tokens + 十六进制tokens）
                token2id = self._create_default_vocab()
        else:
            # 创建默认映射
            token2id = self._create_default_vocab()
        
        return token2id
    
    def _create_default_vocab(self):
        """创建默认词汇表（特殊tokens + 十六进制tokens）"""
        token2id = {}
        # 特殊tokens
        token2id['[PAD]'] = 0
        token2id['[UNK]'] = 1
        token2id['[CLS]'] = 2
        token2id['[SEP]'] = 3
        token2id['[MASK]'] = 4
        
        # 十六进制tokens (0000-ffff)
        for i in range(65536):
            hex_token = f'{i:04x}'
            token2id[hex_token] = i + 5
        
        print(f"[PacketPairDataLoaderV3] 创建默认词汇表: {len(token2id)} 个tokens")
        return token2id
    
    def _load_and_organize_packets(self):
        """加载并按数据集和流ID组织所有数据包"""
        print(f"[PacketPairDataLoaderV3] 正在加载数据包...")
        
        dataset_flow_packets = defaultdict(lambda: defaultdict(list))
        dataset_all_packets = defaultdict(list)
        datasets = set()
        packet_count = 0
        
        try:
            with open(self.dataset_path, 'rb') as f:
                while True:
                    try:
                        instance = pickle.load(f)
                        
                        # 支持4元组格式（新）和3元组格式（旧）
                        if len(instance) == 4:
                            flow_id, packet_idx, packet_tokens, dataset_name = instance
                        elif len(instance) == 3:
                            flow_id, packet_idx, packet_tokens = instance
                            dataset_name = "Unknown"
                        else:
                            continue
                        
                        # 验证数据格式
                        if not isinstance(flow_id, str):
                            flow_id = str(flow_id)
                        if not isinstance(packet_idx, int):
                            packet_idx = int(packet_idx)
                        if not isinstance(packet_tokens, list):
                            packet_tokens = list(packet_tokens)
                        
                        # 保持packet_tokens为十六进制字符串格式
                        # 例如: ['4500', '0020', 'd933', ...]
                        try:
                            # 验证所有tokens都是有效的字符串
                            packet_tokens = [str(x) for x in packet_tokens]
                        except (ValueError, TypeError):
                            continue
                        
                        # 存储数据包
                        dataset_flow_packets[dataset_name][flow_id].append((packet_idx, packet_tokens))
                        dataset_all_packets[dataset_name].append((flow_id, packet_idx, packet_tokens))
                        datasets.add(dataset_name)
                        packet_count += 1
                    
                    except EOFError:
                        break
                    except Exception as e:
                        continue
        
        except Exception as e:
            print(f"[PacketPairDataLoaderV3] 错误: 无法加载数据集: {e}")
            return False
        
        print(f"[PacketPairDataLoaderV3] 加载完成: {packet_count} 个数据包")
        print(f"[PacketPairDataLoaderV3] 数据集数: {len(datasets)}")
        for ds_name in sorted(datasets):
            ds_packet_count = len(dataset_all_packets[ds_name])
            print(f"  - {ds_name}: {ds_packet_count} 个数据包")
        
        self.dataset_flow_packets = dataset_flow_packets
        self.dataset_all_packets = dataset_all_packets
        self.datasets = datasets
        return len(datasets) > 0 and packet_count > 0
    
    def _create_packet_pair(self):
        """
        创建一个数据包对及其标签
        确保只在同一个数据集内生成
        
        Returns:
            (src, label) 其中 src 是 packet1+packet2 的拼接，label 是 0/1/2
        """
        # 检查是否有多个数据集
        if len(self.datasets) == 0:
            return None
        
        # 随机选择一个数据集
        dataset_name = random.choice(list(self.datasets))
        
        # 检查该数据集是否有足够的数据包
        if dataset_name not in self.dataset_all_packets or len(self.dataset_all_packets[dataset_name]) < 2:
            return None
        
        # 随机选择第一个数据包
        flow_id1, packet_idx1, packet_tokens1 = random.choice(self.dataset_all_packets[dataset_name])
        
        if random.random() < self.diff_flow_ratio:
            # 20%: 选择来自不同流的第二个数据包（但同一数据集）
            label = 0
            dataset_packets = self.dataset_all_packets[dataset_name]
            # 不断尝试直到找到不同流的数据包
            for _ in range(10):  # 最多尝试10次
                flow_id2, packet_idx2, packet_tokens2 = random.choice(dataset_packets)
                if flow_id2 != flow_id1:
                    break
        else:
            # 80%: 选择来自同一流的第二个数据包
            flow_packets = self.dataset_flow_packets[dataset_name][flow_id1]
            if len(flow_packets) < 2:
                # 如果同流没有其他数据包，重新选择
                return self._create_packet_pair()
            
            # 从同一流中随机选择另一个数据包
            packet_idx2, packet_tokens2 = random.choice(flow_packets)
            flow_id2 = flow_id1
            
            # 避免选择同一个数据包
            if packet_idx2 == packet_idx1:
                flow_packets_filtered = [p for p in flow_packets if p[0] != packet_idx1]
                if not flow_packets_filtered:
                    return self._create_packet_pair()
                packet_idx2, packet_tokens2 = random.choice(flow_packets_filtered)
            
            # 根据packet_index确定label
            if packet_idx1 < packet_idx2:
                label = 1  # packet1在前
            else:
                label = 2  # packet1在后
        
        # 创建input sequence
        # [CLS] + packet1_tokens + [SEP] + packet2_tokens + [SEP] + padding
        max_len_per_packet = (self.seq_length - 4) // 2
        
        # 截断数据包到合适长度
        packet_tokens1 = packet_tokens1[:max_len_per_packet]
        packet_tokens2 = packet_tokens2[:max_len_per_packet]
        
        src = []
        src.append(CLS_ID)  # CLS token ID (2)
        
        # 将十六进制字符串tokens转换为token IDs
        # 优化：直接从十六进制字符串计算token_id，无需字典查询
        for hex_token in packet_tokens1:
            # hex_token 是十六进制字符串，如 '4500', '0020'
            try:
                # 直接计算: '4500' (hex) → 0x4500 = 17664 (decimal) → +5偏移 = 17669
                val = int(hex_token, 16)
                token_id = val + 5  # +5是5个特殊tokens的偏移
            except (ValueError, TypeError):
                token_id = 1  # [UNK] token ID
            src.append(token_id)
        
        src.append(SEP_ID)  # SEP token ID (3)
        
        for hex_token in packet_tokens2:
            try:
                val = int(hex_token, 16)
                token_id = val + 5
            except (ValueError, TypeError):
                token_id = 1  # [UNK] token ID
            src.append(token_id)
        
        src.append(SEP_ID)  # SEP token ID (3)
        
        # Padding 到 seq_length
        while len(src) < self.seq_length:
            src.append(PAD_ID)  # PAD token ID (0)
        
        src = src[:self.seq_length]
        
        return (src, label)
    
    def __iter__(self):
        """迭代器 - 分批次加载全部数据，动态生成数据包对"""
        print(f"[DataLoader] 开始数据加载和训练...")
        
        # 分批次加载数据（每批50000个样本）
        batch_load_size = 50000
        total_batches_generated = 0
        
        try:
            with open(self.dataset_path, 'rb') as f:
                batch_number = 0
                
                while True:
                    # 清空上一批的数据
                    self.dataset_flow_packets.clear()
                    self.dataset_all_packets.clear()
                    self.datasets.clear()
                    
                    # 加载新的一批数据
                    batch_number += 1
                    
                    packet_count = 0
                    batch_has_data = False
                    
                    # 从文件中加载batch_load_size个样本
                    while packet_count < batch_load_size:
                        try:
                            instance = pickle.load(f)
                            batch_has_data = True
                            
                            # 支持4元组格式（新）和3元组格式（旧）
                            if len(instance) == 4:
                                flow_id, packet_idx, packet_tokens, dataset_name = instance
                            elif len(instance) == 3:
                                flow_id, packet_idx, packet_tokens = instance
                                dataset_name = "Unknown"
                            else:
                                continue
                            
                            # 验证数据格式
                            if not isinstance(flow_id, str):
                                flow_id = str(flow_id)
                            if not isinstance(packet_idx, int):
                                packet_idx = int(packet_idx)
                            if not isinstance(packet_tokens, list):
                                packet_tokens = list(packet_tokens)
                            
                            # 保持packet_tokens为十六进制字符串格式
                            try:
                                packet_tokens = [str(x) for x in packet_tokens]
                            except (ValueError, TypeError):
                                continue
                            
                            # 存储数据包
                            self.dataset_flow_packets[dataset_name][flow_id].append((packet_idx, packet_tokens))
                            self.dataset_all_packets[dataset_name].append((flow_id, packet_idx, packet_tokens))
                            self.datasets.add(dataset_name)
                            packet_count += 1
                        
                        except EOFError:
                            # 文件已读完
                            break
                        except Exception as e:
                            continue
                    
                    # 如果这一批没有加载到任何数据，说明已经到达文件末尾
                    if not batch_has_data or packet_count == 0:
                        break
                    
                    # 从这一批数据中生成数据包对
                    if len(self.datasets) > 0:
                        batch_src = []
                        batch_label = []
                        
                        # 从这一批生成大约10倍的数据包对 (batch_load_size * 10 / batch_size 个batch)
                        target_pairs_in_batch = batch_load_size * 10
                        pairs_generated_in_batch = 0
                        
                        while pairs_generated_in_batch < target_pairs_in_batch:
                            # 创建数据包对
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
                            
                            # 每生成batch_size个对，就yield一次
                            if len(batch_src) >= self.batch_size:
                                yield (
                                    torch.tensor(batch_src, dtype=torch.long),
                                    torch.tensor(batch_label, dtype=torch.long)
                                )
                                total_batches_generated += 1
                                batch_src = []
                                batch_label = []
                        
                        # yield剩余的数据
                        if len(batch_src) > 0:
                            yield (
                                torch.tensor(batch_src, dtype=torch.long),
                                torch.tensor(batch_label, dtype=torch.long)
                            )
                            total_batches_generated += 1
        
        except Exception as e:
            print(f"[DataLoader] 错误: {e}")
            import traceback
            traceback.print_exc()


# 别名，向后兼容
PacketPairDataLoaderV2 = PacketPairDataLoaderV3

# 别名，向后兼容新的导入方式（用于 packet_pair_trainer.py）
PacketPairDataLoader = PacketPairDataLoaderV3
PacketPairDataset = None  # 暂时为None，如需实现再添加

