#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import glob
import json
import binascii
import random
import numpy as np
import scapy.all as scapy
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager


def makedir(path):
    """创建目录，如果已存在则忽略"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        pass


def extract_packet_mfr(packet):
    """
    从单个数据包提取MFR数据
    返回: (header, payload) 或 None
    - header: IP层去除payload后的部分 (包括IP层header、传输层header和应用层的部分header，80字节，160个十六进制字符)
    - payload: 应用层载荷 (240字节，480个十六进制字符)
    """
    try:
        header = None
        payload = ''
        
        # 情况1: 有IP层（正常pcap）
        if 'IP' in packet:
            # 提取整个IP层
            header = (binascii.hexlify(bytes(packet['IP']))).decode()
            
            # 提取应用层载荷
            try:
                payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
                # 从header中移除payload部分
                header = header.replace(payload, '')
            except:
                payload = ''
        
        # 情况2: 没有IP层，但Raw层包含IP数据（匿名化pcap）
        elif 'Raw' in packet:
            # 从Raw层提取数据
            raw_data = (binascii.hexlify(bytes(packet['Raw']))).decode()
            
            # 检查是否是IP数据（检查IP版本号）
            if len(raw_data) >= 8:
                first_byte = int(raw_data[0:2], 16)
                version = (first_byte >> 4) & 0x0F
                
                # 如果是IPv4
                if version == 4:
                    # header: 前160个字符（80字节）
                    # payload: 从160开始的480个字符（240字节）
                    header = raw_data[:160] if len(raw_data) >= 160 else raw_data
                    payload = raw_data[160:640] if len(raw_data) > 160 else ''
                else:
                    return None
            else:
                return None
        else:
            # 既没有IP层也没有Raw层
            return None
        
        if header is None:
            return None
        
        # 标准化header长度为160个十六进制字符 (80字节)
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        
        # 标准化payload长度为480个十六进制字符 (240字节)
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        
        return (header, payload)
    except:
        return None


def read_packet_matrices(pcap_path, max_packets=1000):
    """
    读取pcap文件，为每个数据包生成token矩阵
    返回: 数据包token矩阵列表，每个矩阵为8行字符串列表
    - 每行: 40个bytes = 20个tokens，每个token是2 bytes（4个十六进制字符）
    - 第1-2行: header (80字节 = 40 tokens)，每行20 tokens
    - 第3-8行: payload (240字节 = 120 tokens)，每行20 tokens
    总共: 80 + 240 = 320字节 = 8行
    
    参数:
        pcap_path: pcap文件路径
        max_packets: 最大处理的数据包数量（防止内存溢出）
    """
    try:
        # 使用scapy读取，限制数据包数量
        packets = scapy.rdpcap(pcap_path, count=max_packets)
    except Exception as e:
        # 静默处理错误，避免打印过多信息
        return []
    
    packet_matrices = []
    
    for packet in packets:
        try:
            mfr_data = extract_packet_mfr(packet)
            if mfr_data is None:
                continue
            
            header, payload = mfr_data
            # header: 160个十六进制字符 = 80字节
            # payload: 480个十六进制字符 = 240字节
            
            # 构建矩阵: 8行，每行40个bytes（20个tokens）
            matrix_rows = []
            
            # 第1-2行: header (80字节)，每行40字节=20个tokens
            for row_idx in range(2):
                start = row_idx * 80  # 每行40字节 = 80个十六进制字符
                row_hex = header[start:start+80]
                # 转换为tokens：每4个十六进制字符一个token
                tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                matrix_rows.append(' '.join(tokens))
            
            # 第3-8行: payload (240字节)，每行40字节=20个tokens
            for row_idx in range(6):
                start = row_idx * 80  # 每行40字节 = 80个十六进制字符
                row_hex = payload[start:start+80]
                tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                matrix_rows.append(' '.join(tokens))
            
            packet_matrices.append(matrix_rows)
        except Exception:
            # 跳过有问题的数据包
            continue
    
    return packet_matrices


# 流的前5个数据包的MFR矩阵横向拼接而成
def read_flow_matrix(pcap_path):
    """
    读取pcap文件，提取前5个数据包生成流级别token矩阵
    返回: 8行字符串列表（横向拼接5个数据包矩阵）
    - 每行: 200个bytes = 100个tokens，每个token是2 bytes（4个十六进制字符）
    - 流矩阵前2行: 5个数据包的header横向拼接，每行100 tokens
    - 流矩阵后6行: 5个数据包的payload横向拼接，每行100 tokens
    """
    try:
        # 只读取前5个数据包即可
        packets = scapy.rdpcap(pcap_path, count=5)
    except Exception as e:
        # 静默处理错误
        return None
    
    # 提取前5个数据包的MFR数据
    packet_data = []
    for packet in packets:
        try:
            mfr_data = extract_packet_mfr(packet)
            if mfr_data is None:
                continue
            
            packet_data.append(mfr_data)
            
            if len(packet_data) >= 5:
                break
        except Exception:
            continue
    
    # 如果不足5个包，用零填充
    if len(packet_data) < 5:
        for i in range(5 - len(packet_data)):
            packet_data.append(('0' * 160, '0' * 480))
    
    try:
        # 为每个数据包生成8行token矩阵（每行是十六进制字符串）
        packet_hex_rows = []
        for header, payload in packet_data:
            pkt_rows = []
            # 第1-2行: header (80字节)，每行40字节
            for row_idx in range(2):
                start = row_idx * 80
                pkt_rows.append(header[start:start+80])
            # 第3-8行: payload (240字节)，每行40字节
            for row_idx in range(6):
                start = row_idx * 80
                pkt_rows.append(payload[start:start+80])
            packet_hex_rows.append(pkt_rows)
        
        # 横向拼接5个数据包的矩阵 -> 8行，每行200字节（100个tokens）
        flow_matrix_rows = []
        for row_idx in range(8):  # 8行
            row_hex = ''
            for pkt_idx in range(5):  # 5个数据包
                row_hex += packet_hex_rows[pkt_idx][row_idx]  # 每个数据包贡献40字节
            # 转换为tokens：每4个十六进制字符一个token
            tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
            flow_matrix_rows.append(' '.join(tokens))
        
        return flow_matrix_rows
    except Exception:
        return None


def read_MFR_bytes(pcap_path):
    """
    使用MFR（Multi-Field Representation）方法读取pcap文件
    提取前5个数据包，每个包包含：
    - header: IP层去除payload后的部分（80字节，160个十六进制字符)
    - payload: 应用层载荷数据（Raw层，纯数据部分）- 480个十六进制字符 (240字节)
    总共: 5 * (160 + 480) = 3200个十六进制字符
    
    处理流程：
    1. 提取整个IP层
    2. 提取Raw层（payload）
    3. 从IP层中移除payload部分，得到header
    4. header标准化为80字节，payload标准化为240字节
    """
    try:
        packets = scapy.rdpcap(pcap_path)
    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        return None
    
    data = []
    for packet in packets:
        mfr_data = extract_packet_mfr(packet)
        if mfr_data is None:
            continue
        
        data.append(mfr_data)
        
        if len(data) >= 5:
            break
    
    # 如果不足5个包，用零填充
    if len(data) < 5:
        for i in range(5 - len(data)):
            data.append(('0' * 160, '0' * 480))
    
    # 合并所有header和payload
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    
    return final_data


def read_MFR_sequence(pcap_path):
    """
    读取MFR并转换为适合BERT的序列格式
    2 bytes 组成一个 token
    """
    mfr_hex = read_MFR_bytes(pcap_path)
    if mfr_hex is None:
        return None
    
    # 将十六进制字符串转换为2字节token序列 (0-65535)
    # 每4个十六进制字符（2字节）组成一个token
    token_sequence = [int(mfr_hex[i:i+4], 16) for i in range(0, len(mfr_hex), 4)]
    
    return token_sequence


def read_MFR_token_sequence(pcap_path):
    """
    读取MFR并转换为token序列（每个token由2个字节的16进制组成）
    返回格式: 空格分隔的字符串 "token1 token2 token3 ..."
    每个token是2个字节（4个十六进制字符）
    """
    mfr_hex = read_MFR_bytes(pcap_path)
    if mfr_hex is None:
        return None
    
    # 将十六进制字符串分割为2字节token序列
    # 每4个十六进制字符（2字节）组成一个token
    token_sequence = [mfr_hex[i:i+4] for i in range(0, len(mfr_hex), 4)]
    
    # 返回空格分隔的token序列
    return ' '.join(token_sequence)


def process_pcap_dataset(pcap_root_path, output_path, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    """
    处理pcap数据集，生成适合ET-BERT的训练数据
    
    参数:
        pcap_root_path: pcap文件根目录，期望结构为 root/class_name/*.pcap
        output_path: 输出JSON文件路径
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        valid_ratio: 验证集比例
    """
    
    # 查找所有类别目录
    class_dirs = [d for d in glob.glob(os.path.join(pcap_root_path, '*')) if os.path.isdir(d)]
    
    if len(class_dirs) == 0:
        print(f"No class directories found in {pcap_root_path}")
        return
    
    print(f"Found {len(class_dirs)} classes")
    
    dataset = {
        'train': defaultdict(lambda: {'samples': 0, 'payload': {}}),
        'test': defaultdict(lambda: {'samples': 0, 'payload': {}}),
        'valid': defaultdict(lambda: {'samples': 0, 'payload': {}})
    }
    
    class_to_id = {}
    
    # 处理每个类别
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Processing classes")):
        class_name = os.path.basename(class_dir)
        class_to_id[class_name] = class_idx
        
        # 查找该类别下的所有pcap文件
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        
        print(f"Class {class_name}: {len(pcap_files)} pcap files")
        
        # 打乱文件顺序
        random.shuffle(pcap_files)
        
        # 分割数据集
        n_total = len(pcap_files)
        n_train = int(n_total * train_ratio)
        n_test = int(n_total * test_ratio)
        n_valid = n_total - n_train - n_test
        
        train_files = pcap_files[:n_train]
        test_files = pcap_files[n_train:n_train+n_test]
        valid_files = pcap_files[n_train+n_test:]
        
        # 处理训练集
        for pcap_file in tqdm(train_files, desc=f"Processing {class_name} train", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['train'][class_idx]['samples'] + 1)
                dataset['train'][class_idx]['payload'][sample_id] = token_seq
                dataset['train'][class_idx]['samples'] += 1
        
        # 处理测试集
        for pcap_file in tqdm(test_files, desc=f"Processing {class_name} test", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['test'][class_idx]['samples'] + 1)
                dataset['test'][class_idx]['payload'][sample_id] = token_seq
                dataset['test'][class_idx]['samples'] += 1
        
        # 处理验证集
        for pcap_file in tqdm(valid_files, desc=f"Processing {class_name} valid", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['valid'][class_idx]['samples'] + 1)
                dataset['valid'][class_idx]['payload'][sample_id] = token_seq
                dataset['valid'][class_idx]['samples'] += 1
    
    # 转换defaultdict为普通dict以便JSON序列化
    output_data = {
        'metadata': {
            'class_mapping': class_to_id,
            'num_classes': len(class_to_id)
        },
        'train': {str(k): dict(v) for k, v in dataset['train'].items()},
        'test': {str(k): dict(v) for k, v in dataset['test'].items()},
        'valid': {str(k): dict(v) for k, v in dataset['valid'].items()}
    }
    
    # 保存为JSON
    makedir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Train samples: {sum([v['samples'] for v in dataset['train'].values()])}")
    print(f"Test samples: {sum([v['samples'] for v in dataset['test'].values()])}")
    print(f"Valid samples: {sum([v['samples'] for v in dataset['valid'].values()])}")
    
    return output_data


def convert_json_to_npy(json_path, output_dir):
    """
    将JSON数据转换为NPY格式（可选）
    适用于需要numpy数组格式的场景
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    makedir(output_dir)
    
    for split in ['train', 'test', 'valid']:
        all_sequences = []
        all_labels = []
        
        for class_id, class_data in data[split].items():
            if class_id == 'metadata':
                continue
            for sample_id, payload in class_data['payload'].items():
                # 将token序列（16进制字符串）转换为整数列表
                token_seq = [int(x, 16) for x in payload.split()]
                all_sequences.append(token_seq)
                all_labels.append(int(class_id))
        
        # 保存为numpy数组
        np.save(os.path.join(output_dir, f'{split}_sequences.npy'), np.array(all_sequences, dtype=object))
        np.save(os.path.join(output_dir, f'{split}_labels.npy'), np.array(all_labels))
    
    print(f"NPY files saved to {output_dir}")


def generate_vocab_from_mfr():
    """
    生成MFR的词汇表
    每个token由2个字节的16进制组成，词汇表大小为65536 (0000-ffff)
    加上特殊token
    """
    vocab = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[CLS]': 2,
        '[SEP]': 3,
        '[MASK]': 4,
    }
    
    # 添加2字节16进制token (0000-ffff)
    # 注意：这会生成一个大词汇表，可能需要较长时间
    for i in range(65536):
        hex_token = format(i, '04x')  # 转换为4位16进制字符串（小写）
        vocab[hex_token] = i + 5
    
    return vocab


def save_vocab(vocab_path):
    """保存词汇表到文件"""
    vocab = generate_vocab_from_mfr()
    
    makedir(os.path.dirname(vocab_path))
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    print(f"Vocabulary saved to {vocab_path}")


def process_single_pcap(pcap_file, dataset_name, class_name):
    """
    处理单个pcap文件，提取数据包和流矩阵
    
    参数:
        pcap_file: pcap文件路径
        dataset_name: 数据集名称
        class_name: 类别名称
    
    返回:
        (packet_matrices, flow_matrix) 或 (None, None)
    """
    try:
        # 一次性读取pcap文件，避免重复读取导致的问题
        try:
            packets = scapy.rdpcap(pcap_file, count=1000)
        except Exception as e:
            # 如果读取失败，打印错误信息
            print(f"  ⚠️  无法读取文件 {os.path.basename(pcap_file)}: {str(e)[:50]}")
            return (None, None)
        
        # 从读取的数据包中提取packet_matrices
        packet_matrices = []
        for packet in packets:
            try:
                mfr_data = extract_packet_mfr(packet)
                if mfr_data is None:
                    continue
                
                header, payload = mfr_data
                matrix_rows = []
                
                # 第1-2行: header
                for row_idx in range(2):
                    start = row_idx * 80
                    row_hex = header[start:start+80]
                    tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                    matrix_rows.append(' '.join(tokens))
                
                # 第3-8行: payload
                for row_idx in range(6):
                    start = row_idx * 80
                    row_hex = payload[start:start+80]
                    tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                    matrix_rows.append(' '.join(tokens))
                
                packet_matrices.append(matrix_rows)
            except Exception:
                continue
        
        # 从前5个有效数据包提取flow_matrix（从所有包中提取，而不是只从前5个包）
        flow_matrix = None
        try:
            packet_data = []
            # 从所有包中提取前5个有效的MFR数据包
            for packet in packets:
                try:
                    mfr_data = extract_packet_mfr(packet)
                    if mfr_data is None:
                        continue
                    packet_data.append(mfr_data)
                    if len(packet_data) >= 5:
                        break
                except Exception:
                    continue
            
            # 如果无法提取到任何有效包，返回None（不生成全零矩阵）
            if len(packet_data) == 0:
                flow_matrix = None
            else:
                # 如果不足5个包，用零填充
                if len(packet_data) < 5:
                    for i in range(5 - len(packet_data)):
                        packet_data.append(('0' * 160, '0' * 480))
                
                # 生成流矩阵
                packet_hex_rows = []
                for header, payload in packet_data:
                    pkt_rows = []
                    for row_idx in range(2):
                        start = row_idx * 80
                        pkt_rows.append(header[start:start+80])
                    for row_idx in range(6):
                        start = row_idx * 80
                        pkt_rows.append(payload[start:start+80])
                    packet_hex_rows.append(pkt_rows)
                
                flow_matrix_rows = []
                for row_idx in range(8):
                    row_hex = ''
                    for pkt_idx in range(5):
                        row_hex += packet_hex_rows[pkt_idx][row_idx]
                    tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                    flow_matrix_rows.append(' '.join(tokens))
                
                flow_matrix = flow_matrix_rows
        except Exception as e:
            # 流矩阵生成失败不影响数据包矩阵
            # 记录错误信息以便调试
            print(f"  ⚠️  流矩阵生成失败 {os.path.basename(pcap_file)}: {type(e).__name__}: {str(e)[:100]}")
            import traceback
            print(f"       详细错误: {traceback.format_exc()[:200]}")
        
        return (packet_matrices, flow_matrix)
        
    except MemoryError:
        print(f"  ⚠️  内存不足，跳过文件: {os.path.basename(pcap_file)}")
        return (None, None)
    except KeyboardInterrupt:
        raise  # 允许用户中断
    except Exception as e:
        # 捕获所有异常，避免子进程崩溃
        print(f"  ⚠️  处理文件出错 {os.path.basename(pcap_file)}: {str(e)[:100]}")
        return (None, None)


def process_class_mfr_matrices(class_dir, output_dir, dataset_name, class_name, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1, num_workers=6, batch_size=10000, log_interval=100):
    """
    为单个类别目录生成TXT文本文件（使用多进程）：
    1. packet_mfr/ - 数据包级别的MFR矩阵（每个数据包一个.txt文件，8行×20tokens）
    2. flow_mfr/ - 流级别的MFR矩阵（每条流一个.txt文件，8行×100tokens）
    
    参数:
        class_dir: 类别目录路径，直接包含.pcap文件
        output_dir: 输出目录
        dataset_name: 数据集名称 (如: CICIoT2022)
        class_name: 类别名称 (如: Attacks_Flood)
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        valid_ratio: 验证集比例
        num_workers: 进程数（默认6个）
        batch_size: 每处理多少个文件后保存一次（默认10000，避免内存溢出）
        log_interval: 每处理多少个文件后输出一次日志（默认100）
    """
    
    # 查找该目录下的所有pcap文件
    pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
    
    if len(pcap_files) == 0:
        print(f"⚠️  目录中没有找到.pcap文件: {class_dir}")
        return None
    
    # 打乱文件顺序
    random.shuffle(pcap_files)
    
    # 分割数据集
    n_total = len(pcap_files)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_valid = n_total - n_train - n_test
    
    train_files = pcap_files[:n_train]
    test_files = pcap_files[n_train:n_train+n_test]
    valid_files = pcap_files[n_train+n_test:]
    
    # 创建输出目录结构
    makedir(output_dir)
    
    # 创建packet_mfr和flow_mfr目录，每个目录下有train/test/valid子目录
    packet_base_dir = os.path.join(output_dir, 'packet_mfr')
    flow_base_dir = os.path.join(output_dir, 'flow_mfr')
    
    for split in ['train', 'test', 'valid']:
        makedir(os.path.join(packet_base_dir, split))
        makedir(os.path.join(flow_base_dir, split))
    
    # 统计计数器
    sample_counts = {
        'train': {'packet': 0, 'flow': 0},
        'test': {'packet': 0, 'flow': 0},
        'valid': {'packet': 0, 'flow': 0}
    }
    
    def save_packet_txt(matrix, split_name, packet_id):
        """保存单个数据包矩阵为.txt文件"""
        filename = f"packet_{packet_id:08d}.txt"
        filepath = os.path.join(packet_base_dir, split_name, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            # 第一行：类别标签
            f.write(f"{dataset_name} {class_name}\n")
            # 后续8行：矩阵数据
            for row in matrix:
                f.write(f"{row}\n")
    
    def save_flow_txt(matrix, split_name, flow_id):
        """保存单个流矩阵为.txt文件"""
        filename = f"flow_{flow_id:08d}.txt"
        filepath = os.path.join(flow_base_dir, split_name, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            # 第一行：类别标签
            f.write(f"{dataset_name} {class_name}\n")
            # 后续8行：矩阵数据
            for row in matrix:
                f.write(f"{row}\n")
    
    def process_files_parallel(files, split_name):
        """使用多进程并行处理文件，直接保存为.txt文件"""
        processed_count = 0
        flow_success_count = 0
        flow_fail_count = 0
        
        print(f"    🚀 [{split_name}] 开始处理 {len(files)} 个文件...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(process_single_pcap, pcap_file, dataset_name, class_name): pcap_file 
                      for pcap_file in files}
            
            # 禁用tqdm实时进度条，改用自定义日志输出（每log_interval个文件输出一次）
            # 这样可以大幅减少日志文件的更新频率
            for future in as_completed(futures, timeout=None):
                pcap_file = futures[future]
                try:
                    # 设置超时时间为60秒
                    packet_matrices, flow_matrix = future.result(timeout=60)
                    
                    # 保存数据包矩阵为.txt文件（每个pcap中可能有多个packet）
                    if packet_matrices:
                        for matrix in packet_matrices:
                            sample_counts[split_name]['packet'] += 1
                            save_packet_txt(matrix, split_name, sample_counts[split_name]['packet'])
                    
                    # 保存流矩阵为.txt文件（每个pcap一个flow）
                    if flow_matrix is not None:
                        sample_counts[split_name]['flow'] += 1
                        save_flow_txt(flow_matrix, split_name, sample_counts[split_name]['flow'])
                        flow_success_count += 1
                    else:
                        flow_fail_count += 1
                    
                    processed_count += 1
                    
                    # 每处理log_interval个文件，输出一次日志
                    if processed_count % log_interval == 0:
                        print(f"    📝 [{split_name}] 已处理 {processed_count}/{len(files)} 个文件 ({processed_count/len(files)*100:.1f}%), 数据包样本: {sample_counts[split_name]['packet']}, 流样本: {sample_counts[split_name]['flow']}")
                        
                except TimeoutError:
                    print(f"  ⚠️  处理超时 {os.path.basename(pcap_file)}")
                    flow_fail_count += 1
                except Exception as e:
                    print(f"  ⚠️  处理出错 {os.path.basename(pcap_file)}: {str(e)[:50]}")
                    flow_fail_count += 1
        
        # 打印最终处理统计
        total_files = len(files)
        success_rate = (flow_success_count / total_files * 100) if total_files > 0 else 0
        print(f"    ✅ [{split_name}] 全部完成! 处理 {processed_count}/{total_files} 个文件, 数据包样本: {sample_counts[split_name]['packet']}, 流样本: {sample_counts[split_name]['flow']}")
        print(f"    📊 {split_name} 流生成统计: 成功 {flow_success_count}/{total_files} ({success_rate:.1f}%), 失败 {flow_fail_count}")
    
    # 处理训练集（使用{num_workers}个进程）
    print(f"  处理训练集（{num_workers}进程，每{log_interval}个文件输出日志）...")
    process_files_parallel(train_files, 'train')
    
    # 处理测试集（使用{num_workers}个进程）
    print(f"  处理测试集（{num_workers}进程，每{log_interval}个文件输出日志）...")
    process_files_parallel(test_files, 'test')
    
    # 处理验证集（使用{num_workers}个进程）
    print(f"  处理验证集（{num_workers}进程，每{log_interval}个文件输出日志）...")
    process_files_parallel(valid_files, 'valid')
    
    # 打印统计信息
    print(f"  ✓ 数据包.txt文件: {sample_counts['train']['packet']} train, {sample_counts['test']['packet']} test, {sample_counts['valid']['packet']} valid")
    print(f"  ✓ 流.txt文件: {sample_counts['train']['flow']} train, {sample_counts['test']['flow']} test, {sample_counts['valid']['flow']} valid")
    
    return {
        'packet_samples': sample_counts['train']['packet'] + sample_counts['test']['packet'] + sample_counts['valid']['packet'],
        'flow_samples': sample_counts['train']['flow'] + sample_counts['test']['flow'] + sample_counts['valid']['flow'],
        'counts': sample_counts
    }


def process_single_class_directory(class_dir, output_path, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    """
    处理单个类别目录（直接包含pcap文件）
    
    参数:
        class_dir: 类别目录路径，直接包含.pcap文件
        output_path: 输出JSON文件路径
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        valid_ratio: 验证集比例
    """
    
    # 查找该目录下的所有pcap文件
    pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
    
    if len(pcap_files) == 0:
        print(f"⚠️  目录中没有找到.pcap文件: {class_dir}")
        return None
    
    class_name = os.path.basename(class_dir)
    
    # 打乱文件顺序
    random.shuffle(pcap_files)
    
    # 分割数据集
    n_total = len(pcap_files)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_valid = n_total - n_train - n_test
    
    train_files = pcap_files[:n_train]
    test_files = pcap_files[n_train:n_train+n_test]
    valid_files = pcap_files[n_train+n_test:]
    
    dataset = {
        'train': {'samples': 0, 'payload': {}},
        'test': {'samples': 0, 'payload': {}},
        'valid': {'samples': 0, 'payload': {}}
    }
    
    # 处理训练集
    for pcap_file in tqdm(train_files, desc=f"  Train", leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['train']['samples'] + 1)
            dataset['train']['payload'][sample_id] = token_seq
            dataset['train']['samples'] += 1
    
    # 处理测试集
    for pcap_file in tqdm(test_files, desc=f"  Test", leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['test']['samples'] + 1)
            dataset['test']['payload'][sample_id] = token_seq
            dataset['test']['samples'] += 1
    
    # 处理验证集
    for pcap_file in tqdm(valid_files, desc=f"  Valid", leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['valid']['samples'] + 1)
            dataset['valid']['payload'][sample_id] = token_seq
            dataset['valid']['samples'] += 1
    
    # 转换为输出格式
    output_data = {
        'metadata': {
            'class_name': class_name,
            'num_classes': 1
        },
        'train': dataset['train'],
        'test': dataset['test'],
        'valid': dataset['valid']
    }
    
    # 保存为JSON
    makedir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    return output_data


def process_presplit_dataset(dataset_root, dataset_name, output_dir):
    """
    处理已经预分割好train/test的数据集
    
    参数:
        dataset_root: 数据集根目录
        dataset_name: 数据集名称
        output_dir: 输出目录
    """
    dataset_path = os.path.join(dataset_root, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集不存在: {dataset_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 检查train和test目录
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(train_dir):
        print(f"❌ 训练集目录不存在: {train_dir}")
        return None
    
    if not os.path.exists(test_dir):
        print(f"⚠️  测试集目录不存在: {test_dir}，将只处理训练集")
    
    # 准备输出数据结构
    dataset = {
        'train': defaultdict(lambda: {'samples': 0, 'payload': {}}),
        'test': defaultdict(lambda: {'samples': 0, 'payload': {}})
    }
    
    class_to_id = {}
    
    # 处理训练集
    print(f"\n[Train] 处理训练集...")
    train_classes = [d for d in glob.glob(os.path.join(train_dir, '*')) if os.path.isdir(d)]
    
    for class_idx, class_dir in enumerate(tqdm(train_classes, desc="处理训练集类别")):
        class_name = os.path.basename(class_dir)
        class_to_id[class_name] = class_idx
        
        # 查找所有pcap文件
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        
        print(f"  类别 {class_name}: {len(pcap_files)} 个文件")
        
        # 处理每个pcap文件
        for pcap_file in tqdm(pcap_files, desc=f"  {class_name}", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['train'][class_idx]['samples'] + 1)
                dataset['train'][class_idx]['payload'][sample_id] = token_seq
                dataset['train'][class_idx]['samples'] += 1
    
    # 处理测试集
    if os.path.exists(test_dir):
        print(f"\n[Test] 处理测试集...")
        test_classes = [d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)]
        
        for class_dir in tqdm(test_classes, desc="处理测试集类别"):
            class_name = os.path.basename(class_dir)
            
            # 使用训练集中的类别ID，如果不存在则创建新ID
            if class_name not in class_to_id:
                class_to_id[class_name] = len(class_to_id)
                print(f"  ⚠️  测试集中发现新类别: {class_name}")
            
            class_idx = class_to_id[class_name]
            
            # 查找所有pcap文件
            pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
            if len(pcap_files) == 0:
                pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
            
            print(f"  类别 {class_name}: {len(pcap_files)} 个文件")
            
            # 处理每个pcap文件
            for pcap_file in tqdm(pcap_files, desc=f"  {class_name}", leave=False):
                token_seq = read_MFR_token_sequence(pcap_file)
                if token_seq is not None and len(token_seq) > 0:
                    sample_id = str(dataset['test'][class_idx]['samples'] + 1)
                    dataset['test'][class_idx]['payload'][sample_id] = token_seq
                    dataset['test'][class_idx]['samples'] += 1
    
    # 转换为普通dict
    output_data = {
        'metadata': {
            'dataset_name': dataset_name,
            'class_mapping': class_to_id,
            'num_classes': len(class_to_id)
        },
        'train': {str(k): dict(v) for k, v in dataset['train'].items()},
        'test': {str(k): dict(v) for k, v in dataset['test'].items()}
    }
    
    # 保存JSON
    output_path = os.path.join(output_dir, f"{dataset_name}_processed.json")
    makedir(output_dir)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    # 打印统计信息
    train_total = sum([v['samples'] for v in dataset['train'].values()])
    test_total = sum([v['samples'] for v in dataset['test'].values()])
    
    print(f"\n✓ 数据集 {dataset_name} 处理完成!")
    print(f"  输出文件: {output_path}")
    print(f"  类别数: {len(class_to_id)}")
    print(f"  训练集样本: {train_total}")
    print(f"  测试集样本: {test_total}")
    print(f"  总样本数: {train_total + test_total}")
    
    return output_data


def process_multiple_datasets(datasets_root, dataset_names, output_dir):
    """
    批量处理多个数据集
    
    参数:
        datasets_root: 数据集根目录
        dataset_names: 数据集名称列表
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"批量处理 {len(dataset_names)} 个数据集")
    print(f"{'='*60}")
    
    results = {}
    
    for dataset_name in dataset_names:
        try:
            result = process_presplit_dataset(datasets_root, dataset_name, output_dir)
            if result is not None:
                results[dataset_name] = result
        except Exception as e:
            print(f"❌ 处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总信息
    summary = {
        'total_datasets': len(results),
        'datasets': {}
    }
    
    for dataset_name, data in results.items():
        train_samples = sum([v['samples'] for v in data['train'].values()])
        test_samples = sum([v['samples'] for v in data['test'].values()])
        
        summary['datasets'][dataset_name] = {
            'num_classes': data['metadata']['num_classes'],
            'train_samples': train_samples,
            'test_samples': test_samples,
            'total_samples': train_samples + test_samples
        }
    
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"所有数据集处理完成!")
    print(f"{'='*60}")
    print(f"汇总信息已保存到: {summary_path}")
    print(f"\n处理结果:")
    for dataset_name, info in summary['datasets'].items():
        print(f"  {dataset_name}:")
        print(f"    类别数: {info['num_classes']}")
        print(f"    训练集: {info['train_samples']}")
        print(f"    测试集: {info['test_samples']}")
        print(f"    总计: {info['total_samples']}")
    
    return summary


if __name__ == '__main__':
    # ========================================
    # 数据处理配置
    # ========================================
    
    # 尝试从config.py加载配置
    try:
        from config import DATASETS_ROOT, DATASETS, PROCESSING_MODE
        print("✓ 从config.py加载配置")
    except ImportError:
        print("⚠️  config.py未找到，使用默认配置")
        # 数据集根目录（使用单流Pcap数据集目录 - 包含所有分流后的pcap文件）
        DATASETS_ROOT = "/root/datasets_new/单流Pcap数据集_预训练_随机化/"
        
        # 要处理的数据集列表
        DATASETS = [
            # "NonVPN",
            # "NonTor",
            # "BenignTraffic", 
            "CICIoT2022"
        ]
        PROCESSING_MODE = None  # 如果未定义，使用下面的默认值
    
    # 处理模式配置
    # 'mode1': 数据集目录包含多个类别子目录（每个子目录包含pcap文件）
    # 'mode2': 数据集目录直接包含pcap文件（作为单个类别处理）
    # 'auto': 自动检测模式（推荐）
    if PROCESSING_MODE is None:
        PROCESSING_MODE = 'auto'  # 默认使用模式2（适合当前场景）
    
    # 验证处理模式
    valid_modes = ['mode1', 'mode2', 'auto']
    if PROCESSING_MODE not in valid_modes:
        print(f"❌ 无效的处理模式: {PROCESSING_MODE}")
        print(f"   有效模式: {', '.join(valid_modes)}")
        exit(1)
    
    # 输出目录
    OUTPUT_DIR = "/root/datasets_new/MFR数据集_预训练_随机化/"
    
    # 词汇表路径
    VOCAB_PATH = "/root/我的代码和数据集/Mine/models/mfr_vocab.txt"
    
    # 检查是否有数据集需要处理
    if not DATASETS or len(DATASETS) == 0:
        print("\n❌ 错误: 没有找到可处理的数据集!")
        print(f"请检查数据集根目录: {DATASETS_ROOT}")
        exit(1)
    
    # 主流程
    print("\n" + "=" * 60)
    print("MFR 数据预处理 - 单流Pcap数据集批量处理（多进程版本）")
    print("=" * 60)
    
    # 处理模式说明
    mode_descriptions = {
        'mode1': '模式1: 数据集目录包含多个类别子目录',
        'mode2': '模式2: 数据集目录直接包含pcap文件',
        'auto': '自动模式: 自动检测使用哪种模式'
    }
    print(f"处理模式: {PROCESSING_MODE} - {mode_descriptions.get(PROCESSING_MODE, '未知模式')}")
    print("\n处理模式说明：")
    print("  - mode1: 数据集目录下包含多个类别子目录，每个子目录是一个类别")
    print("           输出路径: output_dir/dataset_name_MFR/class_name/")
    print("  - mode2: 数据集目录下直接包含pcap文件，作为单个类别处理")
    print("           输出路径: output_dir/dataset_name_MFR/")
    print("  - auto:  自动检测目录结构，选择合适的模式")
    print("=" * 60)
    
    # Step 1: 生成词汇表
    print("\n" + "=" * 60)
    print("[Step 1] 生成词汇表")
    print("=" * 60)
    save_vocab(VOCAB_PATH)
    
    # Step 2: 批量处理所有数据集
    print("\n" + "=" * 60)
    print("[Step 2] 批量处理数据集")
    print("=" * 60)
    
    # 创建输出目录
    makedir(OUTPUT_DIR)
    
    results = {}
    for dataset_name in DATASETS:
        dataset_path = os.path.join(DATASETS_ROOT, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"❌ 数据集不存在: {dataset_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")
        print(f"输入目录: {dataset_path}")
        print(f"处理模式: {PROCESSING_MODE}\n")
        
        try:
            # 查找所有类别目录
            class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
            # 过滤掉临时目录
            class_dirs = [d for d in class_dirs 
                         if not os.path.basename(d).startswith('.') 
                         and not os.path.basename(d).startswith('__')]
            
            # 检查数据集目录下是否直接包含pcap文件
            direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcap'))
            if len(direct_pcap_files) == 0:
                direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcapng'))
            
            # 根据处理模式判断
            is_mode2 = False
            if PROCESSING_MODE == 'mode2':
                # 手动选择模式2: 数据集目录直接包含pcap文件
                if len(direct_pcap_files) == 0:
                    print(f"❌ 模式2: 未找到pcap文件: {dataset_path}")
                    continue
                print(f"使用模式2: 数据集目录直接包含pcap文件，作为单个类别处理")
                print(f"找到 {len(direct_pcap_files)} 个pcap文件\n")
                class_dirs = [dataset_path]  # 将数据集目录本身作为类别目录
                is_mode2 = True
            elif PROCESSING_MODE == 'mode1':
                # 手动选择模式1: 数据集目录包含多个类别子目录
                if len(class_dirs) == 0:
                    print(f"❌ 模式1: 未找到类别子目录: {dataset_path}")
                    continue
                print(f"使用模式1: 数据集目录包含 {len(class_dirs)} 个类别子目录\n")
                is_mode2 = False
            else:
                # 自动模式: 自动检测
                if len(class_dirs) == 0:
                    # 没有子目录，检查是否有直接pcap文件
                    if len(direct_pcap_files) == 0:
                        print(f"❌ 未找到类别目录或pcap文件: {dataset_path}")
                        continue
                    print(f"自动检测: 模式2 - 数据集目录直接包含pcap文件，作为单个类别处理")
                    print(f"找到 {len(direct_pcap_files)} 个pcap文件\n")
                    class_dirs = [dataset_path]  # 将数据集目录本身作为类别目录
                    is_mode2 = True
                else:
                    print(f"自动检测: 模式1 - 数据集目录包含 {len(class_dirs)} 个类别子目录\n")
                    is_mode2 = False
            
            # 创建该数据集的MFR输出目录
            dataset_output_dir = os.path.join(OUTPUT_DIR, f"{dataset_name}_MFR")
            makedir(dataset_output_dir)
            
            dataset_stats = {
                'total_packet_samples': 0, 
                'total_flow_samples': 0, 
                'classes': {}
            }
            
            # 为每个类别目录生成TXT文本文件（packet_mfr/ 和 flow_mfr/ 目录）
            for class_idx, class_dir in enumerate(class_dirs, 1):
                if is_mode2:
                    # 模式2: 使用数据集名称作为类别名称
                    class_name = dataset_name
                else:
                    # 模式1: 使用子目录名称作为类别名称
                    class_name = os.path.basename(class_dir)
                
                # 跳过一些临时目录和文件（模式1中已经过滤，这里再次检查确保安全）
                if class_name.startswith('.') or class_name.startswith('__'):
                    continue
                
                # 跳过压缩包等文件
                if not os.path.isdir(class_dir):
                    continue
                
                print(f"\n[{class_idx}/{len(class_dirs)}] 处理类别: {class_name}")
                
                # 模式2: 输出到 dataset_name_MFR/（不创建类别子目录）
                # 模式1: 输出到 dataset_name_MFR/class_name/
                if is_mode2:
                    class_output_dir = dataset_output_dir
                else:
                    class_output_dir = os.path.join(dataset_output_dir, class_name)
                makedir(class_output_dir)
                
                # 使用新函数处理该类别，生成packet_mfr/和flow_mfr/目录及.txt文件
                result = process_class_mfr_matrices(
                    class_dir=class_dir,
                    output_dir=class_output_dir,
                    dataset_name=dataset_name,
                    class_name=class_name,
                    train_ratio=0.7,
                    test_ratio=0.2,
                    valid_ratio=0.1
                )
                
                if result is not None:
                    packet_total = result['packet_samples']
                    flow_total = result['flow_samples']
                    
                    dataset_stats['classes'][class_name] = {
                        'packet_samples': packet_total,
                        'flow_samples': flow_total
                    }
                    dataset_stats['total_packet_samples'] += packet_total
                    dataset_stats['total_flow_samples'] += flow_total
                    
                    print(f"  ✓ {class_name}: {packet_total} 个数据包.txt文件, {flow_total} 个流.txt文件")
            
            results[dataset_name] = dataset_stats
            print(f"\n✓ {dataset_name} 处理完成!")
            print(f"  总数据包样本: {dataset_stats['total_packet_samples']}")
            print(f"  总流样本: {dataset_stats['total_flow_samples']}")
            
        except Exception as e:
            print(f"❌ 处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总信息
    summary_path = os.path.join(OUTPUT_DIR, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 完成
    print("\n" + "=" * 60)
    print("✓ 全部处理完成!")
    print("=" * 60)
    print(f"\n输出文件位置: {OUTPUT_DIR}")
    print("  目录结构:")
    for dataset_name in results.keys():
        if PROCESSING_MODE == 'mode2':
            # 模式2: 数据集目录直接包含pcap文件
            print(f"    {dataset_name}_MFR/")
            print(f"      ├── packet_mfr/        (数据包级别矩阵)")
            print(f"      │   ├── train/         (训练集.txt文件)")
            print(f"      │   ├── test/          (测试集.txt文件)")
            print(f"      │   └── valid/         (验证集.txt文件)")
            print(f"      └── flow_mfr/          (流级别矩阵)")
            print(f"          ├── train/         (训练集.txt文件)")
            print(f"          ├── test/          (测试集.txt文件)")
            print(f"          └── valid/         (验证集.txt文件)")
        else:
            # 模式1: 数据集目录包含多个类别子目录
            print(f"    {dataset_name}_MFR/")
            print(f"      <class_name>/")
            print(f"        ├── packet_mfr/        (数据包级别矩阵)")
            print(f"        │   ├── train/         (训练集.txt文件)")
            print(f"        │   ├── test/          (测试集.txt文件)")
            print(f"        │   └── valid/         (验证集.txt文件)")
            print(f"        └── flow_mfr/          (流级别矩阵)")
            print(f"            ├── train/         (训练集.txt文件)")
            print(f"            ├── test/          (测试集.txt文件)")
            print(f"            └── valid/         (验证集.txt文件)")
    print(f"\n  汇总信息: processing_summary.json")
    print(f"  词汇表: {VOCAB_PATH}")
    
    print("\n📊 处理统计:")
    if results:
        print(f"  成功处理数据集: {len(results)}/{len(DATASETS)}")
        for dataset_name, stats in results.items():
            print(f"    ✓ {dataset_name}: {len(stats['classes'])} 个类别")
            print(f"       - 数据包样本: {stats['total_packet_samples']}")
            print(f"       - 流样本: {stats['total_flow_samples']}")
    
    print("\n💡 后续步骤:")
    print("  1. 查看各个 <dataset>_MFR/<class_name>/packet_mfr/ 和 flow_mfr/ 目录")
    print("  2. 使用生成的.txt文件进行模型训练")
    print("")
    print("📖 数据格式说明:")
    print("  【数据包级别矩阵】packet_mfr/*.txt:")
    print("    - 文件命名: packet_00000001.txt, packet_00000002.txt, ...")
    print("    - 文件格式: 共9行")
    print("      ├─ 第1行: 类别标签 (如: CICIoT2022 Attacks_Flood)")
    print("      └─ 第2-9行: 8行MFR矩阵数据")
    print("    - 每个数据包: 320字节 (header 80字节 + payload 240字节)")
    print("    - Token格式: 2 bytes = 1 token (4个十六进制字符)")
    print("    - 矩阵格式: 8行字符串，每行空格分隔的tokens")
    print("    - 矩阵内容:")
    print("      ├─ 第1-2行: header (每行40字节=20个tokens，空格分隔)")
    print("      └─ 第3-8行: payload (每行40字节=20个tokens，空格分隔)")
    print("    - 示例: '4500 001c 1c46 4000 ...' (每行20个tokens)")
    print("")
    print("  【流级别矩阵】flow_mfr/*.txt:")
    print("    - 文件命名: flow_00000001.txt, flow_00000002.txt, ...")
    print("    - 文件格式: 共9行")
    print("      ├─ 第1行: 类别标签 (如: CICIoT2022 Attacks_Flood)")
    print("      └─ 第2-9行: 8行MFR矩阵数据")
    print("    - 每条流: 5个数据包横向拼接")
    print("    - Token格式: 2 bytes = 1 token (4个十六进制字符)")
    print("    - 矩阵格式: 8行字符串，每行空格分隔的tokens")
    print("    - 矩阵内容:")
    print("      ├─ 第1-2行: 5个数据包的header横向拼接 (每行200字节=100个tokens)")
    print("      └─ 第3-8行: 5个数据包的payload横向拼接 (每行200字节=100个tokens)")
    print("    - 示例: '4500 001c ... 4500 002c ...' (每行100个tokens)")
    print("")
    print("  【性能优化】:")
    print("    - 日志更新：每处理100个文件输出一次进度日志（不使用tqdm实时进度条）")
    print("    - 实时保存：每个数据包/流矩阵立即保存为独立.txt文件")
    print("    - 避免内存溢出和日志文件频繁更新")
    print("")

