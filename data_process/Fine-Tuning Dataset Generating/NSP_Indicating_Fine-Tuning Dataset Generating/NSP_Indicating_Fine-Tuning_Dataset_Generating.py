"""
微调数据集生成脚本
从MFR数据集生成用于微调的数据集（.pt格式）
- 从每个流MFR矩阵（8行×100token，共800个token）提取前5个数据包
- 每个数据包截取80个token
- 每个流的标签来自所属的类别
- 在微调时，直接使用前5个数据包进行分类

微调数据格式:
  (flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)
  
其中：
  - flow_id: 流ID (8位数字字符串)
  - flow_mfr_tokens: 前5个数据包的400个token (5个数据包×80 tokens)
  - category_name: 类别名称 (如 "Idle", "Backdoor")
  - dataset_name: 数据集名称 (如 "CICIoT2022_MFR")
  - label_id: 流量分类标签ID (0~K-1)

微调任务: 给定前5个数据包的MFR矩阵，预测其流量类别标签
"""
import os
import re
import pickle
import argparse
from functools import partial
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# 配置参数
class Config:
    # 源数据路径
    SOURCE_BASE_DIR = "/root/datasets/MFR数据集_微调_随机化/"
    # 目标数据路径
    TARGET_BASE_DIR = "/root/datasets/fine-tuning数据集_随机化/"
    # TARGET_BASE_DIR = "/root/datasets/fine-tuning数据集_随机化_50pct数据/"
    
    # 四个数据集配置
    # 1.正常微调
    DATASETS = {
        "NUDT_MobileTraffic Dataset_all_MFR": {
            "source": "NUDT_MobileTraffic Dataset_all_MFR",
            "target": "finetuning_NUDT_MobileTraffic Dataset_all_MFR.pt",
            "total_flows": 41975  # 数据集总流数（每个流提取前5个数据包×80 tokens）
        },
        "USTC-TFC2016_MFR": {
            "source": "USTC-TFC2016_MFR",
            "target": "finetuning_USTC-TFC2016_MFR.pt",
            "total_flows": 19387
        },
        "ISCXVPN2016_MFR": {
            "source": "ISCXVPN2016_MFR",
            "target": "finetuning_ISCXVPN2016_MFR.pt",
            "total_flows": 13520 # 实际只有13520条
        },
        "CICIoT2023_MFR": {
            "source": "CICIoT2023_MFR",
            "target": "finetuning_CICIoT2023_MFR.pt",
            "total_flows": 25453
        },
        "ISCXVPN2016-APP_MFR": {
            "source": "ISCXVPN2016-APP_MFR",
            "target": "finetuning_ISCXVPN2016-APP_MFR.pt",
            "total_flows": 11592 #实际只有11592条
        }
    }
    '''
    # 2.10%微调数据
    DATASETS = {
        "NUDT_MobileTraffic Dataset_all_MFR": {
            "source": "NUDT_MobileTraffic Dataset_all_MFR",
            "target": "finetuning_NUDT_MobileTraffic Dataset_all_MFR.pt",
            "total_flows": 4197  # 数据集总流数（每个流提取前5个数据包×80 tokens）
        },
        "USTC-TFC2016_MFR": {
            "source": "USTC-TFC2016_MFR",
            "target": "finetuning_USTC-TFC2016_MFR.pt",
            "total_flows": 1938
        },
        "ISCXVPN2016_MFR": {
            "source": "ISCXVPN2016_MFR",
            "target": "finetuning_ISCXVPN2016_MFR.pt",
            "total_flows": 1352 # 实际只有13520条
        },
        "CICIoT2023_MFR": {
            "source": "CICIoT2023_MFR",
            "target": "finetuning_CICIoT2023_MFR.pt",
            "total_flows": 2545
        },
        "ISCXVPN2016-APP_MFR": {
            "source": "ISCXVPN2016-APP_MFR",
            "target": "finetuning_ISCXVPN2016-APP_MFR.pt",
            "total_flows": 1159 #实际只有11592条
        }
    }
    '''
    '''
    # 3.50%微调数据
    DATASETS = {
        "NUDT_MobileTraffic Dataset_all_MFR": {
            "source": "NUDT_MobileTraffic Dataset_all_MFR",
            "target": "finetuning_NUDT_MobileTraffic Dataset_all_MFR.pt",
            "total_flows": 20987  # 数据集总流数（每个流提取前5个数据包×80 tokens）
        },
        "USTC-TFC2016_MFR": {
            "source": "USTC-TFC2016_MFR",
            "target": "finetuning_USTC-TFC2016_MFR.pt",
            "total_flows": 9693
        },
        "ISCXVPN2016_MFR": {
            "source": "ISCXVPN2016_MFR",
            "target": "finetuning_ISCXVPN2016_MFR.pt",
            "total_flows": 6760 # 实际只有13520条
        },
        "CICIoT2023_MFR": {
            "source": "CICIoT2023_MFR",
            "target": "finetuning_CICIoT2023_MFR.pt",
            "total_flows": 12726
        },
        "ISCXVPN2016-APP_MFR": {
            "source": "ISCXVPN2016-APP_MFR",
            "target": "finetuning_ISCXVPN2016-APP_MFR.pt",
            "total_flows": 5796 #实际只有11592条
        }
    }
    '''

    # 随机种子
    RANDOM_SEED = 42


def extract_dataset_name(file_path):
    """
    从文件路径中提取数据集名称
    
    路径格式：.../CICIoT2022_MFR/Idle/flow_mfr/train/flow_00000001.txt
    数据集名称：CICIoT2022_MFR
    """
    match = re.search(r'([A-Za-z0-9_]+_MFR)', file_path)
    if match:
        return match.group(1)
    return "Unknown"


def extract_category_name(file_path):
    """
    从文件路径中提取类别名称
    
    路径格式：.../CICIoT2022_MFR/Idle/flow_mfr/train/flow_00000001.txt
    类别名称：Idle
    """
    # 获取flow_mfr的父目录名（即类别名称）
    parts = Path(file_path).parts
    try:
        flow_mfr_idx = parts.index('flow_mfr')
        if flow_mfr_idx > 0:
            return parts[flow_mfr_idx - 1]
    except (ValueError, IndexError):
        pass
    return "Unknown"


FLOW_ROWS = 8
FLOW_TOKENS_PER_ROW = 100
PACKETS_PER_FLOW = 5
TOKENS_PER_PACKET_ROW = FLOW_TOKENS_PER_ROW // PACKETS_PER_FLOW
HEADER_ROWS = 2
PAYLOAD_ROWS_FOR_MODEL = 2
PACKET_TOKEN_LENGTH = (HEADER_ROWS + PAYLOAD_ROWS_FOR_MODEL) * TOKENS_PER_PACKET_ROW


def split_flow_matrix_to_packet80(flow_tokens_lines):
    """
    将8×100 flow MFR矩阵恢复为5个独立packet序列。

    flow_mfr由5个packet横向拼接：
    - row 0-1: header，每行每包20 tokens，合计40 tokens
    - row 2-7: payload，每行每包20 tokens；模型输入只取前2行payload，合计40 tokens
    """
    if len(flow_tokens_lines) != FLOW_ROWS:
        return None

    for row_tokens in flow_tokens_lines:
        if len(row_tokens) != FLOW_TOKENS_PER_ROW:
            return None

    packet_samples = []
    for packet_idx in range(PACKETS_PER_FLOW):
        col_start = packet_idx * TOKENS_PER_PACKET_ROW
        col_end = col_start + TOKENS_PER_PACKET_ROW

        packet_tokens = []
        for row_idx in range(HEADER_ROWS):
            packet_tokens.extend(flow_tokens_lines[row_idx][col_start:col_end])
        for row_idx in range(HEADER_ROWS, HEADER_ROWS + PAYLOAD_ROWS_FOR_MODEL):
            packet_tokens.extend(flow_tokens_lines[row_idx][col_start:col_end])

        if len(packet_tokens) != PACKET_TOKEN_LENGTH:
            return None
        packet_samples.append(packet_tokens)

    return packet_samples


def read_flow_mfr_finetune(txt_file, label_map):
    """
    读取单个"流MFR矩阵.txt"文件，提取前5个数据包（每个80 tokens）和标签（用于微调）
    
    流MFR矩阵格式：
    - 第1行：标签（如"CICIoT2022 Idle"）
    - 第2-9行：8行×100列的MFR矩阵数据（前2行为header，后6行为payload，5个包横向拼接）
    - 提取前5个数据包，每个数据包80个token（Header 40 + Payload 40），共400个token
    
    参数:
        txt_file: 流MFR矩阵文件路径
        label_map: {类别名称: 标签ID} 的字典
    
    返回:
        tuple: (flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)
               flow_mfr_tokens: 包含400个token (5个数据包×80 tokens)
               如果文件无效，返回None
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 至少需要标签行 + 8行MFR数据
        if len(lines) < 9:
            return None
        
        # 从文件名中提取流ID
        flow_filename = os.path.basename(txt_file)
        if flow_filename.startswith('flow_') and flow_filename.endswith('.txt'):
            flow_id = flow_filename.replace('flow_', '').replace('.txt', '')
        else:
            match = re.search(r'flow_(\d+)', txt_file)
            if match:
                flow_id = match.group(1)
            else:
                return None
        
        # 确保流ID是8位数字格式
        if not flow_id.isdigit():
            return None
        flow_id = flow_id.zfill(8)
        
        # 提取数据集和类别名称
        dataset_name = extract_dataset_name(txt_file)
        category_name = extract_category_name(txt_file)
        
        # 读取MFR token数据（从第2行开始，共8行）
        # flow_mfr为8×100矩阵：前2行为header，后6行为payload，5个包横向拼接。
        flow_tokens_lines = []
        for line in lines[1:9]:  # 取8行
            line = line.strip()
            if line:
                flow_tokens_lines.append(line.split())
        
        packet_tokens_list = split_flow_matrix_to_packet80(flow_tokens_lines)
        if packet_tokens_list is None:
            return None
        
        # 提取前5个数据包，每个数据包80个token（Header 40 + Payload 40）。
        flow_tokens = []
        for packet_tokens in packet_tokens_list[:5]:
            flow_tokens.extend(packet_tokens)
        
        # 获取标签ID
        label_id = label_map.get(category_name, -1)  # -1表示未知类别
        if label_id < 0:
            return None
        
        # 返回(flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)
        return (flow_id, flow_tokens, category_name, dataset_name, label_id)
    
    except Exception as e:
        return None


def collect_flow_files_by_category(source_dir):
    """
    收集指定数据集下所有类别的流文件
    支持两种目录结构：
    1. 模式1（无类别层）：source_dir/flow_mfr/train|test|valid/flow_*.txt
    2. 模式2（有类别层）：source_dir/类别名/flow_mfr/train|test|valid/flow_*.txt
    
    返回: {类别名称: [文件路径列表]}
    """
    from collections import defaultdict
    category_files = defaultdict(list)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"警告: 源目录不存在: {source_dir}")
        return category_files
    
    # 首先检查模式1：source_dir 下直接有 flow_mfr 目录（无类别层）
    flow_mfr_direct = source_path / "flow_mfr"
    if flow_mfr_direct.exists() and flow_mfr_direct.is_dir():
        print(f"  检测到模式1（无类别层）: {flow_mfr_direct}")
        # 收集所有split（train/valid/test）的txt文件
        all_files = []
        for split in ["train", "valid", "test"]:
            split_dir = flow_mfr_direct / split
            if split_dir.exists():
                txt_files = list(split_dir.glob("flow_*.txt"))
                all_files.extend([str(f) for f in txt_files])
        
        # 对于无类别层的数据集，使用数据集名称作为类别名称
        # 注意：extract_category_name 函数会返回 flow_mfr 的父目录名（即数据集名称）
        dataset_name = source_path.name  # 如 "browser_MFR"
        category_name = dataset_name  # 保持完整名称，与 extract_category_name 返回的值一致
        category_files[category_name] = all_files
        print(f"  类别 {category_name}: 找到 {len(all_files)} 个流文件")
        return category_files
    
    # 模式2：source_dir 下有类别子目录，每个类别下有 flow_mfr（有类别层）
    print(f"  检测到模式2（有类别层）")
    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        flow_mfr_base = category_dir / "flow_mfr"
        
        if not flow_mfr_base.exists():
            # 跳过非类别目录（如 packet_mfr 等）
            continue
        
        # 收集该类别下所有split（train/valid/test）的txt文件
        all_files = []
        for split in ["train", "valid", "test"]:
            split_dir = flow_mfr_base / split
            if split_dir.exists():
                txt_files = list(split_dir.glob("flow_*.txt"))
                all_files.extend([str(f) for f in txt_files])
        
        category_files[category_name] = all_files
        print(f"  类别 {category_name}: 找到 {len(all_files)} 个流文件")
    
    return category_files


def calculate_flows_per_class(category_files, total_flows):
    """
    根据每个类别的文件数量，按比例计算每个类别应该采样的流数
    返回: {类别名称: 流数}
    """
    # 统计每个类别的文件总数
    category_counts = {cat: len(files) for cat, files in category_files.items()}
    total_files = sum(category_counts.values())
    
    print(f"\n类别文件数量统计:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} 个流文件")
    print(f"  总计: {total_files} 个流文件")
    
    # 按比例分配流数
    flows_per_class = {}
    allocated_flows = 0
    
    for cat in sorted(category_counts.keys()):
        # 按比例计算该类别应分配的流数
        proportion = category_counts[cat] / total_files
        class_flows = int(total_flows * proportion)
        
        # 确保不超过该类别的实际文件数
        class_flows = min(class_flows, category_counts[cat])
        
        flows_per_class[cat] = class_flows
        allocated_flows += class_flows
    
    # 如果分配的流数不足，从流数最多的类别补充
    if allocated_flows < total_flows:
        shortage = total_flows - allocated_flows
        # 找到文件数最多且还有余量的类别
        for cat in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
            if flows_per_class[cat] < category_counts[cat]:
                available = category_counts[cat] - flows_per_class[cat]
                add_flows = min(shortage, available)
                flows_per_class[cat] += add_flows
                allocated_flows += add_flows
                shortage -= add_flows
                if shortage == 0:
                    break
    
    print(f"\n按比例分配流数:")
    for cat, count in sorted(flows_per_class.items()):
        proportion = count / total_flows * 100
        print(f"  {cat}: {count} 个流 ({proportion:.2f}%)")
    print(f"  总计: {allocated_flows} 个流")
    
    return flows_per_class


def collect_all_flow_files_nsp(mfr_root):
    """
    收集所有"流MFR矩阵.txt"文件（即 flow_*.txt 文件）
    用于NSP微调任务
    
    返回:
        list: 所有流MFR矩阵文件路径列表
    """
    import glob
    all_files = []
    
    # 查找所有flow_mfr目录下的 flow_*.txt 文件
    flow_mfr_dirs = glob.glob(
        os.path.join(mfr_root, '**', 'flow_mfr'), 
        recursive=True
    )
    
    print(f"找到 {len(flow_mfr_dirs)} 个flow_mfr目录")
    
    for flow_dir in flow_mfr_dirs:
        # 在每个flow_mfr目录下查找所有 flow_*.txt 文件
        flow_files = glob.glob(
            os.path.join(flow_dir, '**', 'flow_*.txt'), 
            recursive=True
        )
        print(f"  - 在 {os.path.basename(flow_dir)} 中找到 {len(flow_files)} 个文件")
        all_files.extend(flow_files)
    
    print(f"\n总共收集的流MFR矩阵文件: {len(all_files)}")
    return all_files


def process_files_to_dataset_finetune(category_files, flows_per_class, label_map, output_base_path, 
                                      processes_num=6, log_interval=1000, save_interval=100000, 
                                      train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    将flow txt文件列表处理成微调数据集(.pt格式)，按照8:1:1比例划分为训练集、验证集和测试集
    
    参数:
        category_files: {类别名称: [文件路径列表]}
        flows_per_class: {类别名称: 需要采样的流数}
        label_map: {类别名称: 标签ID}
        output_base_path: 输出文件的基础路径（不含扩展名），将生成 train.pt, dev.pt, test.pt
        processes_num: 进程数（默认6）
        log_interval: 日志更新间隔（默认1000个文件）
        save_interval: 数据保存间隔（默认100000个流）
        train_ratio: 训练集比例（默认0.8）
        dev_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.1）
        random_seed: 随机种子（默认42）
    
    返回:
        tuple: (train_count, dev_count, test_count) 三个数据集的流数
    """
    import random as rnd
    
    # 设置随机种子
    rnd.seed(random_seed)
    
    # 按类别采样流文件
    sampled_file_list = []
    for category_name, file_list in category_files.items():
        n_flows = flows_per_class[category_name]
        
        if n_flows == 0:
            continue
        
        if len(file_list) < n_flows:
            print(f"  警告: 类别 {category_name} 流数量不足 ({len(file_list)} < {n_flows})")
            sampled_file_list.extend(file_list)
        else:
            sampled_files = rnd.sample(file_list, n_flows)
            sampled_file_list.extend(sampled_files)
    
    # 打乱采样后的文件列表
    rnd.shuffle(sampled_file_list)
    
    processed_flow_count = 0
    skipped_count = 0
    total_files = len(sampled_file_list)
    
    print(f"\n{'='*80}")
    print(f"处理微调数据集（按8:1:1划分）")
    print(f"{'='*80}")
    print(f"总流文件数: {total_files}")
    print(f"每个流的token数: 400 (前5个数据包×80 tokens)")
    print(f"划分比例: 训练集={train_ratio*100:.0f}%, 验证集={dev_ratio*100:.0f}%, 测试集={test_ratio*100:.0f}%")
    print(f"使用 {processes_num} 个进程进行并行处理...")
    print(f"输出文件:")
    print(f"  - 训练集: {output_base_path}_train.pt")
    print(f"  - 验证集: {output_base_path}_dev.pt")
    print(f"  - 测试集: {output_base_path}_test.pt")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
    
    # 创建三个数据集文件
    train_writer = open(f"{output_base_path}_train.pt", 'wb')
    dev_writer = open(f"{output_base_path}_dev.pt", 'wb')
    test_writer = open(f"{output_base_path}_test.pt", 'wb')
    
    # 样本缓冲区（分别用于三个数据集）
    train_buffer = []
    dev_buffer = []
    test_buffer = []
    
    # 使用functools.partial创建可序列化的处理函数
    process_with_label_map = partial(read_flow_mfr_finetune, label_map=label_map)
    
    # 使用进程池处理采样后的文件
    all_flow_samples = []
    with Pool(processes=processes_num) as pool:
        file_count = 0
        
        for flow_sample in pool.imap(process_with_label_map, sampled_file_list):
            file_count += 1
            
            # 处理结果
            if flow_sample is not None:
                all_flow_samples.append(flow_sample)
                processed_flow_count += 1
            else:
                skipped_count += 1
            
            # 每处理log_interval个文件，打印一次日志
            if file_count % log_interval == 0:
                print(f"[进度] {file_count}/{total_files} 个流 | "
                      f"已处理={processed_flow_count}, "
                      f"跳过={skipped_count}")
    
    # 按照8:1:1比例划分数据集
    total_samples = len(all_flow_samples)
    train_end = int(total_samples * train_ratio)
    dev_end = train_end + int(total_samples * dev_ratio)
    
    train_samples = all_flow_samples[:train_end]
    dev_samples = all_flow_samples[train_end:dev_end]
    test_samples = all_flow_samples[dev_end:]
    
    print(f"\n数据集划分完成:")
    print(f"  训练集: {len(train_samples)} 个流 ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"  验证集: {len(dev_samples)} 个流 ({len(dev_samples)/total_samples*100:.1f}%)")
    print(f"  测试集: {len(test_samples)} 个流 ({len(test_samples)/total_samples*100:.1f}%)")
    
    # 保存训练集
    print(f"\n[保存训练集] 正在写入 {len(train_samples)} 个流样本...")
    for flow_sample in train_samples:
        pickle.dump(flow_sample, train_writer)
    train_writer.flush()
    train_writer.close()
    print(f"[保存训练集] 完成")
    
    # 保存验证集
    print(f"[保存验证集] 正在写入 {len(dev_samples)} 个流样本...")
    for flow_sample in dev_samples:
        pickle.dump(flow_sample, dev_writer)
    dev_writer.flush()
    dev_writer.close()
    print(f"[保存验证集] 完成")
    
    # 保存测试集
    print(f"[保存测试集] 正在写入 {len(test_samples)} 个流样本...")
    for flow_sample in test_samples:
        pickle.dump(flow_sample, test_writer)
    test_writer.flush()
    test_writer.close()
    print(f"[保存测试集] 完成")
    
    print(f"\n{'='*80}")
    print(f"[最终统计]")
    print(f"{'='*80}")
    print(f"✓ 总共处理的流文件: {file_count}")
    print(f"✓ 成功处理的流: {processed_flow_count}")
    print(f"✗ 跳过的文件数: {skipped_count}")
    print(f"📁 数据集保存到:")
    print(f"  - 训练集: {output_base_path}_train.pt ({len(train_samples)} 个流)")
    print(f"  - 验证集: {output_base_path}_dev.pt ({len(dev_samples)} 个流)")
    print(f"  - 测试集: {output_base_path}_test.pt ({len(test_samples)} 个流)")
    print(f"  每个样本格式: (flow_id, flow_mfr_tokens[400], category_name, dataset_name, label_id)")
    print(f"  flow_mfr_tokens: 前5个数据包×80 tokens = 400 tokens")
    print(f"{'='*80}\n")
    
    return len(train_samples), len(dev_samples), len(test_samples)


def process_dataset(dataset_key, config):
    """
    处理单个数据集，生成微调数据集文件
    按照各类别流数量比例分配，总流数为total_flows
    """
    import random as rnd
    
    dataset_config = config.DATASETS[dataset_key]
    source_dir = os.path.join(config.SOURCE_BASE_DIR, dataset_config["source"])
    target_dir = config.TARGET_BASE_DIR
    total_flows = dataset_config["total_flows"]
    
    print(f"\n{'='*80}")
    print(f"开始处理数据集: {dataset_key}")
    print(f"源目录: {source_dir}")
    print(f"总流数: {total_flows}")
    print(f"目标文件: {dataset_config['target']}")
    print(f"{'='*80}")
    
    # 创建目标目录
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 收集所有类别的所有流文件
    print(f"\n--- 收集流文件 ---")
    category_files = collect_flow_files_by_category(source_dir)
    
    if not category_files:
        print(f"错误: 没有找到任何流文件，跳过此数据集")
        return
    
    # 2. 计算每个类别应该采样的流数（按比例分配）
    flows_per_class = calculate_flows_per_class(category_files, total_flows)
    
    # 3. 创建标签映射（类别名 → 标签ID）
    categories = sorted(category_files.keys())
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    print(f"\n标签映射:")
    for cat, label_id in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {label_id}: {cat}")
    
    # 4. 处理并生成微调数据集（按8:1:1划分为train/dev/test）
    print(f"\n--- 处理并生成数据集（按8:1:1划分） ---")
    # 输出文件基础路径（不含扩展名），将生成 _train.pt, _dev.pt, _test.pt
    output_base = os.path.join(target_dir, dataset_config["target"].replace('.pt', ''))
    train_count, dev_count, test_count = process_files_to_dataset_finetune(
        category_files,
        flows_per_class,
        label_map,
        output_base,
        processes_num=6,
        log_interval=1000,
        save_interval=100000,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        random_seed=config.RANDOM_SEED
    )
    
    print(f"\n数据集 {dataset_key} 处理完成!")
    print(f"  ✓ 输出文件:")
    print(f"    - 训练集: {output_base}_train.pt ({train_count} 个流)")
    print(f"    - 验证集: {output_base}_dev.pt ({dev_count} 个流)")
    print(f"    - 测试集: {output_base}_test.pt ({test_count} 个流)")
    print(f"  ✓ 总流样本数: {train_count + dev_count + test_count}")
    print(f"  ✓ 每个流包含: 400个token (前5个数据包×80 tokens)")
    

def verify_dataset_finetune(dataset_file, num_samples=5):
    """
    验证微调数据集文件的内容
    
    参数:
        dataset_file: 数据集文件路径
        num_samples: 要显示的样本数量
    
    样本格式: (flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)
    """
    print(f"\n{'='*80}")
    print(f"验证微调数据集: {dataset_file}")
    print(f"{'='*80}\n")
    
    try:
        with open(dataset_file, 'rb') as f:
            samples = []
            try:
                while len(samples) < num_samples:
                    sample = pickle.load(f)
                    samples.append(sample)
            except EOFError:
                pass
        
        print(f"✓ 成功加载 {len(samples)} 个样本用于验证\n")
        
        for i, sample in enumerate(samples):
            if len(sample) == 5:
                flow_id, flow_mfr_tokens, category_name, dataset_name, label_id = sample
                print(f"样本 {i+1}:")
                print(f"  ✓ flow_id: {flow_id}")
                print(f"  ✓ flow_mfr_tokens: list({len(flow_mfr_tokens)} tokens)")
                print(f"    前5个数据包MFR矩阵 (5个数据包×80 tokens = 400 tokens)")
                if flow_mfr_tokens:
                    print(f"    前10个tokens: {flow_mfr_tokens[:10]}")
                    print(f"    最后10个tokens: {flow_mfr_tokens[-10:]}")
                print(f"  ✓ category_name: {category_name}")
                print(f"  ✓ dataset_name: {dataset_name}")
                print(f"  ✓ label_id: {label_id}")
            else:
                print(f"样本 {i+1}: 格式不符 (len={len(sample)}, expected 5)")
            print()
        
        print(f"{'='*80}")
        print("✅ 验证完成！")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ 验证错误: {e}")


def main():
    """主函数 - NSP变形任务微调数据集生成"""
    parser = argparse.ArgumentParser(description='生成NSP变形任务微调数据集')
    parser.add_argument('--output_dir', type=str, 
                        default='/root/datasets/fine-tuning数据集_随机化/',
                        help='输出目录（用于保存.pt文件）')
    parser.add_argument('--verify', action='store_true',
                        help='生成数据集后验证它的内容')
    
    args = parser.parse_args()
    
    config = Config()
    # 更新目标目录
    config.TARGET_BASE_DIR = args.output_dir
    
    # 设置随机种子
    import random as rnd
    rnd.seed(config.RANDOM_SEED)
    
    print("="*80)
    print("微调数据集生成工具")
    print("="*80)
    print(f"随机种子: {config.RANDOM_SEED}")
    print(f"数据集数量: {len(config.DATASETS)}")
    print("说明:")
    print("  - 从MFR数据集中分别收集四个数据集的流文件")
    print("  - 每个流提取前5个数据包: 5个数据包×80 tokens = 400个token")
    print("  - 保存前5个数据包的MFR矩阵内容和对应的流量类别标签")
    print("  - 各类别按照其文件数量比例分配流数")
    print("  - 按照8:1:1比例随机划分为训练集、验证集和测试集")
    print("  - 每个数据集生成3个文件: *_train.pt, *_dev.pt, *_test.pt")
    print("  - 输出格式: (flow_id, flow_mfr_tokens[400], category_name, dataset_name, label_id)")
    print("  - 微调任务: 给定前5个数据包，预测其流量分类标签")
    print("="*80)
    
    # 处理每个数据集
    for dataset_key in config.DATASETS.keys():
        try:
            process_dataset(dataset_key, config)
        except Exception as e:
            print(f"\n❌ 错误: 处理数据集 {dataset_key} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 验证生成的数据集（可选）
    if args.verify:
        print("\n" + "="*80)
        print("验证生成的微调数据集...")
        print("="*80)
        for dataset_key in config.DATASETS.keys():
            dataset_config = config.DATASETS[dataset_key]
            output_base = os.path.join(config.TARGET_BASE_DIR, dataset_config["target"].replace('.pt', ''))
            
            # 验证三个数据集文件
            for split_name, split_file in [
                ("训练集", f"{output_base}_train.pt"),
                ("验证集", f"{output_base}_dev.pt"),
                ("测试集", f"{output_base}_test.pt")
            ]:
                if os.path.exists(split_file):
                    print(f"\n验证{split_name}: {split_file}")
                    verify_dataset_finetune(split_file, num_samples=2)
                else:
                    print(f"⚠️  {split_name}文件不存在: {split_file}")
    
    print("\n" + "="*80)
    print("✅ 所有数据集生成完成！")
    print("="*80)
    print(f"\n📁 输出目录: {config.TARGET_BASE_DIR}")
    print(f"\n生成的文件（每个数据集生成3个文件，按8:1:1划分）:")
    for dataset_key in config.DATASETS.keys():
        dataset_config = config.DATASETS[dataset_key]
        base_name = dataset_config['target'].replace('.pt', '')
        print(f"  ✓ {base_name}_train.pt (训练集)")
        print(f"  ✓ {base_name}_dev.pt (验证集)")
        print(f"  ✓ {base_name}_test.pt (测试集)")
    
    print(f"\n📝 数据格式说明:")
    print(f"  每个样本: (flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)")
    print(f"  - flow_id: 8位字符串 (如 '00000123')")
    print(f"  - flow_mfr_tokens: 400个token的列表 (前5个数据包×80 tokens)")
    print(f"    * 4位十六进制字符串(2bytes)组成的列表 (如 ['4500', '0020', 'd893', ...])")
    print(f"  - category_name: 类别名称 (如 'Idle', 'Backdoor', 等)")
    print(f"  - dataset_name: 数据集名称 (如 'CICIoT2022_MFR')")
    print(f"  - label_id: 流量分类标签ID (0~K-1)")
    print(f"\n💡 微调任务:")
    print(f"  - 输入: 前5个数据包的MFR矩阵 (400 tokens)")
    print(f"  - 输出: 流量类别标签")
    print(f"  - 数据集划分: 训练集:验证集:测试集 = 8:1:1")
    print()


if __name__ == "__main__":
    main()

