#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
将MFR数据集中的flow_mfr数据转换为dataset_pretrain.pt

功能：
- 从每个流MFR矩阵（8行×100token，共800个token）中提取5个数据包token序列（每个160个token）
- 生成dataset_pretrain.pt用于预训练任务

输出数据格式：
- 每个样本: (flow_id, packet_index, packet_tokens, dataset_name)
  - flow_id: 8位数字字符串，如 "00000123"
  - packet_index: 整数 (1-5)，代表数据包在流中的序号
  - packet_tokens: 包含160个token的列表，每个token是4位十六进制字符串（2bytes），如 "4500"、"0020" 等
  - dataset_name: 数据集名称
"""

import os
import glob
import pickle
import re
import argparse
from multiprocessing import Pool

# ============================================================================
# 数据集配置
# ============================================================================
# 要处理的数据集列表
# 如果为 None 或空列表，则处理所有数据集
# 可以指定要处理的数据集名称，例如：
# DATASETS_TO_PROCESS = ['NonVPN_MFR', 'NonTor_MFR', 'browser_MFR']
DATASETS_TO_PROCESS = [
    'CICIoT2022_MFR',
    'BenignTraffic_MFR',
    'NonTor_MFR',
    'NonVPN_MFR'
]
# 如果不想处理任何数据集（使用命令行参数），设置为 None
# DATASETS_TO_PROCESS = None
# ============================================================================


def hex_str_to_int(hex_str):
    """将十六进制字符串转换为整数"""
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return 0


def extract_dataset_name(file_path):
    """
    从文件路径中提取数据集名称
    
    支持的路径格式：
    1. 模式1（无类别层）：.../NonVPN_MFR/flow_mfr/train/flow_00000001.txt
       数据集名称：NonVPN_MFR
    2. 模式2（有类别层）：.../CICIoT2022_MFR/Attacks_Flood/flow_mfr/train/flow_00000001.txt
       数据集名称：CICIoT2022_MFR
    
    参数:
        file_path: 文件路径
    
    返回:
        str: 数据集名称，如 "NonVPN_MFR"、"NonTor_MFR"、"browser_MFR"、"BenignTraffic_MFR"、"CICIoT2022_MFR" 等
    """
    # 从路径中查找数据集目录名
    # 数据集目录名通常以 _MFR 结尾
    match = re.search(r'([A-Za-z0-9_]+_MFR)', file_path)
    if match:
        return match.group(1)
    
    # 如果没有找到，使用"Unknown"作为默认值
    return "Unknown"


def read_and_split_flow_to_packets(txt_file):
    """
    读取单个"流MFR矩阵.txt"文件，并将其拆分为5个数据包
    
    流MFR矩阵格式：
    - 第1行：标签（如"NonVPN NonVPN"或"NonTor NonTor"等，会被跳过）
    - 第2-9行：8行×100列的MFR矩阵数据，由5个数据包的MFR矩阵横向拼接
      * Packet1: 列0-19 (共160个token = 8行×20列)
      * Packet2: 列20-39 (共160个token = 8行×20列)
      * Packet3: 列40-59 (共160个token = 8行×20列)
      * Packet4: 列60-79 (共160个token = 8行×20列)
      * Packet5: 列80-99 (共160个token = 8行×20列)
    
    参数:
        txt_file: "流MFR矩阵.txt"文件路径
    
    返回:
        list: [(flow_id, packet_index, packet_tokens, dataset_name), ...] 
              其中flow_id为8位字符串，packet_index为整数(1-5)，
              packet_tokens为160个4位十六进制字符串组成的列表（每个token=2bytes），
              dataset_name为数据集名称
              如果文件无效，返回None
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 第一行是标签（如"CICIoT2022 Idle"），跳过
        # 第2-9行是8行MFR数据，每行200个token
        if len(lines) < 9:  # 至少需要标签行 + 8行MFR数据
            return None
        
        # 从文件名中提取流ID
        # 文件名格式：flow_XXXXXXXX.txt -> 流ID = "XXXXXXXX"
        flow_filename = os.path.basename(txt_file)
        
        # 优先从文件名提取流ID
        if flow_filename.startswith('flow_') and flow_filename.endswith('.txt'):
            flow_id = flow_filename.replace('flow_', '').replace('.txt', '')
        else:
            # 如果文件名不符合格式，尝试从路径查找
            match = re.search(r'flow_(\d+)', txt_file)
            if match:
                flow_id = match.group(1)
            else:
                return None
        
        # 确保流ID是8位数字格式
        if not flow_id.isdigit():
            return None
        
        flow_id = flow_id.zfill(8)  # 补零到8位
        
        # 提取数据集名称
        dataset_name = extract_dataset_name(txt_file)
        
        # 读取MFR token数据（从第2行开始，共8行）
        # 每行100个token，8行共800个token
        flow_tokens_lines = []
        for line in lines[1:9]:  # 取8行
            line = line.strip()
            if line:
                # 分割得到token列表（每个token是4位十六进制）
                tokens = line.split()
                flow_tokens_lines.append(tokens)
        
        # 检查是否有8行数据
        if len(flow_tokens_lines) != 8:
            return None
        
        # 检查每行是否都有100个token
        for i, row_tokens in enumerate(flow_tokens_lines):
            if len(row_tokens) != 100:
                return None
        
        # 拆分为5个数据包（横向拆分）
        # 每个数据包占20列，共8行，所以每个数据包160个token
        packet_samples = []
        for packet_idx in range(1, 6):  # 1到5
            # 计算当前数据包的列范围
            col_start = (packet_idx - 1) * 20  # 0, 20, 40, 60, 80
            col_end = packet_idx * 20          # 20, 40, 60, 80, 100
            
            # 从8行中提取该数据包的token
            hex_tokens = []
            for row_tokens in flow_tokens_lines:
                # 提取该行的col_start到col_end列
                hex_tokens.extend(row_tokens[col_start:col_end])
            
            # 检查token数量
            if len(hex_tokens) != 160:
                continue
            
            # 保留十六进制字符串格式（每个token由2bytes=4位十六进制字符串组成）
            packet_tokens = hex_tokens
            
            # 创建数据包样本：(流ID, 数据包序号, 数据包token列表, 数据集名称)
            packet_sample = (flow_id, packet_idx, packet_tokens, dataset_name)
            packet_samples.append(packet_sample)
        
        # 只有成功拆分出5个数据包时才返回
        if len(packet_samples) == 5:
            return packet_samples
        else:
            return None
    
    except Exception as e:
        return None


def collect_all_flow_files(mfr_root, dataset_names=None):
    """
    收集所有"流MFR矩阵.txt"文件（即 flow_*.txt 文件）
    明确收集 train、test、valid 三个子目录的所有文件
    
    支持的目录结构（两种模式）:
    
    模式1: 数据集目录直接包含flow_mfr（无类别层）
    - MFR数据集_预训练/
      - NonVPN_MFR/
        - flow_mfr/
          - train/flow_00000001.txt, flow_00000002.txt, ...
          - test/flow_XXXXXXXX.txt, ...
          - valid/flow_XXXXXXXX.txt, ...
      - NonTor_MFR/
        - flow_mfr/
          - train/, test/, valid/
      - browser_MFR/
        - flow_mfr/
          - train/, test/, valid/
      - BenignTraffic_MFR/
        - flow_mfr/
          - train/, test/, valid/
    
    模式2: 数据集目录包含类别子目录，每个类别下有flow_mfr（有类别层）
    - MFR数据集_预训练/
      - CICIoT2022_MFR/
        - Attacks_Flood/
          - flow_mfr/
            - train/flow_00000001.txt, ...
            - test/flow_XXXXXXXX.txt, ...
            - valid/flow_XXXXXXXX.txt, ...
        - 其他类别/
          - flow_mfr/
            - train/, test/, valid/
    
    注意：使用递归搜索（**/flow_mfr），两种模式都支持
    
    参数:
        mfr_root: MFR数据集根目录
        dataset_names: 要处理的数据集名称列表（如 ['NonVPN_MFR', 'NonTor_MFR']），
                      如果为None，则处理所有数据集
    
    返回:
        list: 所有流MFR矩阵文件路径列表（包含train、test、valid）
    """
    all_files = []
    split_stats = {'train': 0, 'test': 0, 'valid': 0, 'other': 0}
    dataset_stats = {}  # 统计每个数据集的文件数
    
    # 查找所有flow_mfr目录
    flow_mfr_dirs = glob.glob(
        os.path.join(mfr_root, '**', 'flow_mfr'), 
        recursive=True
    )
    
    print(f"Found {len(flow_mfr_dirs)} flow_mfr directories")
    
    # 如果指定了数据集名称，转换为集合以便快速查找
    if dataset_names is not None:
        dataset_names_set = set(dataset_names)
        print(f"指定处理的数据集: {dataset_names}")
    else:
        dataset_names_set = None
        print("处理所有数据集")
    
    for flow_dir in flow_mfr_dirs:
        # 从路径中提取数据集名称，用于过滤
        dataset_name = extract_dataset_name(flow_dir)
        
        # 如果指定了数据集列表，则只处理指定的数据集
        if dataset_names_set is not None and dataset_name not in dataset_names_set:
            continue
        
        # 初始化该数据集的统计
        if dataset_name not in dataset_stats:
            dataset_stats[dataset_name] = {'train': 0, 'test': 0, 'valid': 0, 'other': 0}
        
        # 明确收集 train、test、valid 三个子目录的文件
        for split in ['train', 'test', 'valid']:
            split_dir = os.path.join(flow_dir, split)
            if os.path.exists(split_dir):
                # 查找该子目录下的所有 flow_*.txt 文件
                flow_files = glob.glob(
                    os.path.join(split_dir, 'flow_*.txt')
                )
                all_files.extend(flow_files)
                split_stats[split] += len(flow_files)
                dataset_stats[dataset_name][split] += len(flow_files)
        
        # 也检查flow_mfr目录下是否有直接的文件（不在train/test/valid子目录中）
        direct_files = glob.glob(
            os.path.join(flow_dir, 'flow_*.txt')
        )
        if direct_files:
            all_files.extend(direct_files)
            split_stats['other'] += len(direct_files)
            dataset_stats[dataset_name]['other'] += len(direct_files)
    
    # 打印统计信息
    print(f"\n文件收集统计（总计）:")
    print(f"  - train: {split_stats['train']} 个文件")
    print(f"  - test:  {split_stats['test']} 个文件")
    print(f"  - valid: {split_stats['valid']} 个文件")
    if split_stats['other'] > 0:
        print(f"  - other: {split_stats['other']} 个文件（不在train/test/valid子目录中）")
    print(f"\nTotal flow matrix files collected: {len(all_files)}")
    
    # 打印每个数据集的统计信息
    if dataset_stats:
        print(f"\n各数据集文件统计:")
        for dataset_name in sorted(dataset_stats.keys()):
            stats = dataset_stats[dataset_name]
            total = sum(stats.values())
            print(f"  {dataset_name}:")
            print(f"    - train: {stats['train']} 个文件")
            print(f"    - test:  {stats['test']} 个文件")
            print(f"    - valid: {stats['valid']} 个文件")
            if stats['other'] > 0:
                print(f"    - other: {stats['other']} 个文件")
            print(f"    - 总计: {total} 个文件")
    
    return all_files


def process_files_to_dataset_pretrain(file_list, output_file, max_samples=None, 
                                      processes_num=6, log_interval=1000, save_interval=100000):
    """
    将flow txt文件列表处理成dataset_pretrain.pt格式文件
    
    参数:
        file_list: txt文件路径列表
        output_file: 输出的.pt文件路径
        max_samples: 最大样本数（None表示处理所有样本）
        processes_num: 进程数（默认6）
        log_interval: 日志更新间隔（默认1000个文件）
        save_interval: 数据保存间隔（默认100000个文件）
    """
    processed_flow_count = 0
    processed_packet_count = 0
    skipped_count = 0
    total_files = len(file_list)
    
    # 如果指定了最大样本数，则截断文件列表
    if max_samples is not None:
        file_list = file_list[:max_samples]
        total_files = len(file_list)
    
    print(f"\n{'='*80}")
    print(f"Processing dataset_pretrain.pt")
    print(f"{'='*80}")
    print(f"Total flow files to process: {total_files}")
    print(f"Expected packet samples: {total_files * 5} (5 packets per flow)")
    print(f"Using {processes_num} processes for parallel processing...")
    print(f"Output file: {output_file}")
    
    # 创建dataset文件（以追加模式打开，支持分批写入）
    dataset_writer = open(output_file, 'wb')
    
    # 数据包样本缓冲区
    packet_buffer = []
    
    # 使用进程池处理所有文件
    with Pool(processes=processes_num) as pool:
        file_count = 0
        
        for packet_samples in pool.imap(read_and_split_flow_to_packets, file_list):
            file_count += 1
            
            # 处理结果
            if packet_samples is not None and len(packet_samples) > 0:
                # 将5个数据包样本添加到缓冲区
                packet_buffer.extend(packet_samples)
                processed_flow_count += 1
                processed_packet_count += len(packet_samples)
            else:
                skipped_count += 1
            
            # 每处理log_interval个文件，打印一次日志
            if file_count % log_interval == 0:
                print(f"[Progress] {file_count}/{total_files} flows | "
                      f"Processed flows={processed_flow_count}, Packets={processed_packet_count}, "
                      f"Skipped={skipped_count}")
            
            # 每处理save_interval个文件，将缓冲区的数据追加写入.pt文件
            if file_count % save_interval == 0 and len(packet_buffer) > 0:
                print(f"[Save] Writing {len(packet_buffer)} packet samples to disk...")
                for packet_sample in packet_buffer:
                    pickle.dump(packet_sample, dataset_writer)
                dataset_writer.flush()
                print(f"[Save] Successfully saved | Total processed: {file_count} flows")
                packet_buffer.clear()
    
    # 处理完所有文件后，写入剩余缓冲区中的数据
    if len(packet_buffer) > 0:
        print(f"\n[Final Save] Writing remaining {len(packet_buffer)} packet samples to disk...")
        for packet_sample in packet_buffer:
            pickle.dump(packet_sample, dataset_writer)
        dataset_writer.flush()
        packet_buffer.clear()
        print(f"[Final Save] Completed")
    
    dataset_writer.close()
    
    print(f"\n{'='*80}")
    print(f"[Final Summary]")
    print(f"{'='*80}")
    print(f"✓ Total flow files processed: {file_count}")
    print(f"✓ Successfully processed flows: {processed_flow_count}")
    print(f"✓ Total packet samples generated: {processed_packet_count}")
    print(f"✗ Skipped: {skipped_count}")
    print(f"📁 Dataset saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return processed_packet_count


def verify_dataset(dataset_file, num_samples=5):
    """
    验证数据集文件的内容
    
    参数:
        dataset_file: 数据集文件路径
        num_samples: 要显示的样本数量
    """
    print(f"\n{'='*80}")
    print(f"Verifying dataset: {dataset_file}")
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
        
        print(f"✓ Successfully loaded {len(samples)} samples for verification\n")
        
        for i, sample in enumerate(samples):
            if len(sample) == 4:
                flow_id, packet_idx, packet_tokens, dataset_name = sample
                print(f"样本 {i+1}:")
                print(f"  ✓ flow_id: {flow_id} (type: {type(flow_id).__name__})")
                print(f"  ✓ packet_index: {packet_idx} (type: {type(packet_idx).__name__})")
                print(f"  ✓ packet_tokens: list({len(packet_tokens)} items, 每个token是4位十六进制字符串)")
                print(f"  ✓ dataset_name: {dataset_name}")
                if packet_tokens:
                    print(f"    前10个元素: {packet_tokens[:10]}")
                    print(f"    最后10个元素: {packet_tokens[-10:]}")
            elif len(sample) == 3:
                # 向后兼容旧格式
                flow_id, packet_idx, packet_tokens = sample
                print(f"样本 {i+1}:")
                print(f"  ✓ flow_id: {flow_id} (type: {type(flow_id).__name__})")
                print(f"  ✓ packet_index: {packet_idx} (type: {type(packet_idx).__name__})")
                print(f"  ✓ packet_tokens: list({len(packet_tokens)} items, 每个token是4位十六进制字符串或整数)")
                if packet_tokens:
                    print(f"    前10个元素: {packet_tokens[:10]}")
                    print(f"    最后10个元素: {packet_tokens[-10:]}")
            else:
                print(f"样本 {i+1}: 格式不符 (len={len(sample)})")
            print()
        
        print(f"{'='*80}")
        print("✅ 验证完成！")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ 验证错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset_pretrain.pt from MFR dataset')
    parser.add_argument('--mfr_root', type=str, 
                        default='/root/datasets_new/MFR数据集_预训练_随机化/',
                        help='Root directory of MFR dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='/root/datasets_new/pt_packet_datasets_随机化/',
                        help='Output directory for dataset_pretrain.pt')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of dataset names to process (e.g., NonVPN_MFR NonTor_MFR). '
                             'If not specified, all datasets will be processed.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of flow samples (None for all)')
    parser.add_argument('--processes_num', type=int, default=6,
                        help='Number of processes for parallel processing (default: 6)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Number of files to process before logging progress (default: 1000)')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Number of files to process before saving to disk (default: 100000)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the generated dataset after creation')
    
    args = parser.parse_args()
    
    # 确定要处理的数据集列表
    # 优先级：命令行参数 > 代码中配置 > 处理所有数据集
    if args.datasets is not None:
        datasets_to_process = args.datasets
        print(f"使用命令行参数指定的数据集: {datasets_to_process}")
    elif DATASETS_TO_PROCESS is not None and len(DATASETS_TO_PROCESS) > 0:
        datasets_to_process = DATASETS_TO_PROCESS
        print(f"使用代码中配置的数据集: {datasets_to_process}")
    else:
        datasets_to_process = None
        print("处理所有数据集（未指定数据集列表）")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集所有flow文件
    print("="*80)
    print("Step 1: Collecting flow files from MFR dataset...")
    print("="*80)
    flow_files = collect_all_flow_files(args.mfr_root, dataset_names=datasets_to_process)
    
    if len(flow_files) == 0:
        print(f"❌ 错误: 未找到任何flow文件")
        return
    
    # 处理并生成dataset_pretrain.pt
    print("\n" + "="*80)
    print("Step 2: Processing and generating dataset_pretrain.pt...")
    print("="*80)
    
    output_file = os.path.join(args.output_dir, 'dataset_pretrain_随机化.pt')
    processed_count = process_files_to_dataset_pretrain(
        flow_files, 
        output_file, 
        args.max_samples, 
        args.processes_num, 
        args.log_interval, 
        args.save_interval
    )
    
    # 验证数据集（可选）
    if args.verify:
        print("\n" + "="*80)
        print("Step 3: Verifying generated dataset...")
        print("="*80)
        verify_dataset(output_file, num_samples=5)
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print(f"\n📁 输出文件: {output_file}")
    print(f"📊 生成的样本数: {processed_count}")
    print(f"\n📝 数据格式说明:")
    print(f"  每个样本: (flow_id, packet_index, packet_tokens, dataset_name)")
    print(f"  - flow_id: 8位字符串 (如 '00000123')")
    print(f"  - packet_index: 整数 (1-5，代表数据包在流中的序号)")
    print(f"  - packet_tokens: 160个4位十六进制字符串(2bytes)组成的列表 (如 ['4500', '0020', 'd893', ...])")
    print(f"  - dataset_name: 数据集名称 (如 'NonVPN_MFR'、'NonTor_MFR'、'browser_MFR'、'BenignTraffic_MFR')")
    print(f"\n📌 用于训练时指定: --dataset_path {output_file}")
    print()


if __name__ == '__main__':
    main()

