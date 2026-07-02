#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
将MFR数据集中的flow_mfr数据转换为packet_mfr数据集
从每个流MFR矩阵（8行×100token，共800个token）中提取5个数据包token序列（每个160个token）
每个数据包样本包含：(流ID, 数据包序号, 数据包token列表)

数据格式说明：
- 流ID (flow_id): 8位数字字符串，如 "00000123"
- 数据包序号 (packet_index): 8位数字字符串，如 "00000001" 到 "00000005"
- 数据包token列表 (packet_tokens): 包含160个token的列表，每个token是2字节的十六进制字符串（如 "4500"）

注意：每个token由2个字节（4位十六进制字符）组成，可直接用于模型输入
"""

import os
import glob
import pickle
from tqdm import tqdm
import argparse
from multiprocessing import Pool


def read_and_split_flow_to_packets(txt_file):
    """
    读取单个flow的txt文件，并将其拆分为5个数据包
    每个数据包包含token序列（每个token由2bytes组成）
    
    参数:
        txt_file: 流txt文件路径
    
    返回:
        list: [(flow_id, packet_index, packet_tokens), ...] 
              其中flow_id为8位字符串，packet_index为8位字符串("00000001"-"00000005")，
              packet_tokens为token列表（每个token是2字节的十六进制字符串）
              如果文件无效，返回None
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 第一行是标签（如"CICIoT2022 Idle"），跳过
        # 第2行开始是MFR数据，每行是token序列（空格分隔的2字节十六进制token）
        if len(lines) < 9:  # 至少需要标签行 + 8行MFR数据
            return None
        
        # 从文件名中提取流ID（固定8位数字格式）
        # 文件名格式：flow_00000123.txt -> flow_id = "00000123"
        flow_filename = os.path.basename(txt_file)
        flow_id = flow_filename.replace('flow_', '').replace('.txt', '')
        
        # 确保流ID是8位数字格式
        if len(flow_id) != 8 or not flow_id.isdigit():
            # 如果格式不对，尝试补零到8位
            try:
                flow_id = str(int(flow_id)).zfill(8)
            except:
                return None
        
        # 读取MFR token数据（从第2行开始，共8行）
        # 每行100个token（每个token是2字节），8行共800个token
        flow_tokens = []
        for line in lines[1:9]:  # 取8行
            line = line.strip()
            if line:
                # 分割得到token列表（每个token是4位十六进制，代表2字节）
                tokens = line.split()
                flow_tokens.extend(tokens)
        
        # 检查token总数（应该是800个：8行×100token/行）
        if len(flow_tokens) != 800:
            return None
        
        # 拆分为5个数据包token序列（每个数据包160个token：8行×20token/行）
        packet_samples = []
        for packet_idx in range(1, 6):  # 1到5
            # 计算当前数据包的token范围（每个数据包160个token）
            start_idx = (packet_idx - 1) * 160
            end_idx = packet_idx * 160
            
            # 提取数据包token序列
            packet_tokens = flow_tokens[start_idx:end_idx]
            
            # 将数据包序号转换为8位字符串格式
            packet_index_str = str(packet_idx).zfill(8)  # 1 -> "00000001"
            
            # 创建数据包样本：(流ID, 数据包序号, 数据包token列表)
            packet_sample = (flow_id, packet_index_str, packet_tokens)
            packet_samples.append(packet_sample)
        
        return packet_samples
    
    except Exception as e:
        return None


def collect_all_flow_files(mfr_root):
    """
    收集所有flow_mfr目录下的txt文件
    
    参数:
        mfr_root: MFR数据集根目录
    
    返回:
        dict: {
            'train': [file_paths],
            'test': [file_paths],
            'valid': [file_paths]
        }
    """
    flow_files = {
        'train': [],
        'test': [],
        'valid': []
    }
    
    # 查找所有flow_mfr目录
    flow_mfr_dirs = glob.glob(os.path.join(mfr_root, '**', 'flow_mfr'), recursive=True)
    
    print(f"Found {len(flow_mfr_dirs)} flow_mfr directories")
    
    for flow_dir in flow_mfr_dirs:
        # 收集train文件
        train_dir = os.path.join(flow_dir, 'train')
        if os.path.exists(train_dir):
            train_files = glob.glob(os.path.join(train_dir, 'flow_*.txt'))
            flow_files['train'].extend(train_files)
        
        # 收集test文件
        test_dir = os.path.join(flow_dir, 'test')
        if os.path.exists(test_dir):
            test_files = glob.glob(os.path.join(test_dir, 'flow_*.txt'))
            flow_files['test'].extend(test_files)
        
        # 收集valid文件
        valid_dir = os.path.join(flow_dir, 'valid')
        if os.path.exists(valid_dir):
            valid_files = glob.glob(os.path.join(valid_dir, 'flow_*.txt'))
            flow_files['valid'].extend(valid_files)
    
    print(f"Total train files: {len(flow_files['train'])}")
    print(f"Total test files: {len(flow_files['test'])}")
    print(f"Total valid files: {len(flow_files['valid'])}")
    
    return flow_files


def process_files_to_packet_dataset(file_list, output_file, max_samples=None, 
                                    processes_num=6, log_interval=1000, save_interval=100000):
    """
    将flow txt文件列表处理成packet数据集.pt格式文件
    
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
    
    print(f"Total flow files to process: {total_files}")
    print(f"Expected packet samples: {total_files * 5} (5 packets per flow)")
    print(f"Using {processes_num} processes for parallel processing...")
    print(f"Log interval: every {log_interval} files")
    print(f"Save interval: every {save_interval} files")
    
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
                      f"Skipped={skipped_count} | Buffer size={len(packet_buffer)}")
            
            # 每处理save_interval个文件，将缓冲区的数据追加写入.pt文件
            if file_count % save_interval == 0 and len(packet_buffer) > 0:
                print(f"[Save] Writing {len(packet_buffer)} packet samples to disk...")
                for packet_sample in packet_buffer:
                    pickle.dump(packet_sample, dataset_writer)
                dataset_writer.flush()
                print(f"[Save] Successfully saved {len(packet_buffer)} packet samples | "
                      f"Total processed: {file_count} flows")
                packet_buffer.clear()
    
    # 处理完所有文件后，写入剩余缓冲区中的数据
    if len(packet_buffer) > 0:
        print(f"[Final Save] Writing remaining {len(packet_buffer)} packet samples to disk...")
        for packet_sample in packet_buffer:
            pickle.dump(packet_sample, dataset_writer)
        dataset_writer.flush()
        packet_buffer.clear()
        print(f"[Final Save] Completed")
    
    dataset_writer.close()
    print(f"\n[Final Summary]")
    print(f"  Total flow files processed: {file_count}")
    print(f"  Successfully processed flows: {processed_flow_count}")
    print(f"  Total packet samples generated: {processed_packet_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Dataset saved to: {output_file}")
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
    print(f"{'='*80}")
    
    try:
        with open(dataset_file, 'rb') as f:
            samples = []
            try:
                while len(samples) < num_samples:
                    sample = pickle.load(f)
                    samples.append(sample)
            except EOFError:
                pass
        
        print(f"Successfully loaded {len(samples)} samples for verification")
        
        for i, (flow_id, packet_idx, packet_tokens) in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  Flow ID: {flow_id} (8-digit string)")
            print(f"  Packet Index: {packet_idx} (8-digit string)")
            print(f"  Number of tokens: {len(packet_tokens)}")
            print(f"  First 10 tokens: {' '.join(packet_tokens[:10])}")
            print(f"  Last 10 tokens: {' '.join(packet_tokens[-10:])}")
        
        print(f"\n{'='*80}")
        print("Verification completed successfully!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during verification: {e}")


def main():
    parser = argparse.ArgumentParser(description='Prepare packet MFR dataset from flow MFR data')
    parser.add_argument('--mfr_root', type=str, 
                        default='/root/datasets/MFR数据集',
                        help='Root directory of MFR dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='/root/datasets/pt_packet_datasets',
                        help='Output directory for packet dataset files')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum number of training flow samples (None for all)')
    parser.add_argument('--max_test_samples', type=int, default=None,
                        help='Maximum number of test flow samples (None for all)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                        help='Maximum number of validation flow samples (None for all)')
    parser.add_argument('--processes_num', type=int, default=6,
                        help='Number of processes for parallel processing (default: 6)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Number of files to process before logging progress (default: 1000)')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Number of files to process before saving to disk (default: 100000)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the generated datasets after creation')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集所有flow文件
    print("="*80)
    print("Step 1: Collecting flow files...")
    print("="*80)
    flow_files = collect_all_flow_files(args.mfr_root)
    
    # 处理训练集
    print("\n" + "="*80)
    print("Step 2: Processing training dataset...")
    print("="*80)
    train_dataset = os.path.join(args.output_dir, 'packet_dataset_train.pt')
    process_files_to_packet_dataset(flow_files['train'], train_dataset, 
                                    args.max_train_samples, args.processes_num, 
                                    args.log_interval, args.save_interval)
    
    # 处理测试集
    print("\n" + "="*80)
    print("Step 3: Processing test dataset...")
    print("="*80)
    test_dataset = os.path.join(args.output_dir, 'packet_dataset_test.pt')
    process_files_to_packet_dataset(flow_files['test'], test_dataset, 
                                    args.max_test_samples, args.processes_num, 
                                    args.log_interval, args.save_interval)
    
    # 处理验证集
    print("\n" + "="*80)
    print("Step 4: Processing validation dataset...")
    print("="*80)
    valid_dataset = os.path.join(args.output_dir, 'packet_dataset_valid.pt')
    process_files_to_packet_dataset(flow_files['valid'], valid_dataset, 
                                    args.max_valid_samples, args.processes_num, 
                                    args.log_interval, args.save_interval)
    
    # 验证数据集（可选）
    if args.verify:
        print("\n" + "="*80)
        print("Step 5: Verifying generated datasets...")
        print("="*80)
        verify_dataset(train_dataset, num_samples=5)
        verify_dataset(test_dataset, num_samples=3)
        verify_dataset(valid_dataset, num_samples=3)
    
    print("\n" + "="*80)
    print("All Done!")
    print("="*80)
    print(f"Training dataset: {train_dataset}")
    print(f"Test dataset: {test_dataset}")
    print(f"Validation dataset: {valid_dataset}")
    print("\nDataset format:")
    print("  Each sample: (flow_id, packet_index, packet_tokens)")
    print("  - flow_id: 8-digit string (e.g., '00000123')")
    print("  - packet_index: 8-digit string ('00000001' to '00000005')")
    print("  - packet_tokens: list of 160 tokens, each token is a 2-byte hex string (e.g., '4500')")
    print("\nNext step: Use these packet datasets for pretraining")


if __name__ == '__main__':
    main()

