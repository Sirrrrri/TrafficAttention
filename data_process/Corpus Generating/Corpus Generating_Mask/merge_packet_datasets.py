#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
合并packet数据集的train、test、valid文件为一个预训练数据集
批量处理模式：每读500000个样本，写入500000个样本
"""

import os
import pickle
import argparse


def merge_datasets_batch(input_files, output_file, batch_size=500000):
    """
    批量合并数据集文件，每读batch_size个样本，写入batch_size个样本
    
    参数:
        input_files: 输入文件列表
        output_file: 输出文件路径
        batch_size: 批处理大小（每次读写多少个样本）
    """
    print(f"\nMerging datasets in batch mode...")
    print(f"Batch size: {batch_size} samples")
    print(f"Output file: {output_file}")
    
    total_count = 0
    batch_buffer = []
    
    with open(output_file, 'wb') as out_f:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: File not found: {input_file}, skipping...")
                continue
            
            print(f"\nProcessing {input_file}...")
            file_count = 0
            
            with open(input_file, 'rb') as in_f:
                try:
                    while True:
                        # 读取一个样本到缓冲区
                        sample = pickle.load(in_f)
                        batch_buffer.append(sample)
                        
                        file_count += 1
                        total_count += 1
                        
                        # 每10000个样本打印一次进度
                        if total_count % 10000 == 0:
                            print(f"  [Reading] Read {total_count} samples (buffer: {len(batch_buffer)})")
                        
                        # 当缓冲区达到batch_size时，批量写入
                        if len(batch_buffer) >= batch_size:
                            print(f"  [Writing] Writing {len(batch_buffer)} samples to disk...")
                            for sample in batch_buffer:
                                pickle.dump(sample, out_f)
                            out_f.flush()
                            print(f"  [SAVED] Wrote and flushed {len(batch_buffer)} samples, total: {total_count}")
                            batch_buffer.clear()
                            
                except EOFError:
                    pass
            
            print(f"  Completed reading {input_file}: {file_count} samples")
        
        # 写入剩余的样本
        if len(batch_buffer) > 0:
            print(f"\n[Writing] Writing remaining {len(batch_buffer)} samples to disk...")
            for sample in batch_buffer:
                pickle.dump(sample, out_f)
            out_f.flush()
            print(f"[SAVED] Wrote and flushed remaining {len(batch_buffer)} samples")
            batch_buffer.clear()
    
    return total_count


def merge_datasets(input_dir, output_file, batch_size=500000):
    """
    合并数据集文件（批量处理模式）
    
    参数:
        input_dir: 输入目录
        output_file: 输出文件路径
        batch_size: 批处理大小（每读batch_size个样本，写入batch_size个样本）
    """
    print("="*80)
    print("Merging packet datasets (Batch mode)")
    print("="*80)
    print(f"Batch size: {batch_size} samples")
    print(f"Processing: Read {batch_size} samples → Write {batch_size} samples → Repeat")
    
    # 定义要合并的文件
    files_to_merge = [
        os.path.join(input_dir, 'packet_dataset_train.pt'),
        os.path.join(input_dir, 'packet_dataset_test.pt'),
        os.path.join(input_dir, 'packet_dataset_valid.pt')
    ]
    
    # 过滤存在的文件
    existing_files = [f for f in files_to_merge if os.path.exists(f)]
    
    if not existing_files:
        print("Error: No dataset files found!")
        return
    
    print(f"\nFiles to merge: {len(existing_files)}")
    for f in existing_files:
        print(f"  - {f}")
    
    # 批量合并数据集
    print("\n" + "="*80)
    print("Starting batch merge...")
    print("="*80)
    
    total_samples = merge_datasets_batch(existing_files, output_file, batch_size)
    
    print("\n" + "="*80)
    print("Merge completed!")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Total samples: {total_samples}")
    
    # 计算文件大小
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        file_size_gb = file_size / (1024 * 1024 * 1024)
        if file_size_gb >= 1:
            print(f"File size: {file_size_gb:.2f} GB")
        else:
            print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Merge packet datasets into one pretrain dataset (batch mode)')
    parser.add_argument('--input_dir', type=str,
                        default='/root/datasets/pt_packet_datasets',
                        help='Input directory containing packet dataset files')
    parser.add_argument('--output_file', type=str,
                        default='/root/datasets/pt_packet_datasets/dataset_pretrain.pt',
                        help='Output merged dataset file')
    parser.add_argument('--batch_size', type=int, default=500000,
                        help='Batch size: read and write N samples at a time (default: 500000)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并数据集（批量处理）
    merge_datasets(args.input_dir, args.output_file, args.batch_size)


if __name__ == '__main__':
    main()

