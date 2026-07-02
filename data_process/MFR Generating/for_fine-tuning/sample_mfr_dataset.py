#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
从MFR数据集中随机抽样.txt文件到MFR抽样数据集
"""

import os
import glob
import random
import shutil
from pathlib import Path

def makedir(path):
    """创建目录，如果已存在则忽略"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        pass

def sample_txt_files(source_dir, target_dir, num_samples=5):
    """
    从源目录中随机抽取指定数量的.txt文件复制到目标目录
    
    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        num_samples: 抽样数量
    
    返回:
        实际复制的文件数量
    """
    # 查找所有.txt文件
    txt_files = glob.glob(os.path.join(source_dir, '*.txt'))
    
    if len(txt_files) == 0:
        return 0
    
    # 随机抽样（如果文件数少于num_samples，就全部复制）
    sample_size = min(len(txt_files), num_samples)
    sampled_files = random.sample(txt_files, sample_size)
    
    # 创建目标目录
    makedir(target_dir)
    
    # 复制文件
    for src_file in sampled_files:
        filename = os.path.basename(src_file)
        dst_file = os.path.join(target_dir, filename)
        shutil.copy2(src_file, dst_file)
    
    return sample_size

def sample_mfr_dataset(source_root, target_root, datasets, num_samples=5):
    """
    从MFR数据集中抽样
    
    参数:
        source_root: 源数据集根目录
        target_root: 目标数据集根目录
        datasets: 要处理的数据集名称列表
        num_samples: 每个子目录抽样的文件数量
    """
    print(f"\n{'='*80}")
    print(f"MFR数据集抽样工具")
    print(f"{'='*80}")
    print(f"源目录: {source_root}")
    print(f"目标目录: {target_root}")
    print(f"每个子目录抽样数量: {num_samples}")
    print(f"{'='*80}\n")
    
    # 创建目标根目录
    makedir(target_root)
    
    total_stats = {
        'total_dirs': 0,
        'total_files': 0,
        'datasets': {}
    }
    
    for dataset_name in datasets:
        dataset_source = os.path.join(source_root, dataset_name)
        dataset_target = os.path.join(target_root, dataset_name)
        
        if not os.path.exists(dataset_source):
            print(f"⚠️  数据集不存在，跳过: {dataset_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_stats = {
            'dirs': 0,
            'files': 0,
            'classes': {}
        }
        
        # 查找所有类别目录
        class_dirs = [d for d in glob.glob(os.path.join(dataset_source, '*')) if os.path.isdir(d)]
        
        for class_dir in sorted(class_dirs):
            class_name = os.path.basename(class_dir)
            
            # 跳过隐藏目录
            if class_name.startswith('.') or class_name.startswith('__'):
                continue
            
            print(f"\n  📁 类别: {class_name}")
            
            class_stats = {
                'packet_mfr': {'train': 0, 'test': 0, 'valid': 0},
                'flow_mfr': {'train': 0, 'test': 0, 'valid': 0}
            }
            
            # 处理 packet_mfr 和 flow_mfr
            for mfr_type in ['packet_mfr', 'flow_mfr']:
                mfr_source = os.path.join(class_dir, mfr_type)
                mfr_target = os.path.join(dataset_target, class_name, mfr_type)
                
                if not os.path.exists(mfr_source):
                    continue
                
                # 处理 train/test/valid
                for split in ['train', 'test', 'valid']:
                    split_source = os.path.join(mfr_source, split)
                    split_target = os.path.join(mfr_target, split)
                    
                    if not os.path.exists(split_source):
                        continue
                    
                    # 抽样并复制文件
                    num_copied = sample_txt_files(split_source, split_target, num_samples)
                    
                    if num_copied > 0:
                        class_stats[mfr_type][split] = num_copied
                        dataset_stats['files'] += num_copied
                        dataset_stats['dirs'] += 1
            
            # 打印类别统计
            packet_total = sum(class_stats['packet_mfr'].values())
            flow_total = sum(class_stats['flow_mfr'].values())
            
            if packet_total > 0 or flow_total > 0:
                print(f"     packet_mfr: train={class_stats['packet_mfr']['train']}, test={class_stats['packet_mfr']['test']}, valid={class_stats['packet_mfr']['valid']} (共{packet_total}个)")
                print(f"     flow_mfr:   train={class_stats['flow_mfr']['train']}, test={class_stats['flow_mfr']['test']}, valid={class_stats['flow_mfr']['valid']} (共{flow_total}个)")
                
                dataset_stats['classes'][class_name] = class_stats
        
        # 数据集统计
        print(f"\n  ✓ {dataset_name} 完成: {dataset_stats['files']} 个文件, {dataset_stats['dirs']} 个子目录")
        
        total_stats['total_dirs'] += dataset_stats['dirs']
        total_stats['total_files'] += dataset_stats['files']
        total_stats['datasets'][dataset_name] = dataset_stats
    
    # 总体统计
    print(f"\n{'='*80}")
    print("抽样完成！")
    print(f"{'='*80}")
    print(f"总共抽样: {total_stats['total_files']} 个文件")
    print(f"涉及子目录: {total_stats['total_dirs']} 个")
    print(f"\n各数据集统计:")
    for dataset_name, stats in total_stats['datasets'].items():
        print(f"  {dataset_name}: {stats['files']} 个文件, {len(stats['classes'])} 个类别")
    
    print(f"\n抽样数据集位置: {target_root}")
    
    return total_stats

if __name__ == '__main__':
    # 配置
    SOURCE_ROOT = "/root/datasets_new/MFR数据集_微调/"
    TARGET_ROOT = "/root/datasets_new/MFR数据集_微调_抽样数据集/"
    
    '''
    DATASETS = [
        "NUDT_MobileTraffic Dataset_all_MFR",
        "CICIoT2023_MFR",
        "ISCXVPN2016_MFR",
        "USTC-TFC2016_MFR", 
        "CSTNET-TLS-1.3_MFR", 
    ]
    '''
    DATASETS = [
        "CSTNET-TLS-1.3_MFR", 
    ]
    
    NUM_SAMPLES = 5  # 每个子目录抽样5个文件
    
    # 执行抽样
    sample_mfr_dataset(SOURCE_ROOT, TARGET_ROOT, DATASETS, NUM_SAMPLES)
    
    print("\n✅ 全部完成!")

