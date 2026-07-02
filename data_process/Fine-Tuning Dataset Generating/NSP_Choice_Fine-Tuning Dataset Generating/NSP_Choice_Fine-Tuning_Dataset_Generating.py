"""
微调数据集生成脚本
从MFR数据集生成用于微调的数据集（.pt格式）
- 从每个流MFR矩阵（8行×100token，共800个token）保存完整的流信息
- 每个流的标签来自所属的类别
- 在微调时，直接使用完整流进行分类

微调数据格式:
  (flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)
  
其中：
  - flow_id: 流ID (8位数字字符串)
  - flow_mfr_tokens: 完整流的800个token (8行×100列)
  - category_name: 类别名称 (如 "Idle", "Backdoor")
  - dataset_name: 数据集名称 (如 "CICIoT2022_MFR")
  - label_id: 流量分类标签ID (0~K-1)

微调任务: 给定完整的流MFR矩阵，预测其流量类别标签
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
    SOURCE_BASE_DIR = "/root/datasets/MFR数据集"
    # 目标数据路径
    TARGET_BASE_DIR = "/root/datasets/fine-tuning数据集_NSP"
    
    # 四个数据集配置
    DATASETS = {
        "CICIoT2022_MFR": {
            "source": "CICIoT2022_MFR",
            "target": "finetuning_nsp_CICIoT2022_MFR.pt",
            "total_flows": 5000  # 数据集总流数（每个流拆分为5个数据包）
        },
        "ISCXTor2016_MFR": {
            "source": "ISCXTor2016_MFR",
            "target": "finetuning_nsp_ISCXTor2016_MFR.pt",
            "total_flows": 14345
        },
        "ISCXVPN2016_MFR": {
            "source": "ISCXVPN2016_MFR",
            "target": "finetuning_nsp_ISCXVPN2016_MFR.pt",
            "total_flows": 5000
        },
        "USTC-TFC2016_MFR": {
            "source": "USTC-TFC2016_MFR",
            "target": "finetuning_nsp_USTC-TFC2016_MFR.pt",
            "total_flows": 5000
        }
    }
    
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


def read_flow_mfr_finetune(txt_file, label_map):
    """
    读取单个"流MFR矩阵.txt"文件，保存完整的流MFR矩阵和标签（用于微调）
    
    流MFR矩阵格式：
    - 第1行：标签（如"CICIoT2022 Idle"）
    - 第2-9行：8行×100列的MFR矩阵数据（共800个token）
    
    参数:
        txt_file: 流MFR矩阵文件路径
        label_map: {类别名称: 标签ID} 的字典
    
    返回:
        tuple: (flow_id, flow_mfr_tokens, category_name, dataset_name, label_id)
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
        # 每行100个token，8行共800个token
        flow_tokens = []
        for line in lines[1:9]:  # 取8行
            line = line.strip()
            if line:
                tokens = line.split()
                flow_tokens.extend(tokens)
        
        # 检查是否有800个token
        if len(flow_tokens) != 800:
            return None
        
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
    返回: {类别名称: [文件路径列表]}
    """
    from collections import defaultdict
    category_files = defaultdict(list)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"警告: 源目录不存在: {source_dir}")
        return category_files
    
    # 遍历所有子目录（类别）
    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        flow_mfr_base = category_dir / "flow_mfr"
        
        if not flow_mfr_base.exists():
            print(f"  警告: 目录不存在: {flow_mfr_base}")
            continue
        
        # 收集该类别下所有split（train/valid/test）的txt文件
        all_files = []
        for split in ["train", "valid", "test"]:
            split_dir = flow_mfr_base / split
            if split_dir.exists():
                txt_files = list(split_dir.glob("*.txt"))
                # 将Path对象转换为字符串，确保与read_and_split_flow_to_packets_nsp兼容
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


def process_files_to_dataset_finetune(category_files, flows_per_class, label_map, output_file, 
                                      processes_num=6, log_interval=1000, save_interval=100000):
    """
    将flow txt文件列表处理成微调数据集(.pt格式)，保存完整流MFR矩阵，支持采样指定数量的流
    
    参数:
        category_files: {类别名称: [文件路径列表]}
        flows_per_class: {类别名称: 需要采样的流数}
        label_map: {类别名称: 标签ID}
        output_file: 输出的.pt文件路径
        processes_num: 进程数（默认6）
        log_interval: 日志更新间隔（默认1000个文件）
        save_interval: 数据保存间隔（默认100000个流）
    
    返回:
        int: 处理的流数
    """
    import random as rnd
    
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
    print(f"处理微调数据集")
    print(f"{'='*80}")
    print(f"总流文件数: {total_files}")
    print(f"每个流的token数: 800 (8行×100列的完整MFR矩阵)")
    print(f"使用 {processes_num} 个进程进行并行处理...")
    print(f"输出文件: {output_file}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 创建dataset文件
    dataset_writer = open(output_file, 'wb')
    
    # 样本缓冲区
    flow_buffer = []
    
    # 使用functools.partial创建可序列化的处理函数
    process_with_label_map = partial(read_flow_mfr_finetune, label_map=label_map)
    
    # 使用进程池处理采样后的文件
    with Pool(processes=processes_num) as pool:
        file_count = 0
        
        for flow_sample in pool.imap(process_with_label_map, sampled_file_list):
            file_count += 1
            
            # 处理结果
            if flow_sample is not None:
                # 将流样本添加到缓冲区
                flow_buffer.append(flow_sample)
                processed_flow_count += 1
            else:
                skipped_count += 1
            
            # 每处理log_interval个文件，打印一次日志
            if file_count % log_interval == 0:
                print(f"[进度] {file_count}/{total_files} 个流 | "
                      f"已处理={processed_flow_count}, "
                      f"跳过={skipped_count}")
            
            # 每处理save_interval个流，将缓冲区的数据写入.pt文件
            if file_count % save_interval == 0 and len(flow_buffer) > 0:
                print(f"[保存] 正在写入 {len(flow_buffer)} 个流样本到磁盘...")
                for flow_sample in flow_buffer:
                    pickle.dump(flow_sample, dataset_writer)
                dataset_writer.flush()
                print(f"[保存] 成功保存 | 总处理: {file_count} 个流")
                flow_buffer.clear()
    
    # 处理完所有文件后，写入剩余缓冲区中的数据
    if len(flow_buffer) > 0:
        print(f"\n[最终保存] 正在写入剩余的 {len(flow_buffer)} 个流样本到磁盘...")
        for flow_sample in flow_buffer:
            pickle.dump(flow_sample, dataset_writer)
        dataset_writer.flush()
        flow_buffer.clear()
        print(f"[最终保存] 完成")
    
    dataset_writer.close()
    
    print(f"\n{'='*80}")
    print(f"[最终统计]")
    print(f"{'='*80}")
    print(f"✓ 总共处理的流文件: {file_count}")
    print(f"✓ 成功处理的流: {processed_flow_count}")
    print(f"✗ 跳过的文件数: {skipped_count}")
    print(f"📁 数据集保存到: {output_file}")
    print(f"  每个样本格式: (flow_id, flow_mfr_tokens[800], category_name, dataset_name, label_id)")
    print(f"{'='*80}\n")
    
    return processed_flow_count


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
    
    # 4. 处理并生成微调数据集
    print(f"\n--- 处理并生成数据集 ---")
    output_file = os.path.join(target_dir, dataset_config["target"])
    processed_count = process_files_to_dataset_finetune(
        category_files,
        flows_per_class,
        label_map,
        output_file,
        processes_num=6,
        log_interval=1000,
        save_interval=100000
    )
    
    print(f"\n数据集 {dataset_key} 处理完成!")
    print(f"  ✓ 输出文件: {output_file}")
    print(f"  ✓ 生成的流样本数: {processed_count}")
    print(f"  ✓ 每个流包含: 800个token的完整MFR矩阵 (8行×100列)")
    

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
                print(f"    完整流MFR矩阵 (8行×100列)")
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
                        default='/root/datasets/fine-tuning数据集_NSP',
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
    print("  - 每个流MFR矩阵: 8行×100token = 800个token（完整流）")
    print("  - 保存完整的流MFR矩阵内容和对应的流量类别标签")
    print("  - 各类别按照其文件数量比例分配流数")
    print("  - 输出格式: (flow_id, flow_mfr_tokens[800], category_name, dataset_name, label_id)")
    print("  - 微调任务: 给定完整流，预测其流量分类标签")
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
            output_file = os.path.join(config.TARGET_BASE_DIR, dataset_config["target"])
            if os.path.exists(output_file):
                verify_dataset_finetune(output_file, num_samples=3)
            else:
                print(f"⚠️  文件不存在: {output_file}")
    
    print("\n" + "="*80)
    print("✅ 所有数据集生成完成！")
    print("="*80)
    print(f"\n📁 输出目录: {config.TARGET_BASE_DIR}")
    print(f"\n生成的文件:")
    for dataset_key in config.DATASETS.keys():
        dataset_config = config.DATASETS[dataset_key]
        print(f"  ✓ {dataset_config['target']}")
    
    print(f"\n📝 数据格式说明:")
    print(f"  每个样本: (flow_id, packet_index, packet_tokens, category_name, dataset_name)")
    print(f"  - flow_id: 8位字符串 (如 '00000123')")
    print(f"  - packet_index: 整数 (1-5，代表数据包在流中的序号)")
    print(f"  - packet_tokens: 160个4位十六进制字符串(2bytes)组成的列表 (如 ['4500', '0020', 'd893', ...])")
    print(f"  - category_name: 类别名称 (如 'Idle', 'Backdoor', 等)")
    print(f"  - dataset_name: 数据集名称 (如 'CICIoT2022_MFR')")
    print(f"\n💡 微调时数据包对生成规则:")
    print(f"  - 20%: 来自不同流 -> label=0")
    print(f"  - 80%: 来自同一流 -> label=1")
    print()


if __name__ == "__main__":
    main()

