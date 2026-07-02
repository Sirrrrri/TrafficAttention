"""
微调数据集生成脚本
从MFR数据集生成用于BERT微调的TSV格式数据集
直接使用flow_mfr目录下的流MFR矩阵（已经是5个数据包MFR矩阵横向拼接而成）
"""
import os
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 配置参数
class Config:
    # 源数据路径
    SOURCE_BASE_DIR = "/root/datasets/MFR数据集"
    # 目标数据路径
    TARGET_BASE_DIR = "/root/datasets/fine-tuning数据集_Mask"
    
    # 四个数据集配置
    DATASETS = {
        "CICIoT2022_MFR": {
            "source": "CICIoT2022_MFR",
            "target": "fine-tuning_CICIoT2022_MFR",
            "total_samples": 5000  # 数据集总样本数
        },
        "ISCXTor2016_MFR": {
            "source": "ISCXTor2016_MFR",
            "target": "fine-tuning_ISCXTor2016_MFR",
            "total_samples": 14345
        },
        "ISCXVPN2016_MFR": {
            "source": "ISCXVPN2016_MFR",
            "target": "fine-tuning_ISCXVPN2016_MFR",
            "total_samples": 5000
        },
        "USTC-TFC2016_MFR": {
            "source": "USTC-TFC2016_MFR",
            "target": "fine-tuning_USTC-TFC2016_MFR",
            "total_samples": 5000
        }
    }
    
    # 数据集划分比例 (train:dev:test = 8:1:1)
    SPLIT_RATIO = {
        "train": 0.8,  # 80%
        "dev": 0.1,    # 10%
        "test": 0.1    # 10%
    }
    
    # 随机种子
    RANDOM_SEED = 42


def read_flow_mfr_file(file_path):
    """
    读取单个flow MFR文件
    返回: (标签名称, 流MFR矩阵内容)
    
    文件格式:
    第1行: 标签信息（如: CICIoT2022 Attacks_Flood）
    第2-N行: 流MFR矩阵数据
    
    每个token由2 bytes（4位十六进制字符）组成
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        return None, None
    
    # 第一行是标签信息
    label_line = lines[0].strip()
    
    # 后续行是流MFR矩阵数据，将所有行拼接成一个连续的字符串
    mfr_lines = [line.strip() for line in lines[1:] if line.strip()]
    # 去掉所有空格，得到连续的十六进制字符串
    hex_string = "".join(mfr_lines).replace(" ", "")
    
    # 将连续的十六进制字符串按每4个字符（2 bytes）分割成token
    tokens = []
    for i in range(0, len(hex_string), 4):
        token = hex_string[i:i+4]
        if len(token) == 4:  # 只保留完整的4位token
            tokens.append(token)
        elif len(token) > 0:  # 如果有不足4位的余数，可以选择填充或忽略
            # 选项1: 右侧填充0到4位
            tokens.append(token.ljust(4, '0'))
            # 选项2: 忽略不完整的token（注释掉上一行，不添加到tokens）
    
    # 用空格连接所有token
    flow_mfr_content = " ".join(tokens)
    
    return label_line, flow_mfr_content


def collect_all_flow_files(source_dir):
    """
    收集指定数据集下所有类别的所有flow文件（合并train/dev/test）
    返回: {类别名称: [文件路径列表]}
    """
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
            print(f"警告: 目录不存在: {flow_mfr_base}")
            continue
        
        # 收集该类别下所有split（train/valid/test）的txt文件
        all_files = []
        for split in ["train", "valid", "test"]:
            split_dir = flow_mfr_base / split
            if split_dir.exists():
                txt_files = list(split_dir.glob("*.txt"))
                all_files.extend(txt_files)
        
        category_files[category_name] = all_files
        print(f"  类别 {category_name}: 找到 {len(all_files)} 个文件")
    
    return category_files


def calculate_samples_per_class(category_files, total_samples):
    """
    根据每个类别的文件数量，按比例计算每个类别应该贡献的样本数
    返回: {类别名称: 样本数}
    """
    # 统计每个类别的文件总数
    category_counts = {cat: len(files) for cat, files in category_files.items()}
    total_files = sum(category_counts.values())
    
    print(f"\n类别文件数量统计:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} 个文件")
    print(f"  总计: {total_files} 个文件")
    
    # 按比例分配样本数
    samples_per_class = {}
    allocated_samples = 0
    
    for cat in sorted(category_counts.keys()):
        # 按比例计算该类别应分配的样本数
        proportion = category_counts[cat] / total_files
        class_samples = int(total_samples * proportion)
        
        # 确保不超过该类别的实际文件数
        class_samples = min(class_samples, category_counts[cat])
        
        samples_per_class[cat] = class_samples
        allocated_samples += class_samples
    
    # 如果分配的样本数不足，从样本数最多的类别补充
    if allocated_samples < total_samples:
        shortage = total_samples - allocated_samples
        # 找到文件数最多且还有余量的类别
        for cat in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
            if samples_per_class[cat] < category_counts[cat]:
                available = category_counts[cat] - samples_per_class[cat]
                add_samples = min(shortage, available)
                samples_per_class[cat] += add_samples
                allocated_samples += add_samples
                shortage -= add_samples
                if shortage == 0:
                    break
    
    print(f"\n按比例分配样本数:")
    for cat, count in sorted(samples_per_class.items()):
        proportion = count / total_samples * 100
        print(f"  {cat}: {count} 个样本 ({proportion:.2f}%)")
    print(f"  总计: {allocated_samples} 个样本")
    
    return samples_per_class


def sample_and_split_dataset(category_files, samples_per_class, split_ratio):
    """
    从每个类别采样指定数量的样本，并按比例划分为train/dev/test
    返回: {
        'train': [(label_id, content, label_line), ...],
        'dev': [...],
        'test': [...]
    }
    """
    # 创建标签映射
    categories = sorted(category_files.keys())
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    
    # 存储各split的样本
    split_samples = {'train': [], 'dev': [], 'test': []}
    
    print(f"\n开始采样和划分数据集...")
    
    for category_name, file_list in tqdm(category_files.items(), desc="处理类别"):
        n_samples = samples_per_class[category_name]
        
        if n_samples == 0:
            continue
        
        # 随机采样
        if len(file_list) < n_samples:
            print(f"  警告: 类别 {category_name} 文件数量不足 ({len(file_list)} < {n_samples})")
            sampled_files = file_list
        else:
            sampled_files = random.sample(file_list, n_samples)
        
        label_id = label_map[category_name]
        
        # 读取样本内容
        samples = []
        for file_path in sampled_files:
            label_line, flow_mfr_content = read_flow_mfr_file(file_path)
            if label_line is None or not flow_mfr_content:
                continue
            samples.append((label_id, flow_mfr_content, label_line))
        
        # 打乱样本
        random.shuffle(samples)
        
        # 按比例划分
        n_total = len(samples)
        n_train = int(n_total * split_ratio['train'])
        n_dev = int(n_total * split_ratio['dev'])
        
        split_samples['train'].extend(samples[:n_train])
        split_samples['dev'].extend(samples[n_train:n_train+n_dev])
        split_samples['test'].extend(samples[n_train+n_dev:])
    
    # 打乱每个split的样本顺序
    for split in split_samples:
        random.shuffle(split_samples[split])
    
    print(f"\n数据集划分完成:")
    print(f"  训练集: {len(split_samples['train'])} 个样本")
    print(f"  验证集: {len(split_samples['dev'])} 个样本")
    print(f"  测试集: {len(split_samples['test'])} 个样本")
    
    return split_samples, label_map


def write_tsv_file(samples, output_file):
    """
    将样本写入TSV文件
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("label\ttext_a\n")
        
        # 写入数据
        for label_id, flow_repr, original_label in samples:
            f.write(f"{label_id}\t{flow_repr}\n")
    
    print(f"✓ 已生成: {output_file} ({len(samples)} 个样本)")


def process_dataset(dataset_key, config):
    """
    处理单个数据集，生成train/dev/test三个文件
    按照各类别样本数量比例分配，总样本数为total_samples
    """
    dataset_config = config.DATASETS[dataset_key]
    source_dir = os.path.join(config.SOURCE_BASE_DIR, dataset_config["source"])
    target_dir = os.path.join(config.TARGET_BASE_DIR, dataset_config["target"])
    total_samples = dataset_config["total_samples"]
    
    print(f"\n{'='*80}")
    print(f"开始处理数据集: {dataset_key}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"总样本数: {total_samples}")
    print(f"划分比例: train={config.SPLIT_RATIO['train']*100:.0f}%, dev={config.SPLIT_RATIO['dev']*100:.0f}%, test={config.SPLIT_RATIO['test']*100:.0f}%")
    print(f"{'='*80}")
    
    # 创建目标目录
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 收集所有类别的所有文件
    print(f"\n--- 收集文件 ---")
    category_files = collect_all_flow_files(source_dir)
    
    if not category_files:
        print(f"错误: 没有找到任何文件，跳过此数据集")
        return
    
    # 2. 计算每个类别应该采样的数量（按比例分配）
    samples_per_class = calculate_samples_per_class(category_files, total_samples)
    
    # 3. 采样并划分数据集
    split_samples, label_map = sample_and_split_dataset(
        category_files, 
        samples_per_class, 
        config.SPLIT_RATIO
    )
    
    # 4. 写入TSV文件
    print(f"\n--- 写入文件 ---")
    for split in ['train', 'dev', 'test']:
        output_file = Path(target_dir) / f"{split}.tsv"
        write_tsv_file(split_samples[split], output_file)
    
    # 5. 保存标签映射
    label_map_file = Path(target_dir) / "label_map.txt"
    with open(label_map_file, 'w', encoding='utf-8') as f:
        f.write("标签ID\t类别名称\n")
        for cat, idx in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{cat}\n")
    
    print(f"✓ 标签映射已保存: {label_map_file}")
    print(f"\n数据集 {dataset_key} 处理完成!")


def main():
    """主函数"""
    config = Config()
    
    # 设置随机种子
    random.seed(config.RANDOM_SEED)
    
    print("="*80)
    print("微调数据集生成工具")
    print("="*80)
    print(f"随机种子: {config.RANDOM_SEED}")
    print(f"数据集数量: {len(config.DATASETS)}")
    print(f"每个数据集总样本数: {list(config.DATASETS.values())[0]['total_samples']}")
    print(f"数据集划分比例: train={config.SPLIT_RATIO['train']*100:.0f}%, dev={config.SPLIT_RATIO['dev']*100:.0f}%, test={config.SPLIT_RATIO['test']*100:.0f}%")
    print("说明:")
    print("  - 直接使用flow_mfr目录下的流MFR矩阵（已包含5个数据包MFR矩阵横向拼接）")
    print("  - 各类别按照其文件数量比例分配样本")
    print("  - 每个类别的样本按8:1:1划分为训练集、验证集、测试集")
    
    # 处理每个数据集
    for dataset_key in config.DATASETS.keys():
        try:
            process_dataset(dataset_key, config)
        except Exception as e:
            print(f"\n错误: 处理数据集 {dataset_key} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("所有数据集处理完成!")
    print("="*80)


if __name__ == "__main__":
    main()

