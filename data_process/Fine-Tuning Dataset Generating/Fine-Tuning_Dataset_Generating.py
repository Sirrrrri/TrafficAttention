#!/usr/bin/python3

import os
import re
import pickle
import argparse
from functools import partial
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


class Config:
    SOURCE_BASE_DIR = "/path/to/mfr_dataset/"
    TARGET_BASE_DIR = "/path/to/finetuning_output/"

    DATASETS = {
        "NUDT_MobileTraffic Dataset_all_MFR": {
            "source": "NUDT_MobileTraffic Dataset_all_MFR",
            "target": "finetuning_NUDT_MobileTraffic Dataset_all_MFR.pt",
            "total_flows": 41975
        },
        "USTC-TFC2016_MFR": {
            "source": "USTC-TFC2016_MFR",
            "target": "finetuning_USTC-TFC2016_MFR.pt",
            "total_flows": 19387
        },
        "ISCXVPN2016_MFR": {
            "source": "ISCXVPN2016_MFR",
            "target": "finetuning_ISCXVPN2016_MFR.pt",
            "total_flows": 13520
        },
        "CICIoT2023_MFR": {
            "source": "CICIoT2023_MFR",
            "target": "finetuning_CICIoT2023_MFR.pt",
            "total_flows": 25453
        },
        "ISCXVPN2016-APP_MFR": {
            "source": "ISCXVPN2016-APP_MFR",
            "target": "finetuning_ISCXVPN2016-APP_MFR.pt",
            "total_flows": 11592
        }
    }

    RANDOM_SEED = 42


def extract_dataset_name(file_path):
    match = re.search(r'([A-Za-z0-9_]+_MFR)', file_path)
    if match:
        return match.group(1)
    return "Unknown"


def extract_category_name(file_path):
    parts = Path(file_path).parts
    try:
        flow_mfr_idx = parts.index('flow_mfr')
        if flow_mfr_idx > 0:
            return parts[flow_mfr_idx - 1]
    except (ValueError, IndexError):
        pass
    return "Unknown"


def read_flow_mfr_finetune(txt_file, label_map):
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 9:
            return None
        flow_filename = os.path.basename(txt_file)
        if flow_filename.startswith('flow_') and flow_filename.endswith('.txt'):
            flow_id = flow_filename.replace('flow_', '').replace('.txt', '')
        else:
            match = re.search(r'flow_(\d+)', txt_file)
            if match:
                flow_id = match.group(1)
            else:
                return None
        if not flow_id.isdigit():
            return None
        flow_id = flow_id.zfill(8)
        dataset_name = extract_dataset_name(txt_file)
        category_name = extract_category_name(txt_file)
        flow_tokens = []
        for line in lines[1:9]:
            line = line.strip()
            if line:
                flow_tokens.extend(line.split())
        if len(flow_tokens) != 800:
            return None
        packet_tokens = []
        for i in range(4):
            start_idx = i * 160
            packet_tokens.extend(flow_tokens[start_idx:start_idx + 80])
        flow_tokens = packet_tokens
        label_id = label_map.get(category_name, -1)
        if label_id < 0:
            return None
        return (flow_id, flow_tokens, category_name, dataset_name, label_id)
    except Exception:
        return None


def collect_flow_files_by_category(source_dir):
    from collections import defaultdict
    category_files = defaultdict(list)
    source_path = Path(source_dir)
    if not source_path.exists():
        return category_files
    flow_mfr_direct = source_path / "flow_mfr"
    if flow_mfr_direct.exists() and flow_mfr_direct.is_dir():
        all_files = []
        for split in ["train", "valid", "test"]:
            split_dir = flow_mfr_direct / split
            if split_dir.exists():
                all_files.extend([str(f) for f in split_dir.glob("flow_*.txt")])
        category_files[source_path.name] = all_files
        return category_files
    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue
        flow_mfr_base = category_dir / "flow_mfr"
        if not flow_mfr_base.exists():
            continue
        all_files = []
        for split in ["train", "valid", "test"]:
            split_dir = flow_mfr_base / split
            if split_dir.exists():
                all_files.extend([str(f) for f in split_dir.glob("flow_*.txt")])
        category_files[category_dir.name] = all_files
    return category_files


def calculate_flows_per_class(category_files, total_flows):
    category_counts = {cat: len(files) for cat, files in category_files.items()}
    total_files = sum(category_counts.values())
    flows_per_class = {}
    allocated_flows = 0
    for cat in sorted(category_counts.keys()):
        class_flows = min(int(total_flows * category_counts[cat] / total_files), category_counts[cat])
        flows_per_class[cat] = class_flows
        allocated_flows += class_flows
    if allocated_flows < total_flows:
        shortage = total_flows - allocated_flows
        for cat in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
            if flows_per_class[cat] < category_counts[cat]:
                add_flows = min(shortage, category_counts[cat] - flows_per_class[cat])
                flows_per_class[cat] += add_flows
                allocated_flows += add_flows
                shortage -= add_flows
                if shortage == 0:
                    break
    return flows_per_class


def collect_all_flow_files_nsp(mfr_root):
    import glob
    all_files = []
    flow_mfr_dirs = glob.glob(os.path.join(mfr_root, '**', 'flow_mfr'), recursive=True)
    for flow_dir in flow_mfr_dirs:
        flow_files = glob.glob(os.path.join(flow_dir, '**', 'flow_*.txt'), recursive=True)
        all_files.extend(flow_files)
    return all_files


def process_files_to_dataset_finetune(category_files, flows_per_class, label_map, output_base_path,
                                      processes_num=6, log_interval=1000, save_interval=100000,
                                      train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, random_seed=42):
    import random as rnd
    rnd.seed(random_seed)
    sampled_file_list = []
    for category_name, file_list in category_files.items():
        n_flows = flows_per_class[category_name]
        if n_flows == 0:
            continue
        if len(file_list) < n_flows:
            sampled_file_list.extend(file_list)
        else:
            sampled_file_list.extend(rnd.sample(file_list, n_flows))
    rnd.shuffle(sampled_file_list)
    processed_flow_count = 0
    skipped_count = 0
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
    train_writer = open(f"{output_base_path}_train.pt", 'wb')
    dev_writer = open(f"{output_base_path}_dev.pt", 'wb')
    test_writer = open(f"{output_base_path}_test.pt", 'wb')
    process_with_label_map = partial(read_flow_mfr_finetune, label_map=label_map)
    all_flow_samples = []
    with Pool(processes=processes_num) as pool:
        for flow_sample in pool.imap(process_with_label_map, sampled_file_list):
            if flow_sample is not None:
                all_flow_samples.append(flow_sample)
                processed_flow_count += 1
            else:
                skipped_count += 1
    total_samples = len(all_flow_samples)
    train_end = int(total_samples * train_ratio)
    dev_end = train_end + int(total_samples * dev_ratio)
    train_samples = all_flow_samples[:train_end]
    dev_samples = all_flow_samples[train_end:dev_end]
    test_samples = all_flow_samples[dev_end:]
    for flow_sample in train_samples:
        pickle.dump(flow_sample, train_writer)
    train_writer.flush()
    train_writer.close()
    for flow_sample in dev_samples:
        pickle.dump(flow_sample, dev_writer)
    dev_writer.flush()
    dev_writer.close()
    for flow_sample in test_samples:
        pickle.dump(flow_sample, test_writer)
    test_writer.flush()
    test_writer.close()
    return len(train_samples), len(dev_samples), len(test_samples)


def process_dataset(dataset_key, config):
    import random as rnd
    dataset_config = config.DATASETS[dataset_key]
    source_dir = os.path.join(config.SOURCE_BASE_DIR, dataset_config["source"])
    target_dir = config.TARGET_BASE_DIR
    total_flows = dataset_config["total_flows"]
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    category_files = collect_flow_files_by_category(source_dir)
    if not category_files:
        return
    flows_per_class = calculate_flows_per_class(category_files, total_flows)
    categories = sorted(category_files.keys())
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    output_base = os.path.join(target_dir, dataset_config["target"].replace('.pt', ''))
    return process_files_to_dataset_finetune(
        category_files, flows_per_class, label_map, output_base,
        processes_num=6, log_interval=1000, save_interval=100000,
        train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1,
        random_seed=config.RANDOM_SEED
    )


def verify_dataset_finetune(dataset_file, num_samples=5):
    try:
        with open(dataset_file, 'rb') as f:
            samples = []
            try:
                while len(samples) < num_samples:
                    samples.append(pickle.load(f))
            except EOFError:
                pass
        for sample in samples:
            if len(sample) == 5:
                flow_id, flow_mfr_tokens, category_name, dataset_name, label_id = sample
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/path/to/finetuning_output/')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config = Config()
    config.TARGET_BASE_DIR = args.output_dir
    import random as rnd
    rnd.seed(config.RANDOM_SEED)
    for dataset_key in config.DATASETS.keys():
        try:
            process_dataset(dataset_key, config)
        except Exception:
            import traceback
            traceback.print_exc()
            continue
    if args.verify:
        for dataset_key in config.DATASETS.keys():
            dataset_config = config.DATASETS[dataset_key]
            output_base = os.path.join(config.TARGET_BASE_DIR, dataset_config["target"].replace('.pt', ''))
            for split_file in [f"{output_base}_train.pt", f"{output_base}_dev.pt", f"{output_base}_test.pt"]:
                if os.path.exists(split_file):
                    verify_dataset_finetune(split_file, num_samples=2)


if __name__ == "__main__":
    main()

