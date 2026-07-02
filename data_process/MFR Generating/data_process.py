#!/usr/bin/python3

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
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def filter_tcp_handshake(packets):
    if len(packets) == 0:
        return packets
    is_tcp = any('TCP' in pkt for pkt in packets)
    if is_tcp and len(packets) > 3:
        return packets[3:]
    elif is_tcp:
        return []
    return packets


def extract_packet_mfr(packet):
    try:
        if 'IP' not in packet:
            return None
        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        return (header, payload)
    except:
        return None


def read_packet_matrices(pcap_path, max_packets=1000):
    try:
        packets = scapy.rdpcap(pcap_path, count=max_packets)
    except Exception:
        return []
    packet_matrices = []
    for packet in packets:
        try:
            mfr_data = extract_packet_mfr(packet)
            if mfr_data is None:
                continue
            header, payload = mfr_data
            matrix_rows = []
            for row_idx in range(2):
                start = row_idx * 80
                row_hex = header[start:start+80]
                tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                matrix_rows.append(' '.join(tokens))
            for row_idx in range(6):
                start = row_idx * 80
                row_hex = payload[start:start+80]
                tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                matrix_rows.append(' '.join(tokens))
            packet_matrices.append(matrix_rows)
        except Exception:
            continue
    return packet_matrices


def read_flow_matrix(pcap_path):
    try:
        packets = scapy.rdpcap(pcap_path, count=8)
    except Exception:
        return None
    packets = filter_tcp_handshake(list(packets))
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
    if len(packet_data) < 5:
        for i in range(5 - len(packet_data)):
            packet_data.append(('0' * 160, '0' * 480))
    try:
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
        return flow_matrix_rows
    except Exception:
        return None


def read_MFR_bytes(pcap_path):
    try:
        packets = scapy.rdpcap(pcap_path, count=8)
    except Exception:
        return None
    packets = filter_tcp_handshake(list(packets))
    data = []
    for packet in packets:
        mfr_data = extract_packet_mfr(packet)
        if mfr_data is None:
            continue
        data.append(mfr_data)
        if len(data) >= 5:
            break
    if len(data) < 5:
        for i in range(5 - len(data)):
            data.append(('0' * 160, '0' * 480))
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data


def read_MFR_sequence(pcap_path):
    mfr_hex = read_MFR_bytes(pcap_path)
    if mfr_hex is None:
        return None
    return [int(mfr_hex[i:i+4], 16) for i in range(0, len(mfr_hex), 4)]


def read_MFR_token_sequence(pcap_path):
    mfr_hex = read_MFR_bytes(pcap_path)
    if mfr_hex is None:
        return None
    return ' '.join([mfr_hex[i:i+4] for i in range(0, len(mfr_hex), 4)])


def process_pcap_dataset(pcap_root_path, output_path, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    class_dirs = [d for d in glob.glob(os.path.join(pcap_root_path, '*')) if os.path.isdir(d)]
    if len(class_dirs) == 0:
        return
    dataset = {
        'train': defaultdict(lambda: {'samples': 0, 'payload': {}}),
        'test': defaultdict(lambda: {'samples': 0, 'payload': {}}),
        'valid': defaultdict(lambda: {'samples': 0, 'payload': {}})
    }
    class_to_id = {}
    for class_idx, class_dir in enumerate(tqdm(class_dirs)):
        class_name = os.path.basename(class_dir)
        class_to_id[class_name] = class_idx
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        random.shuffle(pcap_files)
        n_total = len(pcap_files)
        n_train = int(n_total * train_ratio)
        n_test = int(n_total * test_ratio)
        train_files = pcap_files[:n_train]
        test_files = pcap_files[n_train:n_train+n_test]
        valid_files = pcap_files[n_train+n_test:]
        for pcap_file in tqdm(train_files, leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['train'][class_idx]['samples'] + 1)
                dataset['train'][class_idx]['payload'][sample_id] = token_seq
                dataset['train'][class_idx]['samples'] += 1
        for pcap_file in tqdm(test_files, leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['test'][class_idx]['samples'] + 1)
                dataset['test'][class_idx]['payload'][sample_id] = token_seq
                dataset['test'][class_idx]['samples'] += 1
        for pcap_file in tqdm(valid_files, leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['valid'][class_idx]['samples'] + 1)
                dataset['valid'][class_idx]['payload'][sample_id] = token_seq
                dataset['valid'][class_idx]['samples'] += 1
    output_data = {
        'metadata': {
            'class_mapping': class_to_id,
            'num_classes': len(class_to_id)
        },
        'train': {str(k): dict(v) for k, v in dataset['train'].items()},
        'test': {str(k): dict(v) for k, v in dataset['test'].items()},
        'valid': {str(k): dict(v) for k, v in dataset['valid'].items()}
    }
    makedir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    return output_data


def convert_json_to_npy(json_path, output_dir):
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
                token_seq = [int(x, 16) for x in payload.split()]
                all_sequences.append(token_seq)
                all_labels.append(int(class_id))
        np.save(os.path.join(output_dir, f'{split}_sequences.npy'), np.array(all_sequences, dtype=object))
        np.save(os.path.join(output_dir, f'{split}_labels.npy'), np.array(all_labels))


def generate_vocab_from_mfr():
    vocab = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[CLS]': 2,
        '[SEP]': 3,
        '[MASK]': 4,
    }
    for i in range(65536):
        hex_token = format(i, '04x')
        vocab[hex_token] = i + 5
    return vocab


def save_vocab(vocab_path):
    vocab = generate_vocab_from_mfr()
    makedir(os.path.dirname(vocab_path))
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")


def process_single_pcap(pcap_file, dataset_name, class_name):
    try:
        try:
            packets = scapy.rdpcap(pcap_file, count=1000)
        except Exception:
            return (None, None)
        packets = filter_tcp_handshake(list(packets))
        packet_matrices = []
        for packet in packets:
            try:
                mfr_data = extract_packet_mfr(packet)
                if mfr_data is None:
                    continue
                header, payload = mfr_data
                matrix_rows = []
                for row_idx in range(2):
                    start = row_idx * 80
                    row_hex = header[start:start+80]
                    tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                    matrix_rows.append(' '.join(tokens))
                for row_idx in range(6):
                    start = row_idx * 80
                    row_hex = payload[start:start+80]
                    tokens = [row_hex[i:i+4] for i in range(0, len(row_hex), 4)]
                    matrix_rows.append(' '.join(tokens))
                packet_matrices.append(matrix_rows)
            except Exception:
                continue
        flow_matrix = None
        try:
            packet_data = []
            for packet in packets[:5]:
                try:
                    mfr_data = extract_packet_mfr(packet)
                    if mfr_data is None:
                        continue
                    packet_data.append(mfr_data)
                    if len(packet_data) >= 5:
                        break
                except Exception:
                    continue
            if len(packet_data) < 5:
                for i in range(5 - len(packet_data)):
                    packet_data.append(('0' * 160, '0' * 480))
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
        except Exception:
            pass
        return (packet_matrices, flow_matrix)
    except MemoryError:
        return (None, None)
    except KeyboardInterrupt:
        raise
    except Exception:
        return (None, None)


def process_class_mfr_matrices(class_dir, output_dir, dataset_name, class_name, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1, num_workers=6, batch_size=10000, log_interval=100):
    pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
    if len(pcap_files) == 0:
        return None
    random.shuffle(pcap_files)
    n_total = len(pcap_files)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    train_files = pcap_files[:n_train]
    test_files = pcap_files[n_train:n_train+n_test]
    valid_files = pcap_files[n_train+n_test:]
    makedir(output_dir)
    packet_base_dir = os.path.join(output_dir, 'packet_mfr')
    flow_base_dir = os.path.join(output_dir, 'flow_mfr')
    for split in ['train', 'test', 'valid']:
        makedir(os.path.join(packet_base_dir, split))
        makedir(os.path.join(flow_base_dir, split))
    sample_counts = {
        'train': {'packet': 0, 'flow': 0},
        'test': {'packet': 0, 'flow': 0},
        'valid': {'packet': 0, 'flow': 0}
    }

    def save_packet_txt(matrix, split_name, packet_id):
        filepath = os.path.join(packet_base_dir, split_name, f"packet_{packet_id:08d}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{dataset_name} {class_name}\n")
            for row in matrix:
                f.write(f"{row}\n")

    def save_flow_txt(matrix, split_name, flow_id):
        filepath = os.path.join(flow_base_dir, split_name, f"flow_{flow_id:08d}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{dataset_name} {class_name}\n")
            for row in matrix:
                f.write(f"{row}\n")

    def process_files_parallel(files, split_name):
        processed_count = 0
        flow_success_count = 0
        flow_fail_count = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_pcap, pcap_file, dataset_name, class_name): pcap_file
                      for pcap_file in files}
            for future in as_completed(futures, timeout=None):
                try:
                    packet_matrices, flow_matrix = future.result(timeout=60)
                    if packet_matrices:
                        for matrix in packet_matrices:
                            sample_counts[split_name]['packet'] += 1
                            save_packet_txt(matrix, split_name, sample_counts[split_name]['packet'])
                    if flow_matrix is not None:
                        sample_counts[split_name]['flow'] += 1
                        save_flow_txt(flow_matrix, split_name, sample_counts[split_name]['flow'])
                        flow_success_count += 1
                    else:
                        flow_fail_count += 1
                    processed_count += 1
                except (TimeoutError, Exception):
                    flow_fail_count += 1

    process_files_parallel(train_files, 'train')
    process_files_parallel(test_files, 'test')
    process_files_parallel(valid_files, 'valid')
    return {
        'packet_samples': sample_counts['train']['packet'] + sample_counts['test']['packet'] + sample_counts['valid']['packet'],
        'flow_samples': sample_counts['train']['flow'] + sample_counts['test']['flow'] + sample_counts['valid']['flow'],
        'counts': sample_counts
    }


def process_single_class_directory(class_dir, output_path, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
    if len(pcap_files) == 0:
        return None
    class_name = os.path.basename(class_dir)
    random.shuffle(pcap_files)
    n_total = len(pcap_files)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    train_files = pcap_files[:n_train]
    test_files = pcap_files[n_train:n_train+n_test]
    valid_files = pcap_files[n_train+n_test:]
    dataset = {
        'train': {'samples': 0, 'payload': {}},
        'test': {'samples': 0, 'payload': {}},
        'valid': {'samples': 0, 'payload': {}}
    }
    for pcap_file in tqdm(train_files, leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['train']['samples'] + 1)
            dataset['train']['payload'][sample_id] = token_seq
            dataset['train']['samples'] += 1
    for pcap_file in tqdm(test_files, leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['test']['samples'] + 1)
            dataset['test']['payload'][sample_id] = token_seq
            dataset['test']['samples'] += 1
    for pcap_file in tqdm(valid_files, leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['valid']['samples'] + 1)
            dataset['valid']['payload'][sample_id] = token_seq
            dataset['valid']['samples'] += 1
    output_data = {
        'metadata': {'class_name': class_name, 'num_classes': 1},
        'train': dataset['train'],
        'test': dataset['test'],
        'valid': dataset['valid']
    }
    makedir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    return output_data


def process_presplit_dataset(dataset_root, dataset_name, output_dir):
    dataset_path = os.path.join(dataset_root, dataset_name)
    if not os.path.exists(dataset_path):
        return None
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    if not os.path.exists(train_dir):
        return None
    dataset = {
        'train': defaultdict(lambda: {'samples': 0, 'payload': {}}),
        'test': defaultdict(lambda: {'samples': 0, 'payload': {}})
    }
    class_to_id = {}
    train_classes = [d for d in glob.glob(os.path.join(train_dir, '*')) if os.path.isdir(d)]
    for class_idx, class_dir in enumerate(tqdm(train_classes)):
        class_name = os.path.basename(class_dir)
        class_to_id[class_name] = class_idx
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        for pcap_file in tqdm(pcap_files, leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['train'][class_idx]['samples'] + 1)
                dataset['train'][class_idx]['payload'][sample_id] = token_seq
                dataset['train'][class_idx]['samples'] += 1
    if os.path.exists(test_dir):
        test_classes = [d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)]
        for class_dir in tqdm(test_classes):
            class_name = os.path.basename(class_dir)
            if class_name not in class_to_id:
                class_to_id[class_name] = len(class_to_id)
            class_idx = class_to_id[class_name]
            pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
            if len(pcap_files) == 0:
                pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
            for pcap_file in tqdm(pcap_files, leave=False):
                token_seq = read_MFR_token_sequence(pcap_file)
                if token_seq is not None and len(token_seq) > 0:
                    sample_id = str(dataset['test'][class_idx]['samples'] + 1)
                    dataset['test'][class_idx]['payload'][sample_id] = token_seq
                    dataset['test'][class_idx]['samples'] += 1
    output_data = {
        'metadata': {
            'dataset_name': dataset_name,
            'class_mapping': class_to_id,
            'num_classes': len(class_to_id)
        },
        'train': {str(k): dict(v) for k, v in dataset['train'].items()},
        'test': {str(k): dict(v) for k, v in dataset['test'].items()}
    }
    out_path = os.path.join(output_dir, f"{dataset_name}_processed.json")
    makedir(output_dir)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    return output_data


def process_multiple_datasets(datasets_root, dataset_names, output_dir):
    results = {}
    for dataset_name in dataset_names:
        try:
            result = process_presplit_dataset(datasets_root, dataset_name, output_dir)
            if result is not None:
                results[dataset_name] = result
        except Exception:
            continue
    summary = {'total_datasets': len(results), 'datasets': {}}
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
    return summary


if __name__ == '__main__':
    DATASETS_ROOT = "/path/to/single_flow_pcap_dataset/"
    DATASETS = [
        "NUDT_MobileTraffic Dataset_all",
        "CICIoT2023",
        "ISCXVPN2016",
        "USTC-TFC2016",
        "ISCXVPN2016-APP",
    ]
    OUTPUT_DIR = "/path/to/mfr_output/"
    VOCAB_PATH = "/path/to/models/mfr_vocab.txt"

    if not DATASETS:
        exit(1)

    save_vocab(VOCAB_PATH)
    makedir(OUTPUT_DIR)

    results = {}
    for dataset_name in DATASETS:
        dataset_path = os.path.join(DATASETS_ROOT, dataset_name)
        if not os.path.exists(dataset_path):
            continue
        try:
            dataset_output_dir = os.path.join(OUTPUT_DIR, f"{dataset_name}_MFR")
            makedir(dataset_output_dir)
            class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
            dataset_stats = {'total_packet_samples': 0, 'total_flow_samples': 0, 'classes': {}}
            for class_idx, class_dir in enumerate(class_dirs, 1):
                class_name = os.path.basename(class_dir)
                if class_name.startswith('.') or class_name.startswith('__'):
                    continue
                if not os.path.isdir(class_dir):
                    continue
                class_output_dir = os.path.join(dataset_output_dir, class_name)
                makedir(class_output_dir)
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
                    dataset_stats['classes'][class_name] = {
                        'packet_samples': result['packet_samples'],
                        'flow_samples': result['flow_samples']
                    }
                    dataset_stats['total_packet_samples'] += result['packet_samples']
                    dataset_stats['total_flow_samples'] += result['flow_samples']
            results[dataset_name] = dataset_stats
        except Exception:
            import traceback
            traceback.print_exc()

    summary_path = os.path.join(OUTPUT_DIR, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
