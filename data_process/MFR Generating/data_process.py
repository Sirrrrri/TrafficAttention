


import os
import glob
import json
import random
import numpy as np
import dpkt
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager


def makedir(path):
    
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        pass


def get_ip_offset(datalink):
    
    offset_map = {
        1: 14,
        101: 0,
        12: 0,
        113: 16,
        276: 20,
    }
    return offset_map.get(datalink, None)


def open_pcap_reader(file_handle):
    
    try:
        file_handle.seek(0)
        reader = dpkt.pcap.Reader(file_handle)
        return reader, reader.datalink()
    except Exception:
        pass

    try:
        file_handle.seek(0)
        reader = dpkt.pcapng.Reader(file_handle)
        return reader, reader.datalink()
    except Exception:
        pass

    return None, None


def normalize_mfr_fields(header_bytes, payload_bytes):
    
    header = header_bytes.hex()
    payload = payload_bytes.hex()

    if len(header) > 160:
        header = header[:160]
    elif len(header) < 160:
        header += '0' * (160 - len(header))

    if len(payload) > 480:
        payload = payload[:480]
    elif len(payload) < 480:
        payload += '0' * (480 - len(payload))

    return (header, payload)


def extract_packet_mfr_from_ip_bytes(ip_bytes):
    
    try:
        if len(ip_bytes) < 20:
            return None

        version = (ip_bytes[0] >> 4) & 0x0F
        ihl = (ip_bytes[0] & 0x0F) * 4
        if version != 4 or ihl < 20 or len(ip_bytes) < ihl:
            return None

        total_len = int.from_bytes(ip_bytes[2:4], byteorder='big')
        if total_len < ihl:
            return None
        if total_len > len(ip_bytes):
            total_len = len(ip_bytes)

        ip_packet = ip_bytes[:total_len]
        protocol = ip_packet[9]
        frag_info = int.from_bytes(ip_packet[6:8], byteorder='big')
        frag_offset = frag_info & 0x1FFF
        payload_start = total_len

        if frag_offset == 0 and protocol == 6 and len(ip_packet) >= ihl + 20:
            tcp_header_len = ((ip_packet[ihl + 12] >> 4) & 0x0F) * 4
            if tcp_header_len >= 20 and len(ip_packet) >= ihl + tcp_header_len:
                payload_start = ihl + tcp_header_len
        elif frag_offset == 0 and protocol == 17 and len(ip_packet) >= ihl + 8:
            payload_start = ihl + 8

        header_bytes = ip_packet[:payload_start]
        payload_bytes = ip_packet[payload_start:]
        return normalize_mfr_fields(header_bytes, payload_bytes)
    except Exception:
        return None


def extract_packet_mfr(packet_data, datalink):
    
    ip_offset = get_ip_offset(datalink)
    if ip_offset is None or len(packet_data) < ip_offset + 20:
        return None
    return extract_packet_mfr_from_ip_bytes(packet_data[ip_offset:])


def read_pcap_mfr_data(pcap_path, max_packets=None, max_valid=None):
    
    try:
        with open(pcap_path, 'rb') as f:
            pcap_reader, datalink = open_pcap_reader(f)
            if pcap_reader is None or get_ip_offset(datalink) is None:
                return None

            data = []
            for packet_idx, (_, packet_data) in enumerate(pcap_reader):
                if max_packets is not None and packet_idx >= max_packets:
                    break

                mfr_data = extract_packet_mfr(packet_data, datalink)
                if mfr_data is None:
                    continue

                data.append(mfr_data)
                if max_valid is not None and len(data) >= max_valid:
                    break

            return data
    except Exception:
        return None



def make_flow_matrix(packet_data):
    
    packet_data = list(packet_data[:5])
    if len(packet_data) < 5:
        for _ in range(5 - len(packet_data)):
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

    return flow_matrix_rows




def read_flow_matrix(pcap_path):
    
    packet_data = read_pcap_mfr_data(pcap_path, max_valid=5)
    if not packet_data:
        return None
    return make_flow_matrix(packet_data)


def read_MFR_bytes(pcap_path):
    
    data = read_pcap_mfr_data(pcap_path, max_valid=5)
    if not data:
        return None

    if len(data) < 5:
        for _ in range(5 - len(data)):
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
    
    
    
    token_sequence = [int(mfr_hex[i:i+4], 16) for i in range(0, len(mfr_hex), 4)]
    
    return token_sequence


def read_MFR_token_sequence(pcap_path):
    
    mfr_hex = read_MFR_bytes(pcap_path)
    if mfr_hex is None:
        return None
    
    
    
    token_sequence = [mfr_hex[i:i+4] for i in range(0, len(mfr_hex), 4)]
    
    
    return ' '.join(token_sequence)


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
    
    
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Processing classes")):
        class_name = os.path.basename(class_dir)
        class_to_id[class_name] = class_idx
        
        
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        
        
        
        random.shuffle(pcap_files)
        
        
        n_total = len(pcap_files)
        n_train = int(n_total * train_ratio)
        n_test = int(n_total * test_ratio)
        n_valid = n_total - n_train - n_test
        
        train_files = pcap_files[:n_train]
        test_files = pcap_files[n_train:n_train+n_test]
        valid_files = pcap_files[n_train+n_test:]
        
        
        for pcap_file in tqdm(train_files, desc=f"Processing {class_name} train", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['train'][class_idx]['samples'] + 1)
                dataset['train'][class_idx]['payload'][sample_id] = token_seq
                dataset['train'][class_idx]['samples'] += 1
        
        
        for pcap_file in tqdm(test_files, desc=f"Processing {class_name} test", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['test'][class_idx]['samples'] + 1)
                dataset['test'][class_idx]['payload'][sample_id] = token_seq
                dataset['test'][class_idx]['samples'] += 1
        
        
        for pcap_file in tqdm(valid_files, desc=f"Processing {class_name} valid", leave=False):
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
        mfr_packets = read_pcap_mfr_data(pcap_file, max_packets=1000)
        if not mfr_packets:
            return None

        return make_flow_matrix(mfr_packets[:5])

    except MemoryError:
        return None
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return None


def process_class_mfr_matrices(class_dir, output_dir, dataset_name, class_name, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1, num_workers=6, batch_size=10000, log_interval=100):
    

    
    pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))

    if len(pcap_files) == 0:
        return None

    
    random.shuffle(pcap_files)

    
    n_total = len(pcap_files)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_valid = n_total - n_train - n_test

    train_files = pcap_files[:n_train]
    test_files = pcap_files[n_train:n_train+n_test]
    valid_files = pcap_files[n_train+n_test:]

    
    makedir(output_dir)

    
    flow_base_dir = os.path.join(output_dir, 'flow_mfr')

    for split in ['train', 'test', 'valid']:
        makedir(os.path.join(flow_base_dir, split))

    
    sample_counts = {
        'train': {'flow': 0},
        'test': {'flow': 0},
        'valid': {'flow': 0}
    }

    def save_flow_txt(matrix, split_name, flow_id):
        
        filename = f"flow_{flow_id:08d}.txt"
        filepath = os.path.join(flow_base_dir, split_name, filename)
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
                pcap_file = futures[future]
                try:
                    flow_matrix = future.result(timeout=60)

                    if flow_matrix is not None:
                        sample_counts[split_name]['flow'] += 1
                        save_flow_txt(flow_matrix, split_name, sample_counts[split_name]['flow'])
                        flow_success_count += 1
                    else:
                        flow_fail_count += 1

                    processed_count += 1

                    
                    if processed_count % log_interval == 0:
                        pass

                except TimeoutError:
                    flow_fail_count += 1
                except Exception as e:
                    flow_fail_count += 1

        
        total_files = len(files)
        success_rate = (flow_success_count / total_files * 100) if total_files > 0 else 0

    
    process_files_parallel(train_files, 'train')

    
    process_files_parallel(test_files, 'test')

    
    process_files_parallel(valid_files, 'valid')

    

    return {
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
    n_valid = n_total - n_train - n_test
    
    train_files = pcap_files[:n_train]
    test_files = pcap_files[n_train:n_train+n_test]
    valid_files = pcap_files[n_train+n_test:]
    
    dataset = {
        'train': {'samples': 0, 'payload': {}},
        'test': {'samples': 0, 'payload': {}},
        'valid': {'samples': 0, 'payload': {}}
    }
    
    
    for pcap_file in tqdm(train_files, desc=f"  Train", leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['train']['samples'] + 1)
            dataset['train']['payload'][sample_id] = token_seq
            dataset['train']['samples'] += 1
    
    
    for pcap_file in tqdm(test_files, desc=f"  Test", leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['test']['samples'] + 1)
            dataset['test']['payload'][sample_id] = token_seq
            dataset['test']['samples'] += 1
    
    
    for pcap_file in tqdm(valid_files, desc=f"  Valid", leave=False):
        token_seq = read_MFR_token_sequence(pcap_file)
        if token_seq is not None and len(token_seq) > 0:
            sample_id = str(dataset['valid']['samples'] + 1)
            dataset['valid']['payload'][sample_id] = token_seq
            dataset['valid']['samples'] += 1
    
    
    output_data = {
        'metadata': {
            'class_name': class_name,
            'num_classes': 1
        },
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
    
    for class_idx, class_dir in enumerate(tqdm(train_classes, desc="处理训练集类别")):
        class_name = os.path.basename(class_dir)
        class_to_id[class_name] = class_idx
        
        
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        
        
        
        for pcap_file in tqdm(pcap_files, desc=f"  {class_name}", leave=False):
            token_seq = read_MFR_token_sequence(pcap_file)
            if token_seq is not None and len(token_seq) > 0:
                sample_id = str(dataset['train'][class_idx]['samples'] + 1)
                dataset['train'][class_idx]['payload'][sample_id] = token_seq
                dataset['train'][class_idx]['samples'] += 1
    
    
    if os.path.exists(test_dir):
        test_classes = [d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)]
        
        for class_dir in tqdm(test_classes, desc="处理测试集类别"):
            class_name = os.path.basename(class_dir)
            
            
            if class_name not in class_to_id:
                class_to_id[class_name] = len(class_to_id)
            
            class_idx = class_to_id[class_name]
            
            
            pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
            if len(pcap_files) == 0:
                pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
            
            
            
            for pcap_file in tqdm(pcap_files, desc=f"  {class_name}", leave=False):
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
    
    
    output_path = os.path.join(output_dir, f"{dataset_name}_processed.json")
    makedir(output_dir)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    
    train_total = sum([v['samples'] for v in dataset['train'].values()])
    test_total = sum([v['samples'] for v in dataset['test'].values()])
    
    
    return output_data


def process_multiple_datasets(datasets_root, dataset_names, output_dir):
    
    
    results = {}
    
    for dataset_name in dataset_names:
        try:
            result = process_presplit_dataset(datasets_root, dataset_name, output_dir)
            if result is not None:
                results[dataset_name] = result
        except Exception as e:
            continue
    
    
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
    
    return summary


if __name__ == '__main__':
    
    
    
    
    
    try:
        from config import DATASETS_ROOT, DATASETS, PROCESSING_MODE
    except ImportError:
        
        DATASETS_ROOT = ""
        
        
        DATASETS = []
        PROCESSING_MODE = None  
    
    
    
    
    
    if PROCESSING_MODE is None:
        PROCESSING_MODE = 'auto'  
    
    
    valid_modes = ['mode1', 'mode2', 'auto']
    if PROCESSING_MODE not in valid_modes:
        exit(1)
    
    
    OUTPUT_DIR = ""
    
    
    VOCAB_PATH = ""
    
    
    if not DATASETS or len(DATASETS) == 0:
        exit(1)
    
    
    
    
    save_vocab(VOCAB_PATH)
    
    
    
    
    makedir(OUTPUT_DIR)
    
    results = {}
    for dataset_name in DATASETS:
        dataset_path = os.path.join(DATASETS_ROOT, dataset_name)
        
        if not os.path.exists(dataset_path):
            continue
        
        
        try:
            
            class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
            
            class_dirs = [d for d in class_dirs 
                         if not os.path.basename(d).startswith('.') 
                         and not os.path.basename(d).startswith('__')]
            
            
            direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcap'))
            if len(direct_pcap_files) == 0:
                direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcapng'))
            
            
            is_mode2 = False
            if PROCESSING_MODE == 'mode2':
                
                if len(direct_pcap_files) == 0:
                    continue
                class_dirs = [dataset_path]  
                is_mode2 = True
            elif PROCESSING_MODE == 'mode1':
                
                if len(class_dirs) == 0:
                    continue
                is_mode2 = False
            else:
                
                if len(class_dirs) == 0:
                    
                    if len(direct_pcap_files) == 0:
                        continue
                    class_dirs = [dataset_path]  
                    is_mode2 = True
                else:
                    is_mode2 = False
            
            
            dataset_output_dir = os.path.join(OUTPUT_DIR, f"{dataset_name}_MFR")
            makedir(dataset_output_dir)
            
            dataset_stats = {
                'total_flow_samples': 0, 
                'classes': {}
            }
            
            
            for class_idx, class_dir in enumerate(class_dirs, 1):
                if is_mode2:
                    
                    class_name = dataset_name
                else:
                    
                    class_name = os.path.basename(class_dir)
                
                
                if class_name.startswith('.') or class_name.startswith('__'):
                    continue
                
                
                if not os.path.isdir(class_dir):
                    continue
                
                
                
                
                if is_mode2:
                    class_output_dir = dataset_output_dir
                else:
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
                    flow_total = result['flow_samples']
                    
                    dataset_stats['classes'][class_name] = {
                        'flow_samples': flow_total
                    }
                    dataset_stats['total_flow_samples'] += flow_total
                    
            
            results[dataset_name] = dataset_stats
            
        except Exception:
            pass
    summary_path = os.path.join(OUTPUT_DIR, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    
    if results:
        pass

