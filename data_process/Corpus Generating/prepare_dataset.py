import os
import glob
import pickle
import re
import argparse
import random
from multiprocessing import Pool

FLOW_ROWS = 8
FLOW_TOKENS_PER_ROW = 100
PACKETS_PER_FLOW = 5
TOKENS_PER_PACKET_ROW = FLOW_TOKENS_PER_ROW // PACKETS_PER_FLOW
HEADER_ROWS = 2
PAYLOAD_ROWS_FOR_MODEL = 2
PACKET_TOKEN_LENGTH = (HEADER_ROWS + PAYLOAD_ROWS_FOR_MODEL) * TOKENS_PER_PACKET_ROW


DATASETS_TO_PROCESS = []


def hex_str_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return 0


def extract_dataset_name(file_path):


    match = re.search(r'([A-Za-z0-9_]+_MFR)', file_path)
    if match:
        return match.group(1)


    return "Unknown"


def split_flow_matrix_to_packet80(flow_tokens_lines):
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


def read_flow_to_pretrain_sample(txt_file):
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


        flow_tokens_lines = []
        for line in lines[1:9]:
            line = line.strip()
            if line:

                tokens = line.split()
                flow_tokens_lines.append(tokens)

        packet_tokens_list = split_flow_matrix_to_packet80(flow_tokens_lines)
        if packet_tokens_list is None:
            return None


        flow_mfr_tokens = []
        for packet_tokens in packet_tokens_list:
            flow_mfr_tokens.extend(packet_tokens)

        if len(flow_mfr_tokens) != PACKET_TOKEN_LENGTH * PACKETS_PER_FLOW:
            return None

        return (flow_id, flow_mfr_tokens, dataset_name)

    except Exception as e:
        return None


def collect_all_flow_files(mfr_root, dataset_names=None):
    all_files = []
    split_stats = {'train': 0, 'test': 0, 'valid': 0, 'other': 0}
    dataset_stats = {}


    flow_mfr_dirs = glob.glob(
        os.path.join(mfr_root, '**', 'flow_mfr'),
        recursive=True
    )

    if dataset_names is not None:
        dataset_names_set = set(dataset_names)
    else:
        dataset_names_set = None

    for flow_dir in flow_mfr_dirs:

        dataset_name = extract_dataset_name(flow_dir)


        if dataset_names_set is not None and dataset_name not in dataset_names_set:
            continue


        if dataset_name not in dataset_stats:
            dataset_stats[dataset_name] = {'train': 0, 'test': 0, 'valid': 0, 'other': 0}


        for split in ['train', 'test', 'valid']:
            split_dir = os.path.join(flow_dir, split)
            if os.path.exists(split_dir):

                flow_files = glob.glob(
                    os.path.join(split_dir, 'flow_*.txt')
                )
                all_files.extend(flow_files)
                split_stats[split] += len(flow_files)
                dataset_stats[dataset_name][split] += len(flow_files)


        direct_files = glob.glob(
            os.path.join(flow_dir, 'flow_*.txt')
        )
        if direct_files:
            all_files.extend(direct_files)
            split_stats['other'] += len(direct_files)
            dataset_stats[dataset_name]['other'] += len(direct_files)


    return all_files


def process_files_to_dataset_pretrain(file_list, output_file, max_samples=None,
                                      processes_num=6, log_interval=1000, save_interval=100000,
                                      shuffle_seed=7):
    processed_flow_count = 0
    processed_sample_count = 0
    skipped_count = 0
    total_files = len(file_list)


    if max_samples is not None:
        file_list = file_list[:max_samples]
        total_files = len(file_list)

    flow_samples = []


    with Pool(processes=processes_num) as pool:
        file_count = 0

        for flow_sample in pool.imap(read_flow_to_pretrain_sample, file_list):
            file_count += 1


            if flow_sample is not None:

                flow_samples.append(flow_sample)
                processed_flow_count += 1
                processed_sample_count += 1
            else:
                skipped_count += 1


    rng = random.Random(shuffle_seed)
    rng.shuffle(flow_samples)

    with open(output_file, 'wb') as dataset_writer:
        for sample_idx, flow_sample in enumerate(flow_samples, 1):
            pickle.dump(flow_sample, dataset_writer)
            if save_interval > 0 and sample_idx % save_interval == 0:
                dataset_writer.flush()
        dataset_writer.flush()

    return processed_sample_count


def verify_dataset(dataset_file, num_samples=5):
    try:
        with open(dataset_file, 'rb') as f:
            samples = []
            try:
                while len(samples) < num_samples:
                    sample = pickle.load(f)
                    samples.append(sample)
            except EOFError:
                pass

    except Exception:
        return


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset_pretrain.pt from MFR dataset')
    parser.add_argument('--mfr_root', type=str,
                        default="",
                        help='Root directory of MFR dataset')
    parser.add_argument('--output_dir', type=str,
                        default="",
                        help='Output directory for dataset_pretrain.pt')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of dataset names to process (e.g.,  ). '
                             'If not specified, all datasets will be processed.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of flow samples (None for all)')
    parser.add_argument('--processes_num', type=int, default=6,
                        help='Number of processes for parallel processing (default: 6)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Number of files to process before logging progress (default: 1000)')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Number of samples to write before flushing progress (default: 100000)')
    parser.add_argument('--shuffle_seed', type=int, default=7,
                        help='Random seed used for global shuffle before writing .pt (default: 7)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the generated dataset after creation')

    args = parser.parse_args()


    if args.datasets is not None:
        datasets_to_process = args.datasets
    elif DATASETS_TO_PROCESS is not None and len(DATASETS_TO_PROCESS) > 0:
        datasets_to_process = DATASETS_TO_PROCESS
    else:
        datasets_to_process = None


    os.makedirs(args.output_dir, exist_ok=True)


    flow_files = collect_all_flow_files(args.mfr_root, dataset_names=datasets_to_process)

    if len(flow_files) == 0:
        return

    output_file = os.path.join(args.output_dir, 'dataset_pretrain_随机化.pt')
    processed_count = process_files_to_dataset_pretrain(
        flow_files,
        output_file,
        args.max_samples,
        args.processes_num,
        args.log_interval,
        args.save_interval,
        args.shuffle_seed
    )

    if args.verify:
        verify_dataset(output_file, num_samples=5)


if __name__ == '__main__':
    main()
