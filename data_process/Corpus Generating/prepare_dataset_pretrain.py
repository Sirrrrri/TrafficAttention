#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Convert flow_mfr data from MFR datasets into dataset_pretrain.pt.

Each flow MFR matrix (8 rows x 100 tokens = 800 tokens) is split into
5 packet token sequences (160 tokens each).

Output sample format: (flow_id, packet_index, packet_tokens, dataset_name)
  - flow_id: 8-digit zero-padded string, e.g. "00000123"
  - packet_index: int (1-5), packet position within the flow
  - packet_tokens: list of 160 4-hex-char strings (2 bytes each), e.g. "4500", "0020"
  - dataset_name: name of the source dataset
"""

import os
import glob
import pickle
import re
import argparse
from multiprocessing import Pool

# ============================================================================
# Dataset configuration
# ============================================================================
# List of dataset names to process.
# Set to None or [] to process all datasets found under mfr_root.
# Example: DATASETS_TO_PROCESS = ['NonVPN_MFR', 'browser_MFR']
DATASETS_TO_PROCESS = None
# ============================================================================


def hex_str_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return 0


def extract_dataset_name(file_path):
    """Extract dataset name (ending with _MFR) from a file path."""
    match = re.search(r'([A-Za-z0-9_]+_MFR)', file_path)
    return match.group(1) if match else "Unknown"


def read_and_split_flow_to_packets(txt_file):
    """
    Read a single flow MFR matrix .txt file and split it into 5 packet samples.

    File format:
      - Line 1: label (skipped)
      - Lines 2-9: 8 rows x 100 tokens (5 packets laid out horizontally)
        Packet k occupies columns [(k-1)*20 : k*20] across all 8 rows (160 tokens).

    Returns list of (flow_id, packet_index, packet_tokens, dataset_name) tuples,
    or None if the file is invalid.
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 9:
            return None

        flow_filename = os.path.basename(txt_file)
        if flow_filename.startswith('flow_') and flow_filename.endswith('.txt'):
            flow_id = flow_filename[5:-4]
        else:
            match = re.search(r'flow_(\d+)', txt_file)
            if not match:
                return None
            flow_id = match.group(1)

        if not flow_id.isdigit():
            return None
        flow_id = flow_id.zfill(8)

        dataset_name = extract_dataset_name(txt_file)

        flow_tokens_lines = []
        for line in lines[1:9]:
            tokens = line.strip().split()
            if tokens:
                flow_tokens_lines.append(tokens)

        if len(flow_tokens_lines) != 8:
            return None
        if any(len(row) != 100 for row in flow_tokens_lines):
            return None

        packet_samples = []
        for packet_idx in range(1, 6):
            col_start = (packet_idx - 1) * 20
            col_end = packet_idx * 20
            hex_tokens = []
            for row in flow_tokens_lines:
                hex_tokens.extend(row[col_start:col_end])
            if len(hex_tokens) != 160:
                continue
            packet_samples.append((flow_id, packet_idx, hex_tokens, dataset_name))

        return packet_samples if len(packet_samples) == 5 else None

    except Exception:
        return None


def collect_all_flow_files(mfr_root, dataset_names=None):
    """
    Collect all flow_*.txt files under mfr_root.

    Supports two directory layouts:
      Layout 1 (no category layer):
        <mfr_root>/<DatasetName_MFR>/flow_mfr/{train,test,valid}/flow_*.txt
      Layout 2 (with category layer):
        <mfr_root>/<DatasetName_MFR>/<Category>/flow_mfr/{train,test,valid}/flow_*.txt

    Args:
        mfr_root: root directory of MFR datasets
        dataset_names: list of dataset names to include, or None for all

    Returns:
        list of file paths
    """
    all_files = []
    split_stats = {'train': 0, 'test': 0, 'valid': 0, 'other': 0}
    dataset_stats = {}

    flow_mfr_dirs = glob.glob(os.path.join(mfr_root, '**', 'flow_mfr'), recursive=True)
    print(f"Found {len(flow_mfr_dirs)} flow_mfr directories")

    dataset_names_set = set(dataset_names) if dataset_names else None
    if dataset_names_set:
        print(f"Datasets to process: {dataset_names}")
    else:
        print("Processing all datasets")

    for flow_dir in flow_mfr_dirs:
        dataset_name = extract_dataset_name(flow_dir)
        if dataset_names_set and dataset_name not in dataset_names_set:
            continue

        if dataset_name not in dataset_stats:
            dataset_stats[dataset_name] = {'train': 0, 'test': 0, 'valid': 0, 'other': 0}

        for split in ['train', 'test', 'valid']:
            split_dir = os.path.join(flow_dir, split)
            if os.path.exists(split_dir):
                files = glob.glob(os.path.join(split_dir, 'flow_*.txt'))
                all_files.extend(files)
                split_stats[split] += len(files)
                dataset_stats[dataset_name][split] += len(files)

        direct_files = glob.glob(os.path.join(flow_dir, 'flow_*.txt'))
        if direct_files:
            all_files.extend(direct_files)
            split_stats['other'] += len(direct_files)
            dataset_stats[dataset_name]['other'] += len(direct_files)

    print(f"\nFile collection summary:")
    for split in ['train', 'test', 'valid']:
        print(f"  {split}: {split_stats[split]} files")
    if split_stats['other'] > 0:
        print(f"  other: {split_stats['other']} files (not under train/test/valid)")
    print(f"\nTotal flow matrix files collected: {len(all_files)}")

    if dataset_stats:
        print(f"\nPer-dataset breakdown:")
        for ds in sorted(dataset_stats):
            s = dataset_stats[ds]
            total = sum(s.values())
            print(f"  {ds}: train={s['train']}, test={s['test']}, valid={s['valid']}"
                  + (f", other={s['other']}" if s['other'] else "")
                  + f", total={total}")

    return all_files


def process_files_to_dataset_pretrain(file_list, output_file, max_samples=None,
                                      processes_num=6, log_interval=1000, save_interval=100000):
    """
    Process a list of flow .txt files and write packet samples to output_file.

    Args:
        file_list: list of .txt file paths
        output_file: path for the output .pt file
        max_samples: cap on number of flow files to process (None = all)
        processes_num: worker processes for parallel processing
        log_interval: print progress every N files
        save_interval: flush buffer to disk every N files
    """
    if max_samples is not None:
        file_list = file_list[:max_samples]

    total_files = len(file_list)
    processed_flow_count = 0
    processed_packet_count = 0
    skipped_count = 0

    print(f"\n{'='*80}")
    print(f"Processing dataset_pretrain.pt")
    print(f"{'='*80}")
    print(f"Total flow files to process: {total_files}")
    print(f"Expected packet samples: {total_files * 5} (5 packets per flow)")
    print(f"Using {processes_num} processes for parallel processing...")
    print(f"Output file: {output_file}")

    dataset_writer = open(output_file, 'wb')
    packet_buffer = []

    with Pool(processes=processes_num) as pool:
        file_count = 0
        for packet_samples in pool.imap(read_and_split_flow_to_packets, file_list):
            file_count += 1

            if packet_samples:
                packet_buffer.extend(packet_samples)
                processed_flow_count += 1
                processed_packet_count += len(packet_samples)
            else:
                skipped_count += 1

            if file_count % log_interval == 0:
                print(f"[Progress] {file_count}/{total_files} flows | "
                      f"processed={processed_flow_count}, packets={processed_packet_count}, "
                      f"skipped={skipped_count}")

            if file_count % save_interval == 0 and packet_buffer:
                print(f"[Save] Writing {len(packet_buffer)} packet samples to disk...")
                for sample in packet_buffer:
                    pickle.dump(sample, dataset_writer)
                dataset_writer.flush()
                print(f"[Save] Done | total flows processed: {file_count}")
                packet_buffer.clear()

    if packet_buffer:
        print(f"\n[Final Save] Writing remaining {len(packet_buffer)} packet samples...")
        for sample in packet_buffer:
            pickle.dump(sample, dataset_writer)
        dataset_writer.flush()
        packet_buffer.clear()
        print(f"[Final Save] Completed")

    dataset_writer.close()

    print(f"\n{'='*80}")
    print(f"[Final Summary]")
    print(f"{'='*80}")
    print(f"  Total flow files processed : {file_count}")
    print(f"  Successfully processed     : {processed_flow_count}")
    print(f"  Total packet samples       : {processed_packet_count}")
    print(f"  Skipped                    : {skipped_count}")
    print(f"  Dataset saved to           : {output_file}")
    print(f"{'='*80}\n")

    return processed_packet_count


def verify_dataset(dataset_file, num_samples=5):
    """Load and print the first num_samples entries from a dataset file."""
    print(f"\n{'='*80}")
    print(f"Verifying dataset: {dataset_file}")
    print(f"{'='*80}\n")

    try:
        with open(dataset_file, 'rb') as f:
            samples = []
            try:
                while len(samples) < num_samples:
                    samples.append(pickle.load(f))
            except EOFError:
                pass

        print(f"Loaded {len(samples)} samples for verification\n")

        for i, sample in enumerate(samples):
            if len(sample) == 4:
                flow_id, packet_idx, packet_tokens, dataset_name = sample
                print(f"Sample {i+1}:")
                print(f"  flow_id      : {flow_id} ({type(flow_id).__name__})")
                print(f"  packet_index : {packet_idx} ({type(packet_idx).__name__})")
                print(f"  packet_tokens: {len(packet_tokens)} items (4-hex-char strings)")
                print(f"  dataset_name : {dataset_name}")
                if packet_tokens:
                    print(f"  first 10     : {packet_tokens[:10]}")
                    print(f"  last  10     : {packet_tokens[-10:]}")
            elif len(sample) == 3:
                flow_id, packet_idx, packet_tokens = sample
                print(f"Sample {i+1} (legacy 3-tuple):")
                print(f"  flow_id      : {flow_id}")
                print(f"  packet_index : {packet_idx}")
                print(f"  packet_tokens: {len(packet_tokens)} items")
                if packet_tokens:
                    print(f"  first 10     : {packet_tokens[:10]}")
            else:
                print(f"Sample {i+1}: unexpected format (len={len(sample)})")
            print()

        print(f"{'='*80}")
        print("Verification complete.")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"Verification error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset_pretrain.pt from MFR dataset')
    parser.add_argument('--mfr_root', type=str, required=True,
                        help='Root directory of MFR dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for dataset_pretrain.pt')
    parser.add_argument('--output_name', type=str, default='dataset_pretrain.pt',
                        help='Output filename (default: dataset_pretrain.pt)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to process (e.g. NonVPN_MFR NonTor_MFR). '
                             'Defaults to DATASETS_TO_PROCESS in script, or all if unset.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of flow files to process (default: all)')
    parser.add_argument('--processes_num', type=int, default=6,
                        help='Worker processes for parallel processing (default: 6)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Log progress every N files (default: 1000)')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Flush buffer every N files (default: 100000)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the generated dataset after creation')

    args = parser.parse_args()

    # Priority: CLI --datasets > DATASETS_TO_PROCESS constant > all datasets
    if args.datasets is not None:
        datasets_to_process = args.datasets
        print(f"Using datasets from CLI: {datasets_to_process}")
    elif DATASETS_TO_PROCESS:
        datasets_to_process = DATASETS_TO_PROCESS
        print(f"Using datasets from script config: {datasets_to_process}")
    else:
        datasets_to_process = None
        print("Processing all datasets (no filter specified)")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Step 1: Collecting flow files from MFR dataset...")
    print("=" * 80)
    flow_files = collect_all_flow_files(args.mfr_root, dataset_names=datasets_to_process)

    if not flow_files:
        print("Error: no flow files found. Check --mfr_root and dataset names.")
        return

    print("\n" + "=" * 80)
    print("Step 2: Processing and generating dataset_pretrain.pt...")
    print("=" * 80)

    output_file = os.path.join(args.output_dir, args.output_name)
    processed_count = process_files_to_dataset_pretrain(
        flow_files,
        output_file,
        args.max_samples,
        args.processes_num,
        args.log_interval,
        args.save_interval,
    )

    if args.verify:
        print("\n" + "=" * 80)
        print("Step 3: Verifying generated dataset...")
        print("=" * 80)
        verify_dataset(output_file, num_samples=5)

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"\nOutput file   : {output_file}")
    print(f"Samples saved : {processed_count}")
    print(f"\nSample format : (flow_id, packet_index, packet_tokens, dataset_name)")
    print(f"  flow_id      : 8-digit zero-padded string (e.g. '00000123')")
    print(f"  packet_index : int 1-5")
    print(f"  packet_tokens: list of 160 4-hex-char strings (e.g. ['4500', '0020', ...])")
    print(f"  dataset_name : source dataset name (e.g. 'NonVPN_MFR')")
    print(f"\nTo use in training: --dataset_path {output_file}")
    print()


if __name__ == '__main__':
    main()
