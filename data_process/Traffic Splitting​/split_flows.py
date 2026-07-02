#!/usr/bin/python3
import os
import glob
import subprocess
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import random
import gc
import tempfile
import socket
import dpkt


def check_splitcap_installed():
    if shutil.which('mono') is None:
        return False

    splitcap_paths = [
        '/usr/local/bin/SplitCap.exe',
        '/usr/bin/SplitCap.exe',
        './SplitCap.exe',
        os.path.join(os.path.dirname(__file__), 'SplitCap.exe')
    ]

    for path in splitcap_paths:
        if os.path.exists(path):
            return path

    return False


SPLITCAP_EXE_PATH = None


def random_ipv4():
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def modify_packet_binary(packet_data, has_ether, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
                         random_src_ip_bytes, random_dst_ip_bytes):
    try:
        if len(packet_data) < ip_start + 20:
            return None

        ip_header = packet_data[ip_start:ip_start + 20]
        if (ip_header[0] >> 4) & 0x0F != 4:
            return None

        ip_header_len = (ip_header[0] & 0x0F) * 4
        if len(packet_data) < ip_start + ip_header_len:
            return None

        is_forward = (ip_header[12:16] == first_src_ip_bytes)
        modified_data = bytearray(packet_data)

        if is_forward:
            modified_data[ip_start + 12:ip_start + 16] = random_src_ip_bytes
            modified_data[ip_start + 16:ip_start + 20] = random_dst_ip_bytes
        else:
            modified_data[ip_start + 12:ip_start + 16] = random_dst_ip_bytes
            modified_data[ip_start + 16:ip_start + 20] = random_src_ip_bytes

        protocol = ip_header[9]
        if protocol == 6 or protocol == 17:
            port_start = ip_start + ip_header_len
            if len(modified_data) >= port_start + 4:
                modified_data[port_start:port_start + 4] = b'\x00\x00\x00\x00'

        return bytes(modified_data)
    except Exception:
        return None


def anonymize_flow_pcap(pcap_path):
    temp_path = None
    try:
        with open(pcap_path, 'rb') as f_in:
            try:
                pcap_reader = dpkt.pcap.Reader(f_in)
            except Exception:
                return False

            has_ether = True
            ip_start = 0

            try:
                first_timestamp, first_packet_data = next(pcap_reader)
            except StopIteration:
                return False

            try:
                try:
                    eth = dpkt.ethernet.Ethernet(first_packet_data)
                    if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                        ip = eth.ip
                        has_ether = True
                        ip_start = 14
                    else:
                        has_ether = False
                        ip = dpkt.ip.IP(first_packet_data)
                        ip_start = 0
                except Exception:
                    has_ether = False
                    ip = dpkt.ip.IP(first_packet_data)
                    ip_start = 0

                if ip.v != 4:
                    return False

                first_src_ip_bytes = ip.src
                first_dst_ip_bytes = ip.dst
            except Exception:
                return False

            if first_src_ip_bytes is None or first_dst_ip_bytes is None:
                return False

            random_src_ip_bytes = socket.inet_aton(random_ipv4())
            random_dst_ip_bytes = socket.inet_aton(random_ipv4())

            temp_fd, temp_path = tempfile.mkstemp(suffix='.pcap', dir=os.path.dirname(pcap_path))
            temp_file = os.fdopen(temp_fd, 'wb')
            pcap_writer = dpkt.pcap.Writer(temp_file)

            modified_data = modify_packet_binary(
                first_packet_data, has_ether, ip_start,
                first_src_ip_bytes, first_dst_ip_bytes,
                random_src_ip_bytes, random_dst_ip_bytes
            )
            packet_count = 0
            if modified_data:
                pcap_writer.writepkt(modified_data, first_timestamp)
                packet_count = 1
            del first_packet_data

            for timestamp, packet_data in pcap_reader:
                try:
                    modified_data = modify_packet_binary(
                        packet_data, has_ether, ip_start,
                        first_src_ip_bytes, first_dst_ip_bytes,
                        random_src_ip_bytes, random_dst_ip_bytes
                    )
                    if modified_data:
                        pcap_writer.writepkt(modified_data, timestamp)
                        packet_count += 1
                    del modified_data
                except Exception:
                    continue
                finally:
                    del packet_data

            pcap_writer.close()
            temp_file.close()

        if packet_count == 0:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return False

        try:
            os.replace(temp_path, pcap_path)
            return True
        except Exception:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return False

    except Exception:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False


def split_pcap_with_splitcap_wrapper(args):
    pcap_path, output_dir, splitcap_exe = args
    return split_pcap_with_splitcap(pcap_path, output_dir, splitcap_exe)


def split_pcap_with_splitcap(pcap_path, output_dir, splitcap_exe):
    temp_output = None
    debug_log = os.path.join(output_dir, 'splitcap_debug.log')

    try:
        base_name = os.path.splitext(os.path.basename(pcap_path))[0]
        temp_output = os.path.join(output_dir, f'temp_{base_name}')
        os.makedirs(temp_output, exist_ok=True)

        shutil.copy2(pcap_path, os.path.join(temp_output, os.path.basename(pcap_path)))
        shutil.copy2(splitcap_exe, os.path.join(temp_output, 'SplitCap.exe'))

        cmd = ['mono', 'SplitCap.exe', '-r', os.path.basename(pcap_path), '-s', 'session']

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            cwd=temp_output
        )

        with open(debug_log, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"File: {pcap_path}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working dir: {temp_output}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
            f.write(f"Files in temp dir: {os.listdir(temp_output)}\n")

        if result.returncode != 0:
            shutil.rmtree(temp_output, ignore_errors=True)
            return 0

        flow_files = []
        for fname in os.listdir(temp_output):
            full_path = os.path.join(temp_output, fname)
            if fname == os.path.basename(pcap_path) or fname == 'SplitCap.exe':
                continue
            if os.path.isdir(full_path):
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if os.path.isfile(item_path):
                        flow_files.append(item_path)
            elif os.path.isfile(full_path):
                flow_files.append(full_path)

        flows_count = len(flow_files)

        with open(debug_log, 'a') as f:
            f.write(f"Flow files found: {flows_count}\n")
            if flows_count > 0:
                f.write(f"Flow file list: {[os.path.basename(ff) for ff in flow_files[:5]]}\n")

        if flows_count == 0:
            shutil.rmtree(temp_output, ignore_errors=True)
            return 0

        for idx, flow_file in enumerate(flow_files, 1):
            flow_basename = os.path.basename(flow_file)
            new_filename = f"{base_name}_flow{idx:04d}_{flow_basename}"
            if not new_filename.endswith('.pcap'):
                new_filename += '.pcap'
            new_path = os.path.join(output_dir, new_filename)
            shutil.move(flow_file, new_path)
            anonymize_flow_pcap(new_path)
            if idx % 10 == 0:
                gc.collect()

        del flow_files
        shutil.rmtree(temp_output, ignore_errors=True)
        gc.collect()
        return flows_count

    except subprocess.TimeoutExpired:
        with open(debug_log, 'a') as f:
            f.write(f"\nTIMEOUT: {pcap_path}\n")
        if temp_output and os.path.exists(temp_output):
            shutil.rmtree(temp_output, ignore_errors=True)
        return 0
    except Exception as e:
        with open(debug_log, 'a') as f:
            f.write(f"\nERROR: {pcap_path}: {str(e)}\n")
        if temp_output and os.path.exists(temp_output):
            shutil.rmtree(temp_output, ignore_errors=True)
        return 0


def process_dataset_directory(dataset_path, output_root, num_workers=None):
    dataset_name = os.path.basename(dataset_path)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)

    class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
    if not class_dirs:
        return

    total_pcap_files = 0
    total_flows = 0
    start_time = time.time()

    for class_dir in tqdm(class_dirs, desc=dataset_name):
        class_name = os.path.basename(class_dir)
        if class_name.startswith('.') or class_name.startswith('__'):
            continue

        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))
        if not pcap_files:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        if not pcap_files:
            continue

        output_class_dir = os.path.join(output_root, dataset_name, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        task_args = [(pcap_file, output_class_dir, SPLITCAP_EXE_PATH) for pcap_file in pcap_files]
        num_pcap_files = len(pcap_files)

        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(split_pcap_with_splitcap_wrapper, task_args, chunksize=1),
                total=len(task_args),
                desc=class_name,
                leave=False
            ))
            class_flows = sum(results)
            del results
            gc.collect()

        del task_args, pcap_files
        total_pcap_files += num_pcap_files
        total_flows += class_flows
        gc.collect()


def process_all_datasets(datasets_root, output_root, dataset_names, num_workers=None):
    global SPLITCAP_EXE_PATH

    SPLITCAP_EXE_PATH = check_splitcap_installed()
    if not SPLITCAP_EXE_PATH:
        return

    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)

    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        if not os.path.exists(dataset_path):
            continue
        try:
            process_dataset_directory(
                dataset_path=dataset_path,
                output_root=output_root,
                num_workers=num_workers
            )
        except Exception:
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    try:
        from config import DATASETS_ROOT, DATASETS
    except ImportError:
        DATASETS_ROOT = "/path/to/original_pcap_datasets"
        DATASETS = ["Dataset1", "Dataset2"]

    OUTPUT_ROOT = "/path/to/output_split_flows"
    NUM_WORKERS = 6

    process_all_datasets(
        datasets_root=DATASETS_ROOT,
        output_root=OUTPUT_ROOT,
        dataset_names=DATASETS,
        num_workers=NUM_WORKERS
    )
