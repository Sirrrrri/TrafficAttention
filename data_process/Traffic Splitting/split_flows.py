import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import random
import gc
import tempfile
import socket
import dpkt


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
        datalink = reader.datalink()
        return reader, datalink
    except (ValueError, Exception):
        pass


    try:
        file_handle.seek(0)
        reader = dpkt.pcapng.Reader(file_handle)
        datalink = reader.datalink()
        return reader, datalink
    except (ValueError, AttributeError, Exception):
        pass

    return None, None


def extract_ip_from_packet(packet_data, datalink):
    ip_offset = get_ip_offset(datalink)
    if ip_offset is None:
        return None, None

    if len(packet_data) < ip_offset + 20:
        return None, None

    try:
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(packet_data)
            if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                return eth.ip, ip_offset
            elif eth.type == dpkt.ethernet.ETH_TYPE_IP6:
                return None, None
            else:
                return None, None
        else:

            ip = dpkt.ip.IP(packet_data[ip_offset:])
            if ip.v != 4:
                return None, None
            return ip, ip_offset
    except Exception:
        return None, None


def check_dpkt_available():
    try:
        import dpkt
        return True
    except ImportError:
        return False


def random_ipv4():
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def anonymize_flow_pcap(pcap_path, debug_log=None):
    temp_path = None

    try:

        with open(pcap_path, 'rb') as f_in:

            pcap_reader, datalink = open_pcap_reader(f_in)
            if pcap_reader is None:
                error_msg = "Failed to read pcap/pcapng header"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False


            ip_start = get_ip_offset(datalink)
            if ip_start is None:
                error_msg = f"Unsupported datalink type: {datalink}"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False


            first_packet_data = None
            first_timestamp = None
            first_src_ip = None
            first_dst_ip = None

            try:
                first_timestamp, first_packet_data = next(pcap_reader)
            except StopIteration:
                error_msg = "Empty pcap file (no packets)"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False


            try:
                ip, _ = extract_ip_from_packet(first_packet_data, datalink)
                if ip is None:
                    error_msg = "Failed to extract IP from first packet"
                    if debug_log:
                        with open(debug_log, 'a') as f:
                            f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                    return False

                first_src_ip = socket.inet_ntoa(ip.src)
                first_dst_ip = socket.inet_ntoa(ip.dst)
                first_src_ip_bytes = ip.src
                first_dst_ip_bytes = ip.dst
            except Exception as e:

                error_msg = f"Failed to parse first packet: {str(e)}"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False


            random_src_ip_bytes = socket.inet_aton(random_ipv4())
            random_dst_ip_bytes = socket.inet_aton(random_ipv4())


            temp_fd, temp_path = tempfile.mkstemp(suffix='.pcap', dir=os.path.dirname(pcap_path))
            temp_file = os.fdopen(temp_fd, 'wb')


            pcap_writer = dpkt.pcap.Writer(temp_file, linktype=datalink)


            modified_data = modify_packet_binary(
                first_packet_data, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
                random_src_ip_bytes, random_dst_ip_bytes
            )
            if modified_data:
                pcap_writer.writepkt(modified_data, first_timestamp)
                packet_count = 1
            else:
                packet_count = 0


            for timestamp, packet_data in pcap_reader:
                try:
                    modified_data = modify_packet_binary(
                        packet_data, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
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
            error_msg = "No valid packets processed"
            if debug_log:
                with open(debug_log, 'a') as f:
                    f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return False


        try:
            os.replace(temp_path, pcap_path)
            return True
        except Exception as e:
            error_msg = f"Failed to replace file: {str(e)}"
            if debug_log:
                with open(debug_log, 'a') as f:
                    f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False

    except Exception as e:

        error_msg = f"Unexpected error: {str(e)}"
        if debug_log:
            with open(debug_log, 'a') as f:
                f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def modify_packet_binary(packet_data, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
                         random_src_ip_bytes, random_dst_ip_bytes):
    try:
        if len(packet_data) < ip_start + 20:
            return None


        ip_header = packet_data[ip_start:ip_start + 20]
        ip_version = (ip_header[0] >> 4) & 0x0F
        if ip_version != 4:
            return None


        ip_header_len = (ip_header[0] & 0x0F) * 4
        if len(packet_data) < ip_start + ip_header_len:
            return None


        orig_src_ip_bytes = ip_header[12:16]


        is_forward = (orig_src_ip_bytes == first_src_ip_bytes)


        modified_data = bytearray(packet_data)


        if is_forward:
            modified_data[ip_start + 12:ip_start + 16] = random_src_ip_bytes
            modified_data[ip_start + 16:ip_start + 20] = random_dst_ip_bytes
        else:
            modified_data[ip_start + 12:ip_start + 16] = random_dst_ip_bytes
            modified_data[ip_start + 16:ip_start + 20] = random_src_ip_bytes


        protocol = ip_header[9]


        if protocol == 6:
            tcp_start = ip_start + ip_header_len
            if len(modified_data) >= tcp_start + 4:

                modified_data[tcp_start:tcp_start + 2] = b'\x00\x00'
                modified_data[tcp_start + 2:tcp_start + 4] = b'\x00\x00'
        elif protocol == 17:
            udp_start = ip_start + ip_header_len
            if len(modified_data) >= udp_start + 4:

                modified_data[udp_start:udp_start + 2] = b'\x00\x00'
                modified_data[udp_start + 2:udp_start + 4] = b'\x00\x00'


        return bytes(modified_data)

    except Exception as e:
        return None


def make_flow_key(src_ip, dst_ip, src_port, dst_port, protocol):
    forward = (src_ip, dst_ip, src_port, dst_port, protocol)
    reverse = (dst_ip, src_ip, dst_port, src_port, protocol)

    return min(forward, reverse)


def split_pcap_by_flow(pcap_path, output_dir):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_log = os.path.join(script_dir, 'split_flows_debug.log')

    try:
        base_name = os.path.splitext(os.path.basename(pcap_path))[0]


        flows = {}

        with open(pcap_path, 'rb') as f_in:

            pcap_reader, datalink = open_pcap_reader(f_in)
            if pcap_reader is None:
                with open(debug_log, 'a') as f:
                    f.write(f"Failed to read pcap/pcapng: {pcap_path}\n")
                return 0


            ip_offset = get_ip_offset(datalink)
            if ip_offset is None:
                with open(debug_log, 'a') as f:
                    f.write(f"Unsupported datalink type {datalink}: {pcap_path}\n")
                return 0

            for timestamp, packet_data in pcap_reader:
                try:

                    ip, _ = extract_ip_from_packet(packet_data, datalink)
                    if ip is None:
                        continue

                    src_ip = ip.src
                    dst_ip = ip.dst
                    protocol = ip.p


                    src_port = 0
                    dst_port = 0
                    if protocol == 6:
                        try:
                            tcp = ip.data
                            src_port = tcp.sport
                            dst_port = tcp.dport
                        except:
                            pass
                    elif protocol == 17:
                        try:
                            udp = ip.data
                            src_port = udp.sport
                            dst_port = udp.dport
                        except:
                            pass


                    flow_key = make_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)

                    if flow_key not in flows:
                        flows[flow_key] = []
                    flows[flow_key].append((timestamp, packet_data))

                except Exception:
                    continue

        if len(flows) == 0:
            with open(debug_log, 'a') as f:
                f.write(f"No flows found in: {pcap_path}\n")
            return 0


        processed_count = 0
        failed_count = 0

        for idx, (flow_key, packets) in enumerate(flows.items(), 1):

            flow_filename = f"{base_name}_flow{idx:04d}.pcap"
            flow_path = os.path.join(output_dir, flow_filename)

            try:

                with open(flow_path, 'wb') as f_out:
                    pcap_writer = dpkt.pcap.Writer(f_out, linktype=datalink)
                    for ts, pkt_data in packets:
                        pcap_writer.writepkt(pkt_data, ts)
                    pcap_writer.close()


                if anonymize_flow_pcap(flow_path, debug_log=debug_log):
                    processed_count += 1
                else:

                    failed_count += 1
                    if os.path.exists(flow_path):
                        os.remove(flow_path)

            except Exception as e:
                failed_count += 1
                if os.path.exists(flow_path):
                    try:
                        os.remove(flow_path)
                    except:
                        pass
                continue


            if idx % 10 == 0:
                gc.collect()


        del flows
        gc.collect()


        with open(debug_log, 'a') as f:
            f.write(f"\nSplit Summary for {base_name}:\n")
            f.write(f"  Total flows: {len(flows) if 'flows' in dir() else idx}\n")
            f.write(f"  Successfully processed: {processed_count}\n")
            f.write(f"  Failed: {failed_count}\n")
            if processed_count + failed_count > 0:
                f.write(f"  Success rate: {processed_count/(processed_count+failed_count)*100:.2f}%\n")
            f.write("\n")

        return processed_count

    except Exception as e:
        with open(debug_log, 'a') as f:
            f.write(f"\nERROR: {pcap_path}: {str(e)}\n")
        return 0


def split_pcap_by_flow_wrapper(args):
    pcap_path, output_dir = args
    return split_pcap_by_flow(pcap_path, output_dir)


def process_dataset_directory(dataset_path, output_root, num_workers=None, processing_mode='auto'):
    dataset_name = os.path.basename(dataset_path)


    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)

    class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]

    class_dirs = [d for d in class_dirs
                 if not os.path.basename(d).startswith('.')
                 and not os.path.basename(d).startswith('__')]


    direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcap'))
    if len(direct_pcap_files) == 0:
        direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcapng'))


    is_mode2 = False
    if processing_mode == 'mode2':

        if len(direct_pcap_files) == 0:
            return
        class_dirs = [dataset_path]
        is_mode2 = True
    elif processing_mode == 'mode1':

        if len(class_dirs) == 0:
            return
        is_mode2 = False
    else:

        if len(class_dirs) == 0:

            if len(direct_pcap_files) == 0:
                return
            class_dirs = [dataset_path]
            is_mode2 = True
        else:
            is_mode2 = False

    total_pcap_files = 0
    total_flows = 0
    start_time = time.time()


    for class_dir in tqdm(class_dirs, desc="处理类别"):
        class_name = os.path.basename(class_dir)


        if class_name.startswith('.') or class_name.startswith('__'):
            continue

        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))


        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '*.pcapng'))


        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
            if len(pcap_files) == 0:
                pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcapng'), recursive=True)

        if len(pcap_files) == 0:
            continue

        if is_mode2:
            output_class_dir = os.path.join(output_root, dataset_name)
        else:
            output_class_dir = os.path.join(output_root, dataset_name, class_name)
        os.makedirs(output_class_dir, exist_ok=True)


        task_args = [
            (pcap_file, output_class_dir)
            for pcap_file in pcap_files
        ]


        class_flows = 0
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(split_pcap_by_flow_wrapper, task_args),
                total=len(task_args),
                desc=f"  {class_name}",
                leave=False
            ))
            class_flows = sum(results)

        total_pcap_files += len(pcap_files)
        total_flows += class_flows

        gc.collect()


    elapsed_time = time.time() - start_time


def process_all_datasets(datasets_root, output_root, dataset_names, num_workers=None, processing_mode='auto'):

    if not check_dpkt_available():
        return


    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)


    valid_modes = ['mode1', 'mode2', 'auto']
    if processing_mode not in valid_modes:
        return

    total_start_time = time.time()

    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)

        if not os.path.exists(dataset_path):
            continue

        try:
            process_dataset_directory(
                dataset_path=dataset_path,
                output_root=output_root,
                num_workers=num_workers,
                processing_mode=processing_mode
            )
        except Exception as e:
            continue

    total_elapsed = time.time() - total_start_time


if __name__ == '__main__':

    try:
        from config import DATASETS_ROOT, DATASETS, PROCESSING_MODE
    except ImportError:
        DATASETS_ROOT = ""
        DATASETS = []
        PROCESSING_MODE = None


    if PROCESSING_MODE is None:
        PROCESSING_MODE = 'auto'


    OUTPUT_ROOT = ""


    NUM_WORKERS = 6


    if NUM_WORKERS is None:
        actual_workers = max(1, cpu_count() - 4)
    else:
        actual_workers = NUM_WORKERS


    process_all_datasets(
        datasets_root=DATASETS_ROOT,
        output_root=OUTPUT_ROOT,
        dataset_names=DATASETS,
        num_workers=NUM_WORKERS,
        processing_mode=PROCESSING_MODE
    )
