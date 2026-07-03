#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
使用 dpkt 将pcap文件按流(5-tuple)拆分（纯Python实现，无需Mono/SplitCap）

按5元组(src_ip, dst_ip, src_port, dst_port, protocol)进行双向流合并拆分
将每条流保存为单独的pcap文件

功能：
1. 使用 dpkt 按5-tuple双向流拆分pcap文件
2. 对每条单流pcap文件进行IP地址随机化（每条流内所有包使用相同的随机IP）
3. 将端口号清零（TCP/UDP端口设置为0）
"""

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
    """
    根据pcap文件的datalink类型返回IP头在packet_data中的字节偏移

    常见datalink类型:
        1   (DLT_EN10MB)    - 以太网帧，IP头偏移14字节
        101 (DLT_RAW)       - 裸IP包，IP头偏移0字节
        12  (DLT_RAW 旧版)  - 裸IP包，IP头偏移0字节
        113 (DLT_LINUX_SLL) - Linux Cooked Capture，IP头偏移16字节
        276 (DLT_LINUX_SLL2)- Linux Cooked Capture v2，IP头偏移20字节

    返回:
        int: IP头的字节偏移量，未知类型返回None
    """
    offset_map = {
        1: 14,      # Ethernet
        101: 0,     # Raw IP
        12: 0,      # Raw IP (旧版DLT值)
        113: 16,    # Linux SLL
        276: 20,    # Linux SLL2
    }
    return offset_map.get(datalink, None)


def open_pcap_reader(file_handle):
    """
    尝试用 dpkt 打开 pcap 或 pcapng 文件

    参数:
        file_handle: 已打开的文件对象（rb模式）

    返回:
        (reader, datalink) 元组，reader 是 dpkt 的 Reader 对象，datalink 是链路层类型
        失败返回 (None, None)
    """
    # 先尝试标准 pcap
    try:
        file_handle.seek(0)
        reader = dpkt.pcap.Reader(file_handle)
        datalink = reader.datalink()
        return reader, datalink
    except (ValueError, Exception):
        pass

    # 再尝试 pcapng
    try:
        file_handle.seek(0)
        reader = dpkt.pcapng.Reader(file_handle)
        datalink = reader.datalink()
        return reader, datalink
    except (ValueError, AttributeError, Exception):
        pass

    return None, None


def extract_ip_from_packet(packet_data, datalink):
    """
    根据datalink类型从packet_data中提取IP层对象

    参数:
        packet_data: 原始包数据
        datalink: pcap文件的datalink类型

    返回:
        (ip_obj, ip_offset) 或 (None, None) 如果解析失败
    """
    ip_offset = get_ip_offset(datalink)
    if ip_offset is None:
        return None, None

    if len(packet_data) < ip_offset + 20:  # IP头至少20字节
        return None, None

    try:
        if datalink == 1:  # 以太网
            eth = dpkt.ethernet.Ethernet(packet_data)
            if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                return eth.ip, ip_offset
            elif eth.type == dpkt.ethernet.ETH_TYPE_IP6:
                return None, None  # 跳过IPv6
            else:
                return None, None
        else:
            # Raw IP 或 Linux SLL: 从偏移位置直接解析IP
            ip = dpkt.ip.IP(packet_data[ip_offset:])
            if ip.v != 4:
                return None, None
            return ip, ip_offset
    except Exception:
        return None, None


def check_dpkt_available():
    """
    检查 dpkt 是否已安装
    """
    try:
        import dpkt
        print(f"找到 dpkt: version {dpkt.__version__}")
        return True
    except ImportError:
        print("错误：未找到 dpkt")
        print("\n安装方法：")
        print("  pip install dpkt")
        return False


def random_ipv4():
    """生成随机IPv4地址"""
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def anonymize_flow_pcap(pcap_path, debug_log=None):
    """
    对单流pcap文件进行IP地址随机化和端口清零（使用dpkt直接操作二进制数据）
    每条流内的所有包使用相同的随机IP地址对，端口设置为0

    优化策略：
    1. 使用dpkt库直接操作二进制数据，比Scapy快得多
    2. 流式处理，避免一次性加载全部数据到内存
    3. 在同一目录创建临时文件，使用os.replace进行原子替换
    4. 从第一个包中预先提取IP头信息，避免每个包都重复解析

    参数:
        pcap_path: pcap文件路径
        debug_log: 可选，debug日志文件路径（字符串），如果提供则记录详细错误信息到该文件

    返回:
        bool: 是否成功处理
    """
    temp_path = None

    try:
        # 打开输入pcap文件
        with open(pcap_path, 'rb') as f_in:
            # 读取pcap/pcapng文件头
            pcap_reader, datalink = open_pcap_reader(f_in)
            if pcap_reader is None:
                error_msg = "Failed to read pcap/pcapng header"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False

            # 根据datalink类型确定IP头偏移
            ip_start = get_ip_offset(datalink)
            if ip_start is None:
                error_msg = f"Unsupported datalink type: {datalink}"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False

            # 读取第一个包以确定流参数
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

            # 解析第一个包，提取IP头信息
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
                # 解析失败，返回False
                error_msg = f"Failed to parse first packet: {str(e)}"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False

            # 为这条流生成随机IP地址对（所有包使用相同的IP对）
            random_src_ip_bytes = socket.inet_aton(random_ipv4())
            random_dst_ip_bytes = socket.inet_aton(random_ipv4())

            # 创建临时文件用于写入
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pcap', dir=os.path.dirname(pcap_path))
            temp_file = os.fdopen(temp_fd, 'wb')

            # 创建pcap写入器（保持原始datalink类型）
            pcap_writer = dpkt.pcap.Writer(temp_file, linktype=datalink)

            # 处理第一个包
            modified_data = modify_packet_binary(
                first_packet_data, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
                random_src_ip_bytes, random_dst_ip_bytes
            )
            if modified_data:
                pcap_writer.writepkt(modified_data, first_timestamp)
                packet_count = 1
            else:
                packet_count = 0

            # 流式处理剩余的包
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
                    # 单个包处理失败，跳过该包
                    continue
                finally:
                    del packet_data

            # 关闭文件
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

        # 原子替换原文件
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
        # 清理临时文件
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
    """
    直接修改包的二进制数据：随机化IP地址和清零端口

    参数:
        packet_data: 包的原始二进制数据
        ip_start: IP头起始位置（根据datalink类型预先计算）
        first_src_ip_bytes: 第一个包的源IP（字节格式）
        first_dst_ip_bytes: 第一个包的目标IP（字节格式）
        random_src_ip_bytes: 随机源IP（字节格式）
        random_dst_ip_bytes: 随机目标IP（字节格式）

    返回:
        bytes: 修改后的包数据，如果失败返回None
    """
    try:
        if len(packet_data) < ip_start + 20:  # IP头至少20字节
            return None

        # 解析IP头
        ip_header = packet_data[ip_start:ip_start + 20]
        ip_version = (ip_header[0] >> 4) & 0x0F
        if ip_version != 4:
            return None

        # 获取IP头长度
        ip_header_len = (ip_header[0] & 0x0F) * 4
        if len(packet_data) < ip_start + ip_header_len:
            return None

        # 获取原始IP地址（字节格式）
        orig_src_ip_bytes = ip_header[12:16]

        # 判断包的方向（直接比较字节，避免转换）
        is_forward = (orig_src_ip_bytes == first_src_ip_bytes)

        # 创建修改后的包数据
        modified_data = bytearray(packet_data)

        # 修改IP地址（保持方向性）
        if is_forward:
            modified_data[ip_start + 12:ip_start + 16] = random_src_ip_bytes
            modified_data[ip_start + 16:ip_start + 20] = random_dst_ip_bytes
        else:
            modified_data[ip_start + 12:ip_start + 16] = random_dst_ip_bytes
            modified_data[ip_start + 16:ip_start + 20] = random_src_ip_bytes

        # 获取协议类型
        protocol = ip_header[9]

        # 清零端口（TCP或UDP）
        if protocol == 6:  # TCP
            tcp_start = ip_start + ip_header_len
            if len(modified_data) >= tcp_start + 4:
                # 清零源端口和目标端口（各2字节）
                modified_data[tcp_start:tcp_start + 2] = b'\x00\x00'
                modified_data[tcp_start + 2:tcp_start + 4] = b'\x00\x00'
        elif protocol == 17:  # UDP
            udp_start = ip_start + ip_header_len
            if len(modified_data) >= udp_start + 4:
                # 清零源端口和目标端口（各2字节）
                modified_data[udp_start:udp_start + 2] = b'\x00\x00'
                modified_data[udp_start + 2:udp_start + 4] = b'\x00\x00'

        # 注意：不重新计算校验和，保持原校验和值（虽然会失效，但不影响数据提取）

        return bytes(modified_data)

    except Exception as e:
        return None


def make_flow_key(src_ip, dst_ip, src_port, dst_port, protocol):
    """
    生成双向流的key，确保同一条流的两个方向映射到同一个key
    通过排序确保 (A->B) 和 (B->A) 生成相同的key
    """
    forward = (src_ip, dst_ip, src_port, dst_port, protocol)
    reverse = (dst_ip, src_ip, dst_port, src_port, protocol)
    # 选择较小的元组作为key，确保双向流合并
    return min(forward, reverse)


def split_pcap_by_flow(pcap_path, output_dir):
    """
    使用 dpkt 将一个pcap文件按5-tuple双向流拆分

    按5元组(src_ip, dst_ip, src_port, dst_port, protocol)分组，
    双向流合并（A->B 和 B->A 视为同一条流）

    参数:
        pcap_path: 输入pcap文件路径
        output_dir: 输出目录

    返回:
        flows_count: 成功处理的流数量
    """
    # debug日志文件生成在脚本同目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_log = os.path.join(script_dir, 'split_flows_debug.log')

    try:
        base_name = os.path.splitext(os.path.basename(pcap_path))[0]

        # 读取pcap文件，按5-tuple分组
        flows = {}  # key: flow_key, value: list of (timestamp, packet_data)

        with open(pcap_path, 'rb') as f_in:
            # 尝试打开 pcap 或 pcapng 文件
            pcap_reader, datalink = open_pcap_reader(f_in)
            if pcap_reader is None:
                with open(debug_log, 'a') as f:
                    f.write(f"Failed to read pcap/pcapng: {pcap_path}\n")
                return 0

            # 获取datalink类型，确定IP头偏移
            ip_offset = get_ip_offset(datalink)
            if ip_offset is None:
                with open(debug_log, 'a') as f:
                    f.write(f"Unsupported datalink type {datalink}: {pcap_path}\n")
                return 0

            for timestamp, packet_data in pcap_reader:
                try:
                    # 根据datalink类型提取IP层
                    ip, _ = extract_ip_from_packet(packet_data, datalink)
                    if ip is None:
                        continue

                    src_ip = ip.src
                    dst_ip = ip.dst
                    protocol = ip.p

                    # 提取端口号
                    src_port = 0
                    dst_port = 0
                    if protocol == 6:  # TCP
                        try:
                            tcp = ip.data
                            src_port = tcp.sport
                            dst_port = tcp.dport
                        except:
                            pass
                    elif protocol == 17:  # UDP
                        try:
                            udp = ip.data
                            src_port = udp.sport
                            dst_port = udp.dport
                        except:
                            pass

                    # 生成双向流key
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

        # 将每条流写入单独的pcap文件，然后进行匿名化
        processed_count = 0
        failed_count = 0

        for idx, (flow_key, packets) in enumerate(flows.items(), 1):
            # 生成流文件名
            flow_filename = f"{base_name}_flow{idx:04d}.pcap"
            flow_path = os.path.join(output_dir, flow_filename)

            try:
                # 写入流pcap文件（保持与原始pcap相同的datalink类型）
                with open(flow_path, 'wb') as f_out:
                    pcap_writer = dpkt.pcap.Writer(f_out, linktype=datalink)
                    for ts, pkt_data in packets:
                        pcap_writer.writepkt(pkt_data, ts)
                    pcap_writer.close()

                # 对流文件进行匿名化处理
                if anonymize_flow_pcap(flow_path, debug_log=debug_log):
                    processed_count += 1
                else:
                    # 匿名化失败，删除该流文件
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

            # 每处理10条流强制垃圾回收一次，防止内存累积
            if idx % 10 == 0:
                gc.collect()

        # 释放内存
        del flows
        gc.collect()

        # 记录处理结果
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
    """
    多进程包装函数
    """
    pcap_path, output_dir = args
    return split_pcap_by_flow(pcap_path, output_dir)


def process_dataset_directory(dataset_path, output_root, num_workers=None, processing_mode='auto'):
    """
    处理整个数据集目录（使用多进程 + dpkt流拆分）

    支持两种模式：
    1. 模式1 ('mode1'): 数据集目录下包含多个类别子目录（每个子目录包含pcap文件）
    2. 模式2 ('mode2'): 数据集目录下直接包含pcap文件（作为单个类别处理）
    3. 自动模式 ('auto'): 自动检测使用哪种模式

    参数:
        dataset_path: 数据集目录
        output_root: 输出根目录
        num_workers: 并行进程数（None=自动检测）
        processing_mode: 处理模式 ('mode1', 'mode2', 'auto')
    """
    dataset_name = os.path.basename(dataset_path)

    # 自动检测CPU核心数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)  # 保留4个核心给系统

    print(f"\n{'='*70}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"输入目录: {dataset_path}")
    print(f"输出目录: {output_root}")
    print(f"并行进程数: {num_workers}")
    print(f"处理模式: {processing_mode}\n")

    # 查找所有类别目录
    class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
    # 过滤掉临时目录
    class_dirs = [d for d in class_dirs
                 if not os.path.basename(d).startswith('.')
                 and not os.path.basename(d).startswith('__')]

    # 检查数据集目录下是否直接包含pcap文件
    direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcap'))
    if len(direct_pcap_files) == 0:
        direct_pcap_files = glob.glob(os.path.join(dataset_path, '*.pcapng'))

    # 根据处理模式判断
    is_mode2 = False
    if processing_mode == 'mode2':
        # 手动选择模式2: 数据集目录直接包含pcap文件
        if len(direct_pcap_files) == 0:
            print(f"模式2: 未找到pcap文件: {dataset_path}")
            return
        print(f"使用模式2: 数据集目录直接包含pcap文件，作为单个类别处理")
        print(f"找到 {len(direct_pcap_files)} 个pcap文件\n")
        class_dirs = [dataset_path]
        is_mode2 = True
    elif processing_mode == 'mode1':
        # 手动选择模式1: 数据集目录包含多个类别子目录
        if len(class_dirs) == 0:
            print(f"模式1: 未找到类别子目录: {dataset_path}")
            return
        print(f"使用模式1: 数据集目录包含 {len(class_dirs)} 个类别子目录\n")
        is_mode2 = False
    else:
        # 自动模式: 自动检测
        if len(class_dirs) == 0:
            # 没有子目录，检查是否有直接pcap文件
            if len(direct_pcap_files) == 0:
                print(f"未找到类别目录或pcap文件: {dataset_path}")
                return
            print(f"自动检测: 模式2 - 数据集目录直接包含pcap文件，作为单个类别处理")
            print(f"找到 {len(direct_pcap_files)} 个pcap文件\n")
            class_dirs = [dataset_path]
            is_mode2 = True
        else:
            print(f"自动检测: 模式1 - 数据集目录包含 {len(class_dirs)} 个类别子目录\n")
            is_mode2 = False

    total_pcap_files = 0
    total_flows = 0
    start_time = time.time()

    # 处理每个类别
    for class_dir in tqdm(class_dirs, desc="处理类别"):
        class_name = os.path.basename(class_dir)

        # 跳过临时目录
        if class_name.startswith('.') or class_name.startswith('__'):
            continue

        print(f"\n处理类别: {class_name}")

        # 查找该类别下的所有pcap文件
        pcap_files = glob.glob(os.path.join(class_dir, '*.pcap'))

        # 如果没有找到.pcap文件，尝试查找.pcapng文件
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '*.pcapng'))

        # 如果当前目录没有，递归查找子目录
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
            if len(pcap_files) == 0:
                pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcapng'), recursive=True)

        if len(pcap_files) == 0:
            print(f"  未找到pcap文件")
            continue

        print(f"  找到 {len(pcap_files)} 个pcap文件，使用 {num_workers} 个进程并行处理...")

        # 创建输出目录
        if is_mode2:
            output_class_dir = os.path.join(output_root, dataset_name)
        else:
            output_class_dir = os.path.join(output_root, dataset_name, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # 准备参数列表（用于多进程）
        task_args = [
            (pcap_file, output_class_dir)
            for pcap_file in pcap_files
        ]

        # 使用进程池并行处理
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

        print(f"  从 {len(pcap_files)} 个文件中提取了 {class_flows} 条流")
        gc.collect()

    # 汇总统计
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"{dataset_name} 处理完成!")
    print(f"{'='*70}")
    print(f"原始pcap文件: {total_pcap_files}")
    print(f"提取的流: {total_flows}")
    print(f"平均每个文件: {total_flows/total_pcap_files:.1f} 条流" if total_pcap_files > 0 else "")
    print(f"处理时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"输出位置: {output_root}/{dataset_name}/")


def process_all_datasets(datasets_root, output_root, dataset_names, num_workers=None, processing_mode='auto'):
    """
    批量处理多个数据集（使用多进程 + dpkt纯Python流拆分）

    参数:
        datasets_root: 数据集根目录
        output_root: 输出根目录
        dataset_names: 数据集名称列表
        num_workers: 并行进程数（None=自动检测）
        processing_mode: 处理模式 ('mode1', 'mode2', 'auto')
            - 'mode1': 数据集目录包含多个类别子目录
            - 'mode2': 数据集目录直接包含pcap文件
            - 'auto': 自动检测模式
    """
    # 检查 dpkt 是否可用
    if not check_dpkt_available():
        print("\n终止执行")
        return

    # 自动检测CPU核心数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)  # 保留4个核心给系统

    # 验证处理模式
    valid_modes = ['mode1', 'mode2', 'auto']
    if processing_mode not in valid_modes:
        print(f"无效的处理模式: {processing_mode}")
        print(f"   有效模式: {', '.join(valid_modes)}")
        return

    print("\n" + "=" * 70)
    print("批量流拆分工具（dpkt 纯Python多进程版本）")
    print("=" * 70)
    print(f"\n数据集根目录: {datasets_root}")
    print(f"输出根目录: {output_root}")
    print(f"待处理数据集: {', '.join(dataset_names)}")
    print(f"处理模式: {processing_mode}")
    print(f"CPU核心数: {cpu_count()}")
    print(f"使用进程数: {num_workers}")
    print(f"工具: dpkt (纯Python, 按5-tuple双向流拆分)")

    total_start_time = time.time()

    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"\n数据集不存在: {dataset_path}")
            continue

        try:
            process_dataset_directory(
                dataset_path=dataset_path,
                output_root=output_root,
                num_workers=num_workers,
                processing_mode=processing_mode
            )
        except Exception as e:
            print(f"\n处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 70)
    print("所有数据集处理完成!")
    print("=" * 70)
    print(f"总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"平均速度: 使用 {num_workers} 进程 + dpkt 并行处理")


if __name__ == '__main__':
    # 配置
    try:
        from config import DATASETS_ROOT, DATASETS, PROCESSING_MODE
        print("从config.py加载配置")
    except ImportError:
        print("config.py未找到，使用默认配置")
        # DATASETS_ROOT可以选择两种模式，见mode_descriptions
        DATASETS_ROOT = "/root/datasets/original_datasets/original_pretraining_datasets/"
        DATASETS = ["NonVPN", "browser", "IDS"]
        PROCESSING_MODE = None  # 如果未定义，使用下面的默认值

    # 处理模式配置
    # 'mode1': 数据集目录包含多个类别子目录（每个子目录包含pcap文件）
    # 'mode2': 数据集目录直接包含pcap文件（作为单个类别处理）
    # 'auto': 自动检测模式（推荐）
    if PROCESSING_MODE is None:
        PROCESSING_MODE = 'auto'  # 默认使用自动模式

    # 输出目录（拆分后的单流pcap文件）
    OUTPUT_ROOT = "/root/datasets/单流Pcap数据集_预训练_随机化/"

    # 流拆分参数
    NUM_WORKERS = 6        # 并行进程数（手动设置为6个进程）

    # 如果想使用更多进程，可以修改为：
    # NUM_WORKERS = 40   # 使用40个进程
    # NUM_WORKERS = None # 自动使用 CPU核心数-4 (64核会使用60个进程)

    print("\n" + "=" * 70)
    print("流拆分配置")
    print("=" * 70)
    print(f"工具: dpkt (纯Python, 按5-tuple双向流拆分)")
    print(f"CPU核心数: {cpu_count()}")
    if NUM_WORKERS is None:
        actual_workers = max(1, cpu_count() - 4)
        print(f"并行进程数: {actual_workers} (自动: CPU核心数-4)")
    else:
        print(f"并行进程数: {NUM_WORKERS} (手动设置)")

    # 处理模式说明
    mode_descriptions = {
        'mode1': '模式1: 数据集目录包含多个类别子目录',
        'mode2': '模式2: 数据集目录直接包含pcap文件',
        'auto': '自动模式: 自动检测使用哪种模式'
    }
    print(f"处理模式: {PROCESSING_MODE} - {mode_descriptions.get(PROCESSING_MODE, '未知模式')}")

    print("\n功能说明：")
    print("  1. 使用 dpkt 按5-tuple双向流拆分pcap文件为单流文件")
    print("  2. 对每条流进行IP地址随机化（每条流内所有包使用相同的随机IP）")
    print("  3. 将TCP/UDP端口号清零（设置为0）")
    print("\n处理模式说明：")
    print("  - mode1: 数据集目录下包含多个类别子目录，每个子目录是一个类别")
    print("           输出路径: output_root/dataset_name/class_name/")
    print("  - mode2: 数据集目录下直接包含pcap文件，作为单个类别处理")
    print("           输出路径: output_root/dataset_name/")
    print("  - auto:  自动检测目录结构，选择合适的模式")
    print("\n安装要求：")
    print("  pip install dpkt tqdm")
    print("=" * 70)

    # 执行批量处理
    process_all_datasets(
        datasets_root=DATASETS_ROOT,
        output_root=OUTPUT_ROOT,
        dataset_names=DATASETS,
        num_workers=NUM_WORKERS,
        processing_mode=PROCESSING_MODE
    )

    print("\n后续步骤:")
    print(f"  1. 查看拆分结果: {OUTPUT_ROOT}")
    print(f"  2. 使用拆分后的单流pcap文件进行MFR矩阵生成")
    print(f"  3. 更新data_process.py中的输入路径为: {OUTPUT_ROOT}")
    print("")
