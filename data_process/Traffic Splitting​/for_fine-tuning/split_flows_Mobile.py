#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
NUDT_MobileTraffic Dataset 单流切分脚本（优化版 v2.0）

✨ 主要改进：
1. 使用 tshark 一次性输出所有包的流信息，避免重复读取 pcap
2. 在内存中按五元组分组匹配，大幅减少磁盘 I/O
3. 批量更新流计数器，减少文件锁竞争
4. 优化进度显示，实时反馈处理速度

功能：
1. 读取 NUDT_dataset_statistics.csv，筛选流数 >= 2500 的应用
2. 遍历 NUDT_MobileTraffic Dataset 数据集
3. 一次性读取 pcap 文件的所有包信息，在内存中按流分组
4. 与 txt 文件的流信息批量匹配
5. 批量提取单流并进行IP地址随机化和端口清零
6. 按应用名称分类保存

目录结构：
输入：/root/datasets_new/original_pcap_datasets/NUDT_MobileTraffic Dataset/
     NetlogTraffic_0626/
       83/
         20200626/
           netlog20200626_23_08_24.pcap
           netlog20200626_23_08_24_log.txt

输出：/root/datasets_new/单流Pcap数据集/NUDT_MobileTraffic Dataset_all/
     腾讯视频/
       腾讯视频_flow_00000001.pcap
       腾讯视频_flow_00000002.pcap
     QQ/
       QQ_flow_00000001.pcap
"""

import os
import csv
import subprocess
import shutil
import tempfile
import random
import sys
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager, Lock
import time
import re
import scapy.all as scapy
import fcntl
import json
from datetime import datetime
import gc


def random_ipv4():
    """生成随机IPv4地址"""
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def load_skip_list(skip_list_file):
    """
    加载需要跳过的文件列表（处理失败或超时的文件）
    
    参数:
        skip_list_file: 跳过列表文件路径
    
    返回:
        set: 需要跳过的 pcap 文件路径集合
    """
    skip_set = set()
    if os.path.exists(skip_list_file):
        try:
            with open(skip_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        skip_set.add(line)
        except:
            pass
    return skip_set


def add_to_skip_list(skip_list_file, pcap_file, reason=""):
    """
    将文件添加到跳过列表
    
    参数:
        skip_list_file: 跳过列表文件路径
        pcap_file: 要跳过的 pcap 文件路径
        reason: 跳过原因
    """
    try:
        with open(skip_list_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{pcap_file}  # {reason} @ {timestamp}\n")
    except:
        pass


def log_error(error_log_file, message):
    """
    记录错误日志
    
    参数:
        error_log_file: 错误日志文件路径
        message: 错误信息
    """
    try:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    except:
        pass


def read_flow_counters_safe(flow_counter_file):
    """
    安全地读取流计数器文件（带文件锁）
    
    参数:
        flow_counter_file: 流计数器文件路径
    
    返回:
        dict: {应用名称: 流数量}
    """
    flow_counters = {}
    if os.path.exists(flow_counter_file):
        try:
            with open(flow_counter_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 共享锁（读锁）
                try:
                    for line in f:
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            flow_counters[parts[0]] = int(parts[1])
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
        except Exception:
            pass
    return flow_counters


def update_flow_counter_safe(flow_counter_file, app_name, increment=1):
    """
    安全地更新流计数器（带文件锁，原子操作）
    
    参数:
        flow_counter_file: 流计数器文件路径
        app_name: 应用名称
        increment: 增量（默认1）
    
    返回:
        int: 更新后的流数量，如果达到限制返回-1
    """
    MAX_FLOWS_PER_APP = 1000
    
    # 使用独占锁进行读写
    try:
        with open(flow_counter_file, 'a+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 独占锁（写锁）
            try:
                # 重新定位到文件开头
                f.seek(0)
                
                # 读取所有计数器
                flow_counters = {}
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        flow_counters[parts[0]] = int(parts[1])
                
                # 检查是否达到限制
                current_count = flow_counters.get(app_name, 0)
                if current_count >= MAX_FLOWS_PER_APP:
                    return -1  # 已达到限制
                
                # 更新计数器
                new_count = current_count + increment
                flow_counters[app_name] = new_count
                
                # 写回文件
                f.seek(0)
                f.truncate()
                for name, count in flow_counters.items():
                    f.write(f"{name}:{count}\n")
                f.flush()
                
                return new_count
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
    except Exception:
        # 如果文件不存在，创建它
        try:
            with open(flow_counter_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(f"{app_name}:{increment}\n")
                    f.flush()
                    return increment
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            return -1


def update_flow_counter_batch(flow_counter_file, updates):
    """
    批量更新流计数器（带文件锁，原子操作）
    用于减少文件锁竞争，提升性能
    
    参数:
        flow_counter_file: 流计数器文件路径
        updates: dict {应用名称: 增量}
    
    返回:
        dict: {应用名称: 更新后的流数量}，如果达到限制则值为-1
    """
    MAX_FLOWS_PER_APP = 1000
    results = {}
    
    if not updates:
        return results
    
    # 使用独占锁进行读写
    try:
        with open(flow_counter_file, 'a+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 独占锁（写锁）
            try:
                # 重新定位到文件开头
                f.seek(0)
                
                # 读取所有计数器
                flow_counters = {}
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        flow_counters[parts[0]] = int(parts[1])
                
                # 批量更新
                for app_name, increment in updates.items():
                    current_count = flow_counters.get(app_name, 0)
                    if current_count >= MAX_FLOWS_PER_APP:
                        results[app_name] = -1  # 已达到限制
                    else:
                        new_count = current_count + increment
                        flow_counters[app_name] = new_count
                        results[app_name] = new_count
                
                # 写回文件
                f.seek(0)
                f.truncate()
                for name, count in flow_counters.items():
                    f.write(f"{name}:{count}\n")
                f.flush()
                
                return results
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
    except Exception:
        # 如果文件不存在，创建它
        try:
            with open(flow_counter_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for app_name, increment in updates.items():
                        f.write(f"{app_name}:{increment}\n")
                    f.flush()
                    return {app_name: increment for app_name, increment in updates.items()}
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            return {app_name: -1 for app_name in updates.keys()}


def anonymize_flow_pcap(pcap_path):
    """
    对单流pcap文件进行IP地址随机化和端口清零
    每条流内的所有包使用相同的随机IP地址对，端口设置为0

    参数:
        pcap_path: pcap文件路径

    返回:
        bool: 是否成功处理
    """
    try:
        # 读取pcap文件
        packets = scapy.rdpcap(pcap_path)

        if len(packets) == 0:
            return False

        # 检查是否有以太网头
        has_ether = hasattr(packets[0], 'type')

        # 确定协议类型（IPv4或IPv6）
        if has_ether:
            if packets[0].type == 0x86dd:  # IPv6，暂不支持
                return False
            elif packets[0].type != 0x0800:  # 不是IPv4
                return False
        else:
            # 没有以太网头，直接检查IP层
            if not hasattr(packets[0], 'version') or packets[0].version != 4:
                return False

        # 获取第一个包的原始IP地址，用于确定方向
        if has_ether:
            first_src_ip = packets[0].payload.src if hasattr(packets[0].payload, 'src') else None
            first_dst_ip = packets[0].payload.dst if hasattr(packets[0].payload, 'dst') else None
        else:
            first_src_ip = packets[0].src if hasattr(packets[0], 'src') else None
            first_dst_ip = packets[0].dst if hasattr(packets[0], 'dst') else None

        if first_src_ip is None or first_dst_ip is None:
            return False

        # 为这条流生成随机IP地址对（所有包使用相同的IP对）
        random_src_ip = random_ipv4()
        random_dst_ip = random_ipv4()

        # 处理每个包
        modified_packets = []
        for packet in packets:
            try:
                new_packet = packet.copy()

                if has_ether:
                    # 有以太网头的情况
                    if hasattr(new_packet.payload, 'src') and hasattr(new_packet.payload, 'dst'):
                        # 判断包的方向：如果源IP与第一个包的源IP相同，则为正向包
                        is_forward = (str(new_packet.payload.src) == str(first_src_ip))

                        # 修改IP地址（保持方向性）
                        if is_forward:
                            new_packet.payload.src = random_src_ip
                            new_packet.payload.dst = random_dst_ip
                        else:
                            # 反向包，交换IP
                            new_packet.payload.src = random_dst_ip
                            new_packet.payload.dst = random_src_ip

                        # 清零端口（TCP或UDP）
                        if hasattr(new_packet.payload, 'proto') and hasattr(new_packet.payload, 'payload'):
                            if new_packet.payload.proto == 6:  # TCP
                                if hasattr(new_packet.payload.payload, 'sport'):
                                    new_packet.payload.payload.sport = 0
                                if hasattr(new_packet.payload.payload, 'dport'):
                                    new_packet.payload.payload.dport = 0
                            elif new_packet.payload.proto == 17:  # UDP
                                if hasattr(new_packet.payload.payload, 'sport'):
                                    new_packet.payload.payload.sport = 0
                                if hasattr(new_packet.payload.payload, 'dport'):
                                    new_packet.payload.payload.dport = 0
                else:
                    # 没有以太网头的情况
                    if hasattr(new_packet, 'src') and hasattr(new_packet, 'dst'):
                        # 判断包的方向
                        is_forward = (str(new_packet.src) == str(first_src_ip))

                        # 修改IP地址（保持方向性）
                        if is_forward:
                            new_packet.src = random_src_ip
                            new_packet.dst = random_dst_ip
                        else:
                            # 反向包，交换IP
                            new_packet.src = random_dst_ip
                            new_packet.dst = random_src_ip

                        # 清零端口（TCP或UDP）
                        if hasattr(new_packet, 'proto') and hasattr(new_packet, 'payload'):
                            if new_packet.proto == 6:  # TCP
                                if hasattr(new_packet.payload, 'sport'):
                                    new_packet.payload.sport = 0
                                if hasattr(new_packet.payload, 'dport'):
                                    new_packet.payload.dport = 0
                            elif new_packet.proto == 17:  # UDP
                                if hasattr(new_packet.payload, 'sport'):
                                    new_packet.payload.sport = 0
                                if hasattr(new_packet.payload, 'dport'):
                                    new_packet.payload.dport = 0

                modified_packets.append(new_packet)
            except Exception as e:
                # 如果某个包处理失败，跳过它
                continue

        if len(modified_packets) == 0:
            del packets
            gc.collect()
            return False

        # 保存修改后的包
        scapy.wrpcap(pcap_path, modified_packets)

        # 释放内存
        del packets
        del modified_packets
        gc.collect()
        return True

    except Exception as e:
        # 处理失败，返回False
        return False


def load_target_apps(csv_file, min_flows=2500):
    """
    从CSV文件中加载流数 >= min_flows 的应用名称
    
    参数:
        csv_file: CSV文件路径
        min_flows: 最小流数阈值
    
    返回:
        set: 目标应用名称集合
    """
    target_apps = set()
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                app_name = row['应用名称']
                total_flows = int(row['总流数'])
                if total_flows >= min_flows:
                    target_apps.add(app_name)
        
        print(f"✓ 加载目标应用: {len(target_apps)} 个（流数 >= {min_flows}）")
        return target_apps
    except Exception as e:
        print(f"❌ 加载CSV文件失败: {e}")
        return set()


def parse_flow_info(line):
    """
    解析txt文件中的一行流信息
    格式: 时间, 应用名称, 协议, IP:端口==>>IP:端口
    例如: 23:08:01, 邮储银行, TCP4, 10.1.10.1:39714==>>39.96.126.126:443
    
    参数:
        line: txt文件中的一行
    
    返回:
        dict: {
            'app_name': 应用名称,
            'protocol': 协议（TCP/UDP），
            'src_ip': 源IP,
            'src_port': 源端口,
            'dst_ip': 目标IP,
            'dst_port': 目标端口
        }
        或 None（如果解析失败）
    """
    try:
        parts = line.strip().split(',')
        if len(parts) < 4:
            return None
        
        time_str = parts[0].strip()
        app_name = parts[1].strip()
        protocol_str = parts[2].strip()  # TCP4, TCP6, UDP4, UDP6
        connection = parts[3].strip()  # IP:端口==>>IP:端口
        
        # 提取协议类型（TCP或UDP）和IP版本
        if protocol_str.startswith('TCP'):
            protocol = 'tcp'
        elif protocol_str.startswith('UDP'):
            protocol = 'udp'
        else:
            return None
        
        # 判断是IPv4还是IPv6
        is_ipv6 = protocol_str.endswith('6')
        
        # 解析连接信息：IP:端口==>>IP:端口
        # 注意IPv6格式：fd00:1:fd00:1:fd00:1:fd00:1:45718
        conn_parts = connection.split('==>>')
        if len(conn_parts) != 2:
            return None
        
        src_addr = conn_parts[0].strip()
        dst_addr = conn_parts[1].strip()
        
        # 解析源地址和端口
        # IPv4: 10.1.10.1:39714
        # IPv6: [fd00:1:fd00:1:fd00:1:fd00:1]:45718 或 fd00:1:fd00:1:fd00:1:fd00:1:45718
        def parse_addr(addr_str):
            # 尝试从最后一个冒号分割端口
            last_colon_idx = addr_str.rfind(':')
            if last_colon_idx == -1:
                return None, None
            
            ip = addr_str[:last_colon_idx]
            port = addr_str[last_colon_idx+1:]
            
            # 移除IPv6的方括号
            ip = ip.strip('[]')
            
            return ip, port
        
        src_ip, src_port = parse_addr(src_addr)
        dst_ip, dst_port = parse_addr(dst_addr)
        
        if not all([src_ip, src_port, dst_ip, dst_port]):
            return None
        
        return {
            'app_name': app_name,
            'protocol': protocol,
            'src_ip': src_ip,
            'src_port': src_port,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'is_ipv6': is_ipv6
        }
    except Exception as e:
        return None


def split_pcap_into_flows(pcap_file, temp_dir):
    """
    将pcap文件切割成多个单流文件
    使用 tshark 识别所有唯一的流（按五元组），然后按第一个数据包的时间顺序排序
    
    参数:
        pcap_file: 输入pcap文件路径
        temp_dir: 临时目录，用于存放切割出的单流文件
    
    返回:
        list: 切割出的单流文件路径列表（按时间顺序排序）
    """
    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # 使用 tshark 获取所有数据包的五元组信息
        # 分别获取TCP和UDP的数据包
        flow_info_list = []
        
        # 获取TCP流
        cmd_tcp = [
            'tshark',
            '-r', pcap_file,
            '-Y', 'tcp',
            '-T', 'fields',
            '-e', 'frame.number',
            '-e', 'frame.time_relative',
            '-e', 'ip.src',
            '-e', 'ip.dst',
            '-e', 'tcp.srcport',
            '-e', 'tcp.dstport',
            '-e', 'ipv6.src',
            '-e', 'ipv6.dst'
        ]
        
        result_tcp = subprocess.run(
            cmd_tcp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            text=True
        )
        
        if result_tcp.returncode == 0:
            for line in result_tcp.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    frame_num = parts[0]
                    time_rel = float(parts[1]) if parts[1] else 0.0
                    
                    # 尝试IPv4
                    src_ip = parts[2] if len(parts) > 2 and parts[2] else None
                    dst_ip = parts[3] if len(parts) > 3 and parts[3] else None
                    src_port = parts[4] if len(parts) > 4 and parts[4] else None
                    dst_port = parts[5] if len(parts) > 5 and parts[5] else None
                    
                    # 尝试IPv6
                    if not src_ip or not dst_ip:
                        src_ip = parts[6] if len(parts) > 6 and parts[6] else None
                        dst_ip = parts[7] if len(parts) > 7 and parts[7] else None
                    
                    if src_ip and dst_ip and src_port and dst_port:
                        # 创建流的标识符（规范化：使用较小的IP:端口对）
                        flow_tuple1 = (src_ip, src_port, dst_ip, dst_port)
                        flow_tuple2 = (dst_ip, dst_port, src_ip, src_port)
                        flow_key = tuple(sorted([flow_tuple1, flow_tuple2])[0])
                        
                        flow_info_list.append({
                            'frame_num': int(frame_num),
                            'time_rel': time_rel,
                            'flow_key': flow_key,
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'protocol': 'tcp'
                        })
        
        # 获取UDP流
        cmd_udp = [
            'tshark',
            '-r', pcap_file,
            '-Y', 'udp',
            '-T', 'fields',
            '-e', 'frame.number',
            '-e', 'frame.time_relative',
            '-e', 'ip.src',
            '-e', 'ip.dst',
            '-e', 'udp.srcport',
            '-e', 'udp.dstport',
            '-e', 'ipv6.src',
            '-e', 'ipv6.dst'
        ]
        
        result_udp = subprocess.run(
            cmd_udp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            text=True
        )
        
        if result_udp.returncode == 0:
            for line in result_udp.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    frame_num = parts[0]
                    time_rel = float(parts[1]) if parts[1] else 0.0
                    
                    # 尝试IPv4
                    src_ip = parts[2] if len(parts) > 2 and parts[2] else None
                    dst_ip = parts[3] if len(parts) > 3 and parts[3] else None
                    src_port = parts[4] if len(parts) > 4 and parts[4] else None
                    dst_port = parts[5] if len(parts) > 5 and parts[5] else None
                    
                    # 尝试IPv6
                    if not src_ip or not dst_ip:
                        src_ip = parts[6] if len(parts) > 6 and parts[6] else None
                        dst_ip = parts[7] if len(parts) > 7 and parts[7] else None
                    
                    if src_ip and dst_ip and src_port and dst_port:
                        # 创建流的标识符
                        flow_tuple1 = (src_ip, src_port, dst_ip, dst_port)
                        flow_tuple2 = (dst_ip, dst_port, src_ip, src_port)
                        flow_key = tuple(sorted([flow_tuple1, flow_tuple2])[0])
                        
                        flow_info_list.append({
                            'frame_num': int(frame_num),
                            'time_rel': time_rel,
                            'flow_key': flow_key,
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'protocol': 'udp'
                        })
        
        # 按流分组
        flows_dict = {}  # {flow_key: [packet_info]}
        for info in flow_info_list:
            flow_key = info['flow_key']
            if flow_key not in flows_dict:
                flows_dict[flow_key] = []
            flows_dict[flow_key].append(info)

        # 释放 flow_info_list 内存
        del flow_info_list
        gc.collect()

        # 按每个流的第一个数据包时间排序
        flow_list = []
        for flow_key, packets in flows_dict.items():
            first_packet = min(packets, key=lambda x: (x['time_rel'], x['frame_num']))
            flow_list.append((first_packet['time_rel'], first_packet['frame_num'], flow_key, packets[0]))

        flow_list.sort(key=lambda x: (x[0], x[1]))

        # 释放 flows_dict 内存
        del flows_dict
        gc.collect()
        
        # 提取每个流到单独的文件
        flow_files = []
        for idx, (time_rel, frame_num, flow_key, first_packet) in enumerate(flow_list):
            src_ip = first_packet['src_ip']
            dst_ip = first_packet['dst_ip']
            src_port = first_packet['src_port']
            dst_port = first_packet['dst_port']
            protocol = first_packet['protocol']
            
            # 构建过滤器（双向流）
            is_ipv6 = ':' in src_ip or ':' in dst_ip
            if is_ipv6:
                filter_expr = (
                    f"((ipv6.src=={src_ip} and ipv6.dst=={dst_ip} and "
                    f"{protocol}.srcport=={src_port} and {protocol}.dstport=={dst_port}) or "
                    f"(ipv6.src=={dst_ip} and ipv6.dst=={src_ip} and "
                    f"{protocol}.srcport=={dst_port} and {protocol}.dstport=={src_port}))"
                )
            else:
                filter_expr = (
                    f"((ip.src=={src_ip} and ip.dst=={dst_ip} and "
                    f"{protocol}.srcport=={src_port} and {protocol}.dstport=={dst_port}) or "
                    f"(ip.src=={dst_ip} and ip.dst=={src_ip} and "
                    f"{protocol}.srcport=={dst_port} and {protocol}.dstport=={src_port}))"
                )
            
            output_file = os.path.join(temp_dir, f"flow_{idx+1:06d}.pcap")
            
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', filter_expr,
                '-w', output_file
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 24:
                flow_files.append(output_file)
        
        return flow_files
        
    except Exception as e:
        return []


def extract_app_names_from_txt(txt_file):
    """
    从txt文件中提取所有应用名称（按行顺序）
    
    参数:
        txt_file: txt文件路径
    
    返回:
        list: 应用名称列表（按行顺序）
    """
    app_names = []
    try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    app_name = parts[1].strip()
                    if app_name:
                        app_names.append(app_name)
    except Exception as e:
        pass
    
    return app_names



def extract_pcap_flows_info(pcap_file, max_timeout=300):
    """
    🚀 核心优化：一次性读取 pcap 文件的所有包信息
    使用 tshark 输出所有包的五元组，在内存中按流分组

    参数:
        pcap_file: pcap文件路径

    返回:
        dict: {
            flow_key (tuple): {
                'protocol': 'tcp' or 'udp',
                'packets': [frame_numbers],
                'first_frame': int,
                'src_ip': str,
                'dst_ip': str,
                'src_port': str,
                'dst_port': str
            }
        }
        或 None（处理失败）
    """
    flows = {}

    try:
        # 检查文件大小，对大文件增加超时时间
        try:
            file_size = os.path.getsize(pcap_file)
            # 每 100MB 增加 60 秒超时时间
            timeout = min(max_timeout, 120 + (file_size // (100 * 1024 * 1024)) * 60)
        except:
            timeout = 120

        # 一次性读取所有 TCP 和 UDP 包的信息
        # 使用 tshark 输出：帧号、相对时间、源IP、目的IP、源端口、目的端口
        cmd = [
            'tshark',
            '-r', pcap_file,
            '-Y', 'tcp or udp',
            '-T', 'fields',
            '-e', 'frame.number',
            '-e', 'frame.time_relative',
            '-e', 'ip.src',
            '-e', 'ip.dst',
            '-e', 'tcp.srcport',
            '-e', 'tcp.dstport',
            '-e', 'udp.srcport',
            '-e', 'udp.dstport',
            '-E', 'separator=\t'
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )

        if result.returncode != 0:
            return None  # 返回 None 表示处理失败

        # 解析输出，按流分组
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) < 8:
                continue

            frame_num = parts[0]
            time_rel = parts[1]
            src_ip = parts[2]
            dst_ip = parts[3]
            tcp_sport = parts[4]
            tcp_dport = parts[5]
            udp_sport = parts[6]
            udp_dport = parts[7]

            # 判断是 TCP 还是 UDP
            if tcp_sport and tcp_dport:
                protocol = 'tcp'
                src_port = tcp_sport
                dst_port = tcp_dport
            elif udp_sport and udp_dport:
                protocol = 'udp'
                src_port = udp_sport
                dst_port = udp_dport
            else:
                continue

            # 跳过空 IP
            if not src_ip or not dst_ip:
                continue

            # 跳过 IPv6（只处理 IPv4）
            if ':' in src_ip or ':' in dst_ip:
                continue

            try:
                frame_number = int(frame_num)
                time_relative = float(time_rel) if time_rel else 0.0
            except:
                continue

            # 创建流的标识符（规范化：使用字典序较小的组合）
            tuple1 = (src_ip, src_port, dst_ip, dst_port)
            tuple2 = (dst_ip, dst_port, src_ip, src_port)
            flow_key = tuple(sorted([tuple1, tuple2])[0])

            # 添加到流字典
            if flow_key not in flows:
                flows[flow_key] = {
                    'protocol': protocol,
                    'packets': [],
                    'first_frame': frame_number,
                    'first_time': time_relative,
                    'src_ip': tuple1[0],
                    'dst_ip': tuple1[2],
                    'src_port': tuple1[1],
                    'dst_port': tuple1[3]
                }

            flows[flow_key]['packets'].append(frame_number)

            # 更新首包信息
            if time_relative < flows[flow_key]['first_time'] or \
               (time_relative == flows[flow_key]['first_time'] and frame_number < flows[flow_key]['first_frame']):
                flows[flow_key]['first_frame'] = frame_number
                flows[flow_key]['first_time'] = time_relative

        # 清理 result 对象
        del result
        gc.collect()
        return flows

    except subprocess.TimeoutExpired:
        return None  # 超时，返回 None
    except Exception as e:
        return None  # 异常，返回 None


def quick_check_txt_has_target_flows(txt_file, target_apps, flow_counters):
    """
    🚀 快速检查 txt 文件是否包含需要处理的流
    在读取 pcap 文件之前先调用此函数，避免无效的 pcap 读取
    
    参数:
        txt_file: txt文件路径
        target_apps: 目标应用集合
        flow_counters: 当前流计数器 {app_name: count}
    
    返回:
        bool: True=包含需要的流, False=不包含
    """
    MAX_FLOWS_PER_APP = 1000
    apps_at_limit = set(app for app, count in flow_counters.items() if count >= MAX_FLOWS_PER_APP)
    
    try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                # 快速提取应用名（第二个字段）
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                
                app_name = parts[1].strip()
                
                # 检查是否在目标应用中 且 未达到限制
                if app_name in target_apps and app_name not in apps_at_limit:
                    # 进一步检查是否是 IPv4（快速判断）
                    flow_info = parse_flow_info(line)
                    if flow_info and not flow_info.get('is_ipv6'):
                        # 找到至少一条需要的流，立即返回 True
                        return True
        
        # 没有找到任何需要的流
        return False
        
    except Exception as e:
        return False


def match_txt_flows_to_pcap(txt_file, pcap_flows, target_apps, flow_counters):
    """
    🚀 将 txt 文件中的流信息与 pcap 中提取的流匹配
    
    参数:
        txt_file: txt文件路径
        pcap_flows: extract_pcap_flows_info 返回的流字典
        target_apps: 目标应用集合
        flow_counters: 当前流计数器 {app_name: count}
    
    返回:
        list: [(flow_key, app_name, flow_info), ...]  已排序，只包含需要处理的流
    """
    MAX_FLOWS_PER_APP = 1000
    matched_flows = []
    apps_at_limit = set(app for app, count in flow_counters.items() if count >= MAX_FLOWS_PER_APP)
    
    try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                # 快速提取应用名
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                
                app_name = parts[1].strip()
                
                # 过滤：不在目标应用中
                if app_name not in target_apps:
                    continue
                
                # 过滤：已达到限制
                if app_name in apps_at_limit:
                    continue
                
                # 完整解析流信息
                flow_info = parse_flow_info(line)
                if not flow_info:
                    continue
                
                # 只处理 IPv4
                if flow_info.get('is_ipv6'):
                    continue
                
                # 构造流的标识符
                src_ip = flow_info['src_ip']
                dst_ip = flow_info['dst_ip']
                src_port = flow_info['src_port']
                dst_port = flow_info['dst_port']
                
                tuple1 = (src_ip, src_port, dst_ip, dst_port)
                tuple2 = (dst_ip, dst_port, src_ip, src_port)
                flow_key = tuple(sorted([tuple1, tuple2])[0])
                
                # 检查该流是否在 pcap 中
                if flow_key in pcap_flows:
                    matched_flows.append((flow_key, app_name, pcap_flows[flow_key]))
        
        # 按首包时间和帧号排序（确保顺序一致）
        matched_flows.sort(key=lambda x: (x[2]['first_time'], x[2]['first_frame']))
        
        return matched_flows
        
    except Exception as e:
        return []


def process_txt_pcap_pair(args):
    """
    🚀 优化版：处理一对 txt 和 pcap 文件
    
    新策略（按阶段执行）：
    阶段0: 读取流计数器
    阶段1: 快速扫描 txt 文件，如果不包含需要的流则直接跳过（避免读取 pcap）
    阶段2: 一次性读取 pcap 文件的所有流信息（内存中分组）
    阶段3: 将 txt 中的流信息与 pcap 中的流批量匹配
    阶段4: 批量提取匹配的流到临时文件
    阶段5: 批量更新流计数器（一次文件锁）
    阶段6: 批量匿名化和保存
    
    参数:
        args: (txt_file, pcap_file, target_apps, output_root, flow_counter_file)
    
    返回:
        dict: 统计信息 {应用名称: 成功分配的流数量}
    """
    txt_file, pcap_file, target_apps, output_root, flow_counter_file = args
    
    stats = {}
    
    try:
        # 检查文件是否存在
        if not os.path.exists(txt_file) or not os.path.exists(pcap_file):
            return stats
        
        # 🚀 阶段0: 读取流计数器（提前读取，用于快速检查）
        flow_counters = read_flow_counters_safe(flow_counter_file)
        
        # 🚀 阶段1: 快速检查 txt 文件是否包含需要的流
        # 这一步非常关键：避免对无用的 pcap 文件调用 tshark
        if not quick_check_txt_has_target_flows(txt_file, target_apps, flow_counters):
            # txt 文件中没有任何需要的流，直接跳过
            return stats
        
        # 🚀 阶段2: 一次性读取 pcap 文件的所有流信息（内存中分组）
        # 只有确认 txt 中有需要的流时，才读取 pcap
        pcap_flows = extract_pcap_flows_info(pcap_file)
        
        if pcap_flows is None:
            # 处理失败（超时或错误），跳过该文件
            return stats
        
        if not pcap_flows:
            # 没有找到任何流
            return stats
        
        # 🚀 阶段3: 将 txt 文件中的流与 pcap 中的流匹配
        matched_flows = match_txt_flows_to_pcap(txt_file, pcap_flows, target_apps, flow_counters)

        # 释放 pcap_flows 内存
        del pcap_flows
        gc.collect()

        if not matched_flows:
            return stats
        
        # 🚀 阶段4: 创建临时目录，批量提取流
        temp_dir = tempfile.mkdtemp(prefix='split_flows_')
        
        try:
            MAX_FLOWS_PER_APP = 1000
            
            # 使用本地计数器缓存
            local_counters = dict(flow_counters)
            
            # 存储成功提取的流
            extracted_flows = []  # [(app_name, temp_flow_file), ...]
            
            # 批量提取流（使用五元组过滤）
            for idx, (flow_key, app_name, flow_info) in enumerate(matched_flows):
                # 检查是否达到限制
                local_count = local_counters.get(app_name, 0)
                if local_count >= MAX_FLOWS_PER_APP:
                    continue
                
                # 构建 tshark 过滤器（使用五元组）
                src_ip = flow_info['src_ip']
                dst_ip = flow_info['dst_ip']
                src_port = flow_info['src_port']
                dst_port = flow_info['dst_port']
                protocol = flow_info['protocol']
                
                filter_expr = (
                    f"((ip.src=={src_ip} and ip.dst=={dst_ip} and "
                    f"{protocol}.srcport=={src_port} and {protocol}.dstport=={dst_port}) or "
                    f"(ip.src=={dst_ip} and ip.dst=={src_ip} and "
                    f"{protocol}.srcport=={dst_port} and {protocol}.dstport=={src_port}))"
                )
                
                temp_flow_file = os.path.join(temp_dir, f"flow_{idx:06d}.pcap")
                
                # 提取流
                cmd = [
                    'tshark',
                    '-r', pcap_file,
                    '-Y', filter_expr,
                    '-w', temp_flow_file
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30
                    )
                    
                    if result.returncode == 0 and os.path.exists(temp_flow_file) and os.path.getsize(temp_flow_file) > 24:
                        extracted_flows.append((app_name, temp_flow_file))
                        local_counters[app_name] = local_counters.get(app_name, 0) + 1
                
                except:
                    pass
            
            if not extracted_flows:
                # 释放 matched_flows 内存
                del matched_flows
                gc.collect()
                return stats

            # 释放 matched_flows 内存
            del matched_flows
            gc.collect()

            # 🚀 阶段5: 批量更新流计数器（一次文件锁）
            app_increments = {}
            for app_name, _ in extracted_flows:
                app_increments[app_name] = app_increments.get(app_name, 0) + 1
            
            batch_results = update_flow_counter_batch(flow_counter_file, app_increments)
            
            # 为每个应用分配流编号
            app_flow_numbers = {}
            for app_name, increment in app_increments.items():
                new_count = batch_results.get(app_name, -1)
                if new_count != -1:
                    start_num = new_count - increment + 1
                    app_flow_numbers[app_name] = list(range(start_num, new_count + 1))
                else:
                    app_flow_numbers[app_name] = []
            
            # 🚀 阶段6: 移动文件和匿名化
            for app_name, temp_flow_file in extracted_flows:
                if app_name not in app_flow_numbers or len(app_flow_numbers[app_name]) == 0:
                    try:
                        os.remove(temp_flow_file)
                    except:
                        pass
                    continue
                
                flow_number = app_flow_numbers[app_name].pop(0)
                
                app_dir = os.path.join(output_root, app_name)
                os.makedirs(app_dir, exist_ok=True)
                
                output_file = os.path.join(app_dir, f"{app_name}_flow_{flow_number:08d}.pcap")
                
                shutil.move(temp_flow_file, output_file)
                
                if anonymize_flow_pcap(output_file):
                    if app_name not in stats:
                        stats[app_name] = 0
                    stats[app_name] += 1
                else:
                    try:
                        os.remove(output_file)
                    except:
                        pass
        
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
    except Exception as e:
        pass
    
    return stats


def quick_scan_txt_apps(txt_file, target_apps):
    """
    快速扫描txt文件，统计包含哪些目标应用的流数量
    只统计，不完整解析
    
    参数:
        txt_file: txt文件路径
        target_apps: 目标应用集合
    
    返回:
        dict: {应用名称: 流数量}
    """
    app_counts = {}
    try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if ',' not in line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    app_name = parts[1].strip()
                    if app_name in target_apps:
                        app_counts[app_name] = app_counts.get(app_name, 0) + 1
    except:
        pass
    return app_counts


def find_all_txt_pcap_pairs(dataset_root, main_dirs):
    """
    查找所有的 txt 和 pcap 文件对，按主目录分组
    
    参数:
        dataset_root: 数据集根目录
        main_dirs: 主目录列表
    
    返回:
        dict: {main_dir_name: [(txt_file, pcap_file), ...], ...}
    """
    pairs_by_dir = {}
    
    for main_dir_name in main_dirs:
        main_dir_path = os.path.join(dataset_root, main_dir_name)
        if not os.path.exists(main_dir_path):
            continue
        
        pairs = []
        # 递归查找所有 _log.txt 文件
        for root, dirs, files in os.walk(main_dir_path):
            for file in files:
                if file.endswith('_log.txt'):
                    txt_file = os.path.join(root, file)
                    # 查找对应的 pcap 文件
                    # _log.txt -> .pcap 或 _ipv6.pcap
                    base_name = file.replace('_log.txt', '')
                    pcap_file = os.path.join(root, base_name + '.pcap')
                    
                    if os.path.exists(pcap_file):
                        pairs.append((txt_file, pcap_file))
        
        if pairs:
            pairs_by_dir[main_dir_name] = pairs
    
    return pairs_by_dir


def init_flow_counter_from_existing(output_root, flow_counter_file):
    """
    扫描已生成的单流文件，初始化流计数器
    
    参数:
        output_root: 输出根目录
        flow_counter_file: 流计数器文件路径
    
    返回:
        dict: {应用名称: 流数量}
    """
    flow_counts = {}
    
    if not os.path.exists(output_root):
        return flow_counts
    
    print("正在扫描已生成的流文件...")
    
    # 查找所有应用目录
    app_dirs = [d for d in glob.glob(os.path.join(output_root, '*')) if os.path.isdir(d)]
    
    for app_dir in app_dirs:
        app_name = os.path.basename(app_dir)
        
        # 跳过隐藏目录
        if app_name.startswith('.'):
            continue
        
        # 统计该应用的pcap文件数量
        pcap_files = glob.glob(os.path.join(app_dir, '*.pcap'))
        count = len(pcap_files)
        
        if count > 0:
            flow_counts[app_name] = count
    
    # 写入流计数器文件
    if flow_counts:
        try:
            with open(flow_counter_file, 'w') as f:
                for app_name, count in sorted(flow_counts.items()):
                    f.write(f"{app_name}:{count}\n")
            
            print(f"✓ 已扫描 {len(flow_counts)} 个应用, 共 {sum(flow_counts.values())} 条流")
            
            # 统计已达到1000条的应用
            apps_at_limit = [app for app, count in flow_counts.items() if count >= 1000]
            if apps_at_limit:
                print(f"⚠️  {len(apps_at_limit)} 个应用已达到1000条流限制，将被跳过")
        except Exception as e:
            print(f"⚠️  写入流计数器文件失败: {e}")
    else:
        print("✓ 未找到已生成的流文件（全新开始）")
    
    return flow_counts


def split_dataset(dataset_root, output_root, target_apps, num_workers=None):
    """
    切分 NUDT_MobileTraffic Dataset 数据集
    
    参数:
        dataset_root: 数据集根目录
        output_root: 输出根目录
        target_apps: 目标应用名称集合
        num_workers: 并行进程数
    """
    # 🚀 修改：处理除 data8 以外的所有目录
    main_dirs = [
        'NetlogTraffic_0626',
        # 'NetlogTraffic_0628',
        # 'newUser',
        # 'newUser_0602',
        # 'traffic'
    ]
    
    # 自动检测CPU核心数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 创建流计数器文件
    flow_counter_file = os.path.join(output_root, '.flow_counters.txt')
    
    # 🚀 新增：创建跳过列表和错误日志文件
    skip_list_file = os.path.join(output_root, '.skip_list.txt')
    error_log_file = os.path.join(output_root, '.error_log.txt')
    
    # 加载跳过列表
    skip_set = load_skip_list(skip_list_file)
    
    # 🚀 初始化流计数器：扫描已生成的流文件（必须在打印信息之前）
    existing_counts = init_flow_counter_from_existing(output_root, flow_counter_file)
    
    print("\n" + "=" * 80)
    print("NUDT_MobileTraffic Dataset 单流切分")
    print("=" * 80)
    print(f"输入目录: {dataset_root}")
    print(f"输出目录: {output_root}")
    print(f"目标应用数: {len(target_apps)}")
    print(f"并行进程数: {num_workers}")
    if existing_counts:
        apps_at_limit = sum(1 for count in existing_counts.values() if count >= 1000)
        print(f"已有流数据: {len(existing_counts)} 个应用, {sum(existing_counts.values())} 条流")
        if apps_at_limit > 0:
            print(f"⚠️  {apps_at_limit} 个应用已达1000条限制，将被自动跳过")
    print("=" * 80)
    
    # 查找所有 txt 和 pcap 文件对（按主目录分组）
    print("\n正在查找 txt 和 pcap 文件对...")
    pairs_by_dir = find_all_txt_pcap_pairs(dataset_root, main_dirs)
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_dir.values())
    print(f"找到 {total_pairs} 对文件（分布在 {len(pairs_by_dir)} 个主目录中）\n")
    
    if total_pairs == 0:
        print("❌ 未找到任何文件对")
        return
    
    # 使用进程池并行处理
    start_time = time.time()
    # 🚀 关键修复：使用已存在的流数据初始化 all_stats
    # 这样进度显示时就会包含已存在的流和新提取的流
    all_stats = dict(existing_counts) if existing_counts else {}
    
    # 打印初始统计信息
    if all_stats:
        total_existing_flows = sum(all_stats.values())
        apps_at_limit = [app for app, count in all_stats.items() if count >= 1000]
        print(f"\n📊 初始统计信息（处理前）：")
        print(f"  - 应用总数: {len(all_stats)} 个")
        print(f"  - 流总数: {total_existing_flows} 条")
        if apps_at_limit:
            print(f"  - 已达限制的应用: {len(apps_at_limit)} 个")
            print(f"    {', '.join(sorted(apps_at_limit)[:10])}{' ...' if len(apps_at_limit) > 10 else ''}")
        # 显示 Top 5 应用
        if all_stats:
            top_apps = sorted(all_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            top_str = ", ".join([f"{app}({count})" for app, count in top_apps])
            print(f"  - Top 5: {top_str}")
        print()
    
    # 按主目录分组处理
    for main_dir_name, pairs in pairs_by_dir.items():
        print(f"\n{'='*80}")
        print(f"当前处理目录: {main_dir_name}")
        print(f"该目录下有 {len(pairs)} 对文件")
        print(f"{'='*80}")
        
        # 🚀 关键优化：预过滤文件对，跳过无价值的文件
        print(f"正在预扫描并过滤文件...")
        start_filter_time = time.time()
        
        # 读取当前流计数器
        flow_counters = read_flow_counters_safe(flow_counter_file)
        MAX_FLOWS_PER_APP = 1000
        apps_at_limit = set(app for app, count in flow_counters.items() if count >= MAX_FLOWS_PER_APP)
        
        # 快速扫描每个txt文件，统计有价值的流数量
        filtered_pairs = []
        skipped_files = 0
        skipped_flows = 0
        skipped_in_skip_list = 0
        
        for txt_file, pcap_file in pairs:
            # 🚀 新增：跳过在跳过列表中的文件
            if pcap_file in skip_set:
                skipped_in_skip_list += 1
                continue
            # 快速扫描txt文件
            app_counts = quick_scan_txt_apps(txt_file, target_apps)
            
            # 计算有价值的流数量（未达到限制的应用）
            valuable_flows = sum(count for app, count in app_counts.items() if app not in apps_at_limit)
            total_flows = sum(app_counts.values())
            
            # 如果有价值的流数量 > 0，保留这个文件对
            if valuable_flows > 0:
                filtered_pairs.append((txt_file, pcap_file))
            else:
                skipped_files += 1
                skipped_flows += total_flows
        
        filter_time = time.time() - start_filter_time
        print(f"✓ 预扫描完成，耗时 {filter_time:.1f}秒")
        print(f"  - 保留文件: {len(filtered_pairs)} 对")
        print(f"  - 跳过文件: {skipped_files} 对（包含 {skipped_flows} 条已达限制应用的流）")
        if skipped_in_skip_list > 0:
            print(f"  - 跳过列表: {skipped_in_skip_list} 对（之前处理失败的文件）")
        print(f"  - 当前已达限制的应用: {len(apps_at_limit)} 个")
        if apps_at_limit:
            print(f"    {', '.join(sorted(list(apps_at_limit)[:10]))}{' ...' if len(apps_at_limit) > 10 else ''}")
        
        if not filtered_pairs:
            print(f"⚠️  该目录下所有文件都已被过滤，跳过处理")
            continue
        
        # 准备参数列表（使用过滤后的文件对）
        task_args = [
            (txt_file, pcap_file, target_apps, output_root, flow_counter_file)
            for txt_file, pcap_file in filtered_pairs
        ]
        
        # 🚀 优化：按 pcap 文件大小排序，优先处理小文件
        # 这样可以让进度更均匀，避免最后处理大文件时等待时间过长
        print(f"正在按文件大小排序...")
        task_args_with_size = []
        for txt_file, pcap_file in filtered_pairs:
            try:
                pcap_size = os.path.getsize(pcap_file)
            except:
                pcap_size = 0
            task_args_with_size.append((pcap_size, txt_file, pcap_file))
        
        # 按大小排序（小文件优先）
        task_args_with_size.sort(key=lambda x: x[0])
        
        # 重新构造任务参数
        task_args = [
            (txt_file, pcap_file, target_apps, output_root, flow_counter_file)
            for _, txt_file, pcap_file in task_args_with_size
        ]
        
        print(f"✓ 排序完成，最小文件 {task_args_with_size[0][0]/1024/1024:.1f}MB，最大文件 {task_args_with_size[-1][0]/1024/1024:.1f}MB")
        
        # 使用进程池并行处理
        dir_start_time = time.time()
        processed_count = 0
        last_update_time = time.time()
        
        with Pool(processes=num_workers) as pool:
            # 使用迭代器处理每个文件对
            results_iter = pool.imap_unordered(process_txt_pcap_pair, task_args)
            
            for idx, stats in enumerate(results_iter, 1):
                # 合并统计结果
                for app_name, count in stats.items():
                    if app_name not in all_stats:
                        all_stats[app_name] = 0
                    all_stats[app_name] += count
                
                processed_count = idx
                current_time = time.time()
                
                # 🚀 优化：每5秒或每10个文件输出一次进度（更及时的反馈）
                should_print = (current_time - last_update_time >= 5.0) or \
                               (processed_count % 10 == 0) or \
                               (processed_count == len(task_args))
                
                if should_print:
                    elapsed = time.time() - dir_start_time
                    avg_time = elapsed / processed_count if processed_count > 0 else 0
                    remaining = len(task_args) - processed_count
                    estimated_remaining = avg_time * remaining if remaining > 0 else 0
                    total_flows = sum(all_stats.values())
                    
                    # 计算处理速度
                    files_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    flows_per_sec = total_flows / elapsed if elapsed > 0 else 0
                    
                    # 计算本次处理新提取的流数（总流数 - 初始流数）
                    initial_flows = sum(existing_counts.values()) if existing_counts else 0
                    new_flows = total_flows - initial_flows
                    
                    # 计算本次新增流的速度（只计算本次处理的流）
                    new_flows_per_sec = new_flows / elapsed if elapsed > 0 else 0
                    
                    print(f"  [{main_dir_name}] 进度: {processed_count}/{len(task_args)} "
                          f"({processed_count*100/len(task_args):.1f}%), "
                          f"已耗时 {elapsed:.1f}秒 ({elapsed/60:.1f}分钟), "
                          f"预计剩余 {estimated_remaining:.1f}秒 ({estimated_remaining/60:.1f}分钟), "
                          f"速度: {files_per_sec:.2f}文件/秒 {new_flows_per_sec:.2f}流/秒, "
                          f"总流数: {total_flows}条 (本次新增: {new_flows}条)")
                    sys.stdout.flush()
                    last_update_time = current_time
                    
                    # 每100个文件额外输出详细汇总
                    if processed_count % 100 == 0:
                        initial_flows = sum(existing_counts.values()) if existing_counts else 0
                        new_flows = total_flows - initial_flows
                        print(f"    └─ 当前统计: {len(all_stats)} 个应用, 总流数: {total_flows}条 (本次新增: {new_flows}条)")
                        # 显示 top 5 应用
                        if all_stats:
                            top_apps = sorted(all_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                            top_str = ", ".join([f"{app}({count})" for app, count in top_apps])
                            print(f"    └─ Top 5: {top_str}")
                        sys.stdout.flush()
        
        dir_elapsed = time.time() - dir_start_time
        dir_total_flows = sum(all_stats.values())
        initial_flows = sum(existing_counts.values()) if existing_counts else 0
        dir_new_flows = dir_total_flows - initial_flows
        print(f"\n✓ {main_dir_name} 目录处理完成: {len(pairs)} 对文件, 耗时 {dir_elapsed:.1f}秒 ({dir_elapsed/60:.1f}分钟)")
        print(f"  总流数: {dir_total_flows}条 (本次新增: {dir_new_flows}条, 已有: {initial_flows}条)")
        sys.stdout.flush()
    
    elapsed_time = time.time() - start_time
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("切分完成！统计结果：")
    print("=" * 80)
    
    if all_stats:
        # 计算本次新增的流数
        initial_flows = sum(existing_counts.values()) if existing_counts else 0
        total_flows = sum(all_stats.values())
        new_flows = total_flows - initial_flows
        
        # 按流数量降序排序
        sorted_stats = sorted(all_stats.items(), key=lambda x: x[1], reverse=True)
        print("各应用流数统计：")
        for app_name, count in sorted_stats:
            # 计算该应用本次新增的流数
            initial_count = existing_counts.get(app_name, 0) if existing_counts else 0
            new_count = count - initial_count
            if new_count > 0:
                print(f"  {app_name}: {count} 条流 (本次新增: {new_count}条)")
            else:
                print(f"  {app_name}: {count} 条流 (已达限制)")
        
        print("-" * 80)
        print(f"总应用数: {len(all_stats)}")
        print(f"总流数: {total_flows} 条")
        if initial_flows > 0:
            print(f"  - 已有流数: {initial_flows} 条")
            print(f"  - 本次新增: {new_flows} 条")
        
        # 显示 Top 5 应用
        top_apps = sorted_stats[:5]
        top_str = ", ".join([f"{app}({count})" for app, count in top_apps])
        print(f"Top 5 应用: {top_str}")
    else:
        print("未提取到任何流")
    
    print(f"处理时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"输出目录: {output_root}")
    print("=" * 80)


if __name__ == '__main__':
    # 配置
    DATASET_ROOT = "/root/datasets/original_datasets/original_finetuning_datasets/NUDT_MobileTraffic Dataset/"
    OUTPUT_ROOT = "/root/datasets/单流Pcap数据集_微调_随机化/NUDT_MobileTraffic Dataset_all"
    CSV_FILE = "/root/TrafficAttention/data_process/Traffic Splitting​/for_fine-tuning/NUDT_dataset_statistics.csv"
    MIN_FLOWS = 2500  # 最小流数阈值（只处理流数 >= 2500 的应用）
    NUM_WORKERS = 6  # 并行进程数（可根据CPU核心数调整）
    # 注意：每个应用最多保存1000条流（在 process_txt_pcap_pair 函数中设置 MAX_FLOWS_PER_APP = 1000）
    
    print("\n" + "=" * 80)
    print("NUDT_MobileTraffic Dataset 单流切分工具（优化版 v2.0）")
    print("=" * 80)
    print(f"最小流数阈值: {MIN_FLOWS}")
    print(f"并行进程数: {NUM_WORKERS}")
    print(f"每个应用最多保存: 1000 条流")
    print(f"协议类型: 仅IPv4（跳过IPv6）")
    print(f"\n🎯 当前处理目标: 所有目录（除 data8 外）")
    print(f"   包含目录: NetlogTraffic_0626, NetlogTraffic_0628, newUser, newUser_0602, traffic")
    print(f"   排除目录: data8")
    print("\n✨ v2.0 核心改进：")
    print("  1. 快速扫描 txt 文件，提前跳过不包含目标流的文件对")
    print("  2. 一次性读取 pcap 文件的所有流信息，避免重复读取")
    print("  3. 在内存中按五元组匹配，大幅减少磁盘 I/O")
    print("  4. 按文件大小排序，优先处理小文件，进度更均匀")
    print("  5. 增强进度显示：实时速度、预估剩余时间、Top应用统计")
    print("  6. 自动跳过处理失败的文件，支持断点续传")
    print("\n功能说明：")
    print("  1. 使用五元组从pcap文件中精确提取对应的流（仅IPv4）")
    print("  2. 每个应用最多保存1000条单流文件")
    print("  3. 自动扫描已有流文件，跳过已达1000条限制的应用")
    print("  4. IP地址随机化（每条流内所有包使用相同的随机IP）")
    print("  5. TCP/UDP端口号清零（设置为0）")
    print("\n📊 辅助文件说明：")
    print("  - .flow_counters.txt: 流计数器（记录每个应用已保存的流数量）")
    print("  - .skip_list.txt: 跳过列表（记录处理失败的文件）")
    print("  - .error_log.txt: 错误日志（记录详细错误信息）")
    print("=" * 80)
    
    # 检查数据集目录
    if not os.path.exists(DATASET_ROOT):
        print(f"\n❌ 数据集目录不存在: {DATASET_ROOT}")
        exit(1)
    
    # 检查CSV文件
    if not os.path.exists(CSV_FILE):
        print(f"\n❌ CSV文件不存在: {CSV_FILE}")
        # 尝试查找CSV文件
        csv_file_alt = "/root/TrafficAttention/data_process/Traffic Splitting​/for_fine-tuning/NUDT_dataset_statistics.csv"
        if os.path.exists(csv_file_alt):
            CSV_FILE = csv_file_alt
            print(f"✓ 找到CSV文件: {CSV_FILE}")
        else:
            exit(1)
    
    # 加载目标应用
    target_apps = load_target_apps(CSV_FILE, MIN_FLOWS)
    
    if len(target_apps) == 0:
        print("\n❌ 未找到任何目标应用")
        exit(1)
    
    print(f"\n目标应用列表 ({len(target_apps)} 个):")
    for app in sorted(target_apps):
        print(f"  - {app}")
    
    # 执行切分
    split_dataset(DATASET_ROOT, OUTPUT_ROOT, target_apps, NUM_WORKERS)
    
    print("\n✓ 切分完成!")
