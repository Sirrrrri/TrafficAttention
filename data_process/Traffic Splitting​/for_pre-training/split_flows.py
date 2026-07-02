#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
使用 splitcap 工具将pcap文件按流拆分
每个pcap文件可能包含多条流，使用 splitcap 按会话(session)分组
将每条流保存为单独的pcap文件

功能：
1. 使用 SplitCap.exe 按会话拆分pcap文件
2. 对每条单流pcap文件进行IP地址随机化（每条流内所有包使用相同的随机IP）
3. 将端口号清零（TCP/UDP端口设置为0）
"""

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
import struct
import socket
import dpkt


def check_splitcap_installed():
    """
    检查 Mono 和 SplitCap.exe 是否已安装
    """
    # 检查 mono 是否安装
    mono_path = shutil.which('mono')
    if mono_path is None:
        print("❌ 错误：未找到 mono")
        print("\n安装方法：")
        print("  Ubuntu/Debian: sudo apt-get install mono-complete")
        print("  CentOS/RHEL: sudo yum install mono-complete")
        return False
    
    # 检查 SplitCap.exe 是否存在
    splitcap_paths = [
        '/usr/local/bin/SplitCap.exe',
        '/usr/bin/SplitCap.exe',
        './SplitCap.exe',
        os.path.join(os.path.dirname(__file__), 'SplitCap.exe')
    ]
    
    splitcap_exe = None
    for path in splitcap_paths:
        if os.path.exists(path):
            splitcap_exe = path
            break
    
    if splitcap_exe is None:
        print("❌ 错误：未找到 SplitCap.exe")
        print("\n请将 SplitCap.exe 放置在以下任一位置：")
        for path in splitcap_paths:
            print(f"  - {path}")
        print("\n下载地址: https://github.com/clay584/SplitCap")
        return False
    
    print(f"✓ 找到 SplitCap.exe: {splitcap_exe}")
    return splitcap_exe


# 全局变量存储 SplitCap.exe 路径
SPLITCAP_EXE_PATH = None



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
            # 读取pcap文件头
            try:
                pcap_reader = dpkt.pcap.Reader(f_in)
            except Exception as e:
                # pcap文件格式错误
                error_msg = f"Failed to read pcap header: {str(e)}"
                if debug_log:
                    with open(debug_log, 'a') as f:
                        f.write(f"  ERROR [{os.path.basename(pcap_path)}]: {error_msg}\n")
                return False

            # 读取第一个包以确定流参数
            first_packet_data = None
            first_timestamp = None
            first_src_ip = None
            first_dst_ip = None
            has_ether = True
            ip_start = 0

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
                # 先尝试解析为以太网帧
                try:
                    eth = dpkt.ethernet.Ethernet(first_packet_data)
                    if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                        # 有以太网头且是IPv4
                        ip = eth.ip
                        has_ether = True
                        ip_start = 14
                    else:
                        # 有以太网头但不是IPv4，尝试直接解析IP（可能是没有以太网头的pcap）
                        has_ether = False
                        ip = dpkt.ip.IP(first_packet_data)
                        ip_start = 0
                except:
                    # 解析以太网帧失败，尝试直接解析IP（可能是没有以太网头的pcap）
                    has_ether = False
                    ip = dpkt.ip.IP(first_packet_data)
                    ip_start = 0

                if ip.v != 4:  # 不是IPv4
                    error_msg = f"Not IPv4 (version={ip.v})"
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

            if first_src_ip is None or first_dst_ip is None:
                error_msg = "Failed to extract IP addresses from first packet"
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

            # 创建pcap写入器
            pcap_writer = dpkt.pcap.Writer(temp_file)

            # 处理第一个包
            modified_data = modify_packet_binary(
                first_packet_data, has_ether, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
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
                        packet_data, has_ether, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
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


def modify_packet_binary(packet_data, has_ether, ip_start, first_src_ip_bytes, first_dst_ip_bytes,
                         random_src_ip_bytes, random_dst_ip_bytes):
    """
    直接修改包的二进制数据：随机化IP地址和清零端口

    参数:
        packet_data: 包的原始二进制数据
        has_ether: 是否有以太网头
        ip_start: IP头起始位置（已预先计算）
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



def split_pcap_with_splitcap_wrapper(args):
    """
    多进程包装函数
    """
    pcap_path, output_dir, splitcap_exe = args
    return split_pcap_with_splitcap(pcap_path, output_dir, splitcap_exe)


def split_pcap_with_splitcap(pcap_path, output_dir, splitcap_exe):
    """
    使用 SplitCap.exe (通过 Mono) 将一个pcap文件按流拆分（优化版：减少I/O操作）
    
    I/O优化策略：
    1. 使用符号链接替代复制SplitCap.exe（减少一次文件复制）
    2. 在临时目录中直接处理流文件，避免先移动再处理
    3. 处理完成后再一次性移动到输出目录，减少文件移动次数
    
    参数:
        pcap_path: 输入pcap文件路径
        output_dir: 输出目录
        splitcap_exe: SplitCap.exe 的路径
    
    返回:
        flows_count: 拆分出的流数量
    """
    temp_output = None
    # debug日志文件生成在脚本同目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_log = os.path.join(script_dir, 'splitcap_debug.log')
    
    try:
        # 创建临时工作目录（SplitCap会在工作目录中生成文件）
        base_name = os.path.splitext(os.path.basename(pcap_path))[0]
        temp_output = os.path.join(output_dir, f'temp_{base_name}')
        os.makedirs(temp_output, exist_ok=True)
        
        # 将pcap文件复制到临时目录（SplitCap.exe和pcap要在同一目录下）
        temp_pcap = os.path.join(temp_output, os.path.basename(pcap_path))
        shutil.copy2(pcap_path, temp_pcap)
        
        # 使用符号链接替代复制SplitCap.exe（减少I/O操作）
        # 如果符号链接失败，回退到复制
        temp_splitcap = os.path.join(temp_output, 'SplitCap.exe')
        try:
            if os.path.exists(temp_splitcap):
                os.remove(temp_splitcap)
            os.symlink(os.path.abspath(splitcap_exe), temp_splitcap)
        except (OSError, AttributeError):
            # 如果符号链接失败（如Windows或不支持），使用复制
            shutil.copy2(splitcap_exe, temp_splitcap)
        
        # 构建 SplitCap 命令
        # mono SplitCap.exe -r <input.pcap> -s session
        # -r: 输入文件（相对路径）
        # -s: 按会话(session/flow)拆分
        # 注意：根据图片说明，不使用-o参数，SplitCap会在当前目录生成文件
        cmd = [
            'mono',
            'SplitCap.exe',
            '-r', os.path.basename(pcap_path),
            '-s', 'session'
        ]
        
        # 执行 splitcap（在临时目录中执行）
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=temp_output  # 在临时目录中执行命令
        )
        
        # 记录调试信息（只记录前几个文件的详细信息）
        with open(debug_log, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"File: {pcap_path}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working dir: {temp_output}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
            
            # 列出临时目录中的所有文件
            temp_files = os.listdir(temp_output)
            f.write(f"Files in temp dir: {temp_files}\n")
        
        if result.returncode != 0:
            # splitcap 失败，清理临时目录
            shutil.rmtree(temp_output, ignore_errors=True)
            return 0
        
        # 统计生成的流文件
        # SplitCap可能会创建一个子目录来存放拆分的流文件
        all_files = os.listdir(temp_output)
        
        # 查找流文件
        flow_files = []
        for f in all_files:
            full_path = os.path.join(temp_output, f)
            
            # 排除输入文件和SplitCap.exe
            if f == os.path.basename(pcap_path) or f == 'SplitCap.exe':
                continue
            
            # 排除以tmp开头的临时文件（SplitCap可能生成的空文件）
            if f.startswith('tmp') and f.endswith('.pcap'):
                with open(debug_log, 'a') as log_f:
                    log_f.write(f"  Skipping temporary file: {f}\n")
                continue
            
            # 如果是目录，可能是SplitCap创建的输出目录，进入查找文件
            if os.path.isdir(full_path):
                # 在子目录中查找所有文件（包括没有.pcap扩展名的）
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if os.path.isfile(item_path):
                        # 排除以tmp开头的临时文件
                        if item.startswith('tmp') and (item.endswith('.pcap') or '.pcap' in item):
                            with open(debug_log, 'a') as log_f:
                                log_f.write(f"  Skipping temporary file: {item}\n")
                            continue
                        # 检查文件大小，过滤掉空文件（pcap文件头至少24字节）
                        try:
                            if os.path.getsize(item_path) < 24:
                                with open(debug_log, 'a') as log_f:
                                    log_f.write(f"  Skipping empty/small file: {item} (size={os.path.getsize(item_path)} bytes)\n")
                                continue
                        except:
                            pass
                        flow_files.append(item_path)
            # 如果是文件，直接添加（可能是没有扩展名的pcap文件）
            elif os.path.isfile(full_path):
                # 检查文件大小，过滤掉空文件
                try:
                    if os.path.getsize(full_path) < 24:
                        with open(debug_log, 'a') as log_f:
                            log_f.write(f"  Skipping empty/small file: {f} (size={os.path.getsize(full_path)} bytes)\n")
                        continue
                except:
                    pass
                flow_files.append(full_path)
        
        flows_count = len(flow_files)
        
        # 记录文件统计
        with open(debug_log, 'a') as f:
            f.write(f"All files: {all_files}\n")
            f.write(f"Flow files found: {flows_count}\n")
            if flows_count > 0:
                f.write(f"Flow file list: {[os.path.basename(ff) for ff in flow_files[:5]]}\n")
        
        # 记录处理前的统计
        with open(debug_log, 'a') as f:
            f.write(f"Processing {flows_count} flow files for anonymization...\n")
        
        if flows_count == 0:
            shutil.rmtree(temp_output, ignore_errors=True)
            return 0
        
        # 优化：在临时目录中直接处理流文件，避免先移动再处理
        # 处理完成后再一次性移动到输出目录，减少I/O操作
        processed_flow_files = []
        failed_count = 0
        failed_files = []
        
        with open(debug_log, 'a') as f:
            f.write(f"Starting anonymization of {flows_count} flow files...\n")
        
        for idx, flow_file in enumerate(flow_files, 1):
            # 在临时目录中直接进行匿名化处理（避免额外的文件移动）
            if not anonymize_flow_pcap(flow_file, debug_log=debug_log):
                # 如果处理失败，记录到日志
                failed_count += 1
                failed_files.append(os.path.basename(flow_file))
                # 每100个失败记录一次进度
                if failed_count % 100 == 0:
                    with open(debug_log, 'a') as f:
                        f.write(f"  Progress: {idx}/{flows_count} processed, {failed_count} failed\n")
                continue
            
            # 生成新的流文件名（添加.pcap扩展名）
            flow_basename = os.path.basename(flow_file)
            # 如果文件没有.pcap扩展名，添加上
            if not flow_basename.endswith('.pcap'):
                new_filename = f"{base_name}_flow{idx:04d}_{flow_basename}.pcap"
            else:
                new_filename = f"{base_name}_flow{idx:04d}_{flow_basename}"
            new_path = os.path.join(output_dir, new_filename)
            
            # 处理完成后再移动到最终输出目录（一次性移动，减少I/O）
            shutil.move(flow_file, new_path)
            processed_flow_files.append(new_path)

            # 每处理10条流强制垃圾回收一次，防止内存累积
            if idx % 10 == 0:
                gc.collect()
        
        # 更新流数量（只统计成功处理的）
        flows_count = len(processed_flow_files)
        
        # 记录处理结果
        with open(debug_log, 'a') as f:
            f.write(f"\nAnonymization Summary for {base_name}:\n")
            f.write(f"  Total flow files: {len(flow_files)}\n")
            f.write(f"  Successfully processed: {flows_count}\n")
            f.write(f"  Failed: {failed_count}\n")
            f.write(f"  Success rate: {flows_count/len(flow_files)*100:.2f}%\n")
            if failed_count > 0 and failed_count <= 20:
                f.write(f"  Failed files (first 20): {failed_files[:20]}\n")
            elif failed_count > 20:
                f.write(f"  Failed files (first 20): {failed_files[:20]}\n")
                f.write(f"  ... and {failed_count - 20} more failed files\n")
            f.write("\n")
        
        # 清理临时目录
        shutil.rmtree(temp_output, ignore_errors=True)
        
        # 强制垃圾回收，释放内存
        gc.collect()
        
        return flows_count
        
    except subprocess.TimeoutExpired:
        # 超时，清理
        with open(debug_log, 'a') as f:
            f.write(f"\nTIMEOUT: {pcap_path}\n")
        if temp_output and os.path.exists(temp_output):
            shutil.rmtree(temp_output, ignore_errors=True)
        return 0
    except Exception as e:
        # 其他错误，清理
        with open(debug_log, 'a') as f:
            f.write(f"\nERROR: {pcap_path}: {str(e)}\n")
        if temp_output and os.path.exists(temp_output):
            shutil.rmtree(temp_output, ignore_errors=True)
        return 0


def process_dataset_directory(dataset_path, output_root, num_workers=None, processing_mode='auto'):
    """
    处理整个数据集目录（使用多进程 + splitcap）
    
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
            print(f"❌ 模式2: 未找到pcap文件: {dataset_path}")
            return
        print(f"使用模式2: 数据集目录直接包含pcap文件，作为单个类别处理")
        print(f"找到 {len(direct_pcap_files)} 个pcap文件\n")
        class_dirs = [dataset_path]
        is_mode2 = True
    elif processing_mode == 'mode1':
        # 手动选择模式1: 数据集目录包含多个类别子目录
        if len(class_dirs) == 0:
            print(f"❌ 模式1: 未找到类别子目录: {dataset_path}")
            return
        print(f"使用模式1: 数据集目录包含 {len(class_dirs)} 个类别子目录\n")
        is_mode2 = False
    else:
        # 自动模式: 自动检测
        if len(class_dirs) == 0:
            # 没有子目录，检查是否有直接pcap文件
            if len(direct_pcap_files) == 0:
                print(f"❌ 未找到类别目录或pcap文件: {dataset_path}")
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
        
        # 跳过临时目录（模式1中已经过滤，这里再次检查确保安全）
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
            print(f"  ⚠️  未找到pcap文件")
            continue
        
        print(f"  找到 {len(pcap_files)} 个pcap文件，使用 {num_workers} 个进程并行处理...")
        
        # 创建输出目录
        # 模式2: 数据集目录直接包含pcap文件，输出到 output_root/dataset_name/
        # 模式1: 数据集目录包含类别子目录，输出到 output_root/dataset_name/class_name/
        if is_mode2:
            output_class_dir = os.path.join(output_root, dataset_name)
        else:
            output_class_dir = os.path.join(output_root, dataset_name, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # 准备参数列表（用于多进程）
        task_args = [
            (pcap_file, output_class_dir, SPLITCAP_EXE_PATH)
            for pcap_file in pcap_files
        ]
        
        # 使用进程池并行处理
        class_flows = 0
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered提高效率，配合tqdm显示进度
            results = list(tqdm(
                pool.imap_unordered(split_pcap_with_splitcap_wrapper, task_args),
                total=len(task_args),
                desc=f"  {class_name}",
                leave=False
            ))
            class_flows = sum(results)
        
        total_pcap_files += len(pcap_files)
        total_flows += class_flows

        print(f"  ✓ 从 {len(pcap_files)} 个文件中提取了 {class_flows} 条流")
        gc.collect()
    
    # 汇总统计
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ {dataset_name} 处理完成!")
    print(f"{'='*70}")
    print(f"原始pcap文件: {total_pcap_files}")
    print(f"提取的流: {total_flows}")
    print(f"平均每个文件: {total_flows/total_pcap_files:.1f} 条流" if total_pcap_files > 0 else "")
    print(f"处理时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"输出位置: {output_root}/{dataset_name}/")


def process_all_datasets(datasets_root, output_root, dataset_names, num_workers=None, processing_mode='auto'):
    """
    批量处理多个数据集（使用多进程 + SplitCap.exe）
    
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
    global SPLITCAP_EXE_PATH
    
    # 检查 SplitCap 是否安装
    SPLITCAP_EXE_PATH = check_splitcap_installed()
    if not SPLITCAP_EXE_PATH:
        print("\n❌ 终止执行")
        return
    
    # 自动检测CPU核心数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 4)  # 保留4个核心给系统
    
    # 验证处理模式
    valid_modes = ['mode1', 'mode2', 'auto']
    if processing_mode not in valid_modes:
        print(f"❌ 无效的处理模式: {processing_mode}")
        print(f"   有效模式: {', '.join(valid_modes)}")
        return
    
    print("\n" + "🌊" * 35)
    print("批量流拆分工具（SplitCap.exe + Mono 多进程版本）")
    print("🌊" * 35)
    print(f"\n数据集根目录: {datasets_root}")
    print(f"输出根目录: {output_root}")
    print(f"待处理数据集: {', '.join(dataset_names)}")
    print(f"处理模式: {processing_mode}")
    print(f"CPU核心数: {cpu_count()}")
    print(f"使用进程数: {num_workers}")
    print(f"工具: Mono + SplitCap.exe")
    print(f"SplitCap路径: {SPLITCAP_EXE_PATH}")
    
    all_stats = {}
    total_start_time = time.time()
    
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"\n❌ 数据集不存在: {dataset_path}")
            continue
        
        try:
            process_dataset_directory(
                dataset_path=dataset_path,
                output_root=output_root,
                num_workers=num_workers,
                processing_mode=processing_mode
            )
        except Exception as e:
            print(f"\n❌ 处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 70)
    print("✓ 所有数据集处理完成!")
    print("=" * 70)
    print(f"总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"平均速度: 使用 {num_workers} 进程 + Mono/SplitCap.exe 并行处理")


if __name__ == '__main__':
    # 配置
    try:
        from config import DATASETS_ROOT, DATASETS, PROCESSING_MODE
        print("✓ 从config.py加载配置")
    except ImportError:
        print("⚠️  config.py未找到，使用默认配置")
        # DATASETS_ROOT可以选择两种模式，见mode_descriptions
        DATASETS_ROOT = "/root/datasets_new/original_pcap_datasets-for_pre-training/"
        DATASETS = ["NonVPN", "NonTor", "BenignTraffic", "CICIoT2022"]
        PROCESSING_MODE = None  # 如果未定义，使用下面的默认值
    
    # 处理模式配置
    # 'mode1': 数据集目录包含多个类别子目录（每个子目录包含pcap文件）
    # 'mode2': 数据集目录直接包含pcap文件（作为单个类别处理）
    # 'auto': 自动检测模式（推荐）
    if PROCESSING_MODE is None:
        PROCESSING_MODE = 'auto'  # 默认使用模式2（适合当前场景）
    
    # 输出目录（拆分后的单流pcap文件）
    OUTPUT_ROOT = "/root/datasets_new/单流Pcap数据集_预训练_随机化/"
    
    # 流拆分参数
    NUM_WORKERS = 6        # 并行进程数（手动设置为6个进程）
    
    # 如果想使用更多进程，可以修改为：
    # NUM_WORKERS = 40   # 使用40个进程
    # NUM_WORKERS = None # 自动使用 CPU核心数-4 (64核会使用60个进程)
    
    print("\n" + "=" * 70)
    print("流拆分配置")
    print("=" * 70)
    print(f"工具: Mono + SplitCap.exe (按session拆分)")
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
    print("  1. 使用 SplitCap.exe 按会话拆分pcap文件为单流文件")
    print("  2. 对每条流进行IP地址随机化（每条流内所有包使用相同的随机IP）")
    print("  3. 将TCP/UDP端口号清零（设置为0）")
    print("\n处理模式说明：")
    print("  - mode1: 数据集目录下包含多个类别子目录，每个子目录是一个类别")
    print("           输出路径: output_root/dataset_name/class_name/")
    print("  - mode2: 数据集目录下直接包含pcap文件，作为单个类别处理")
    print("           输出路径: output_root/dataset_name/")
    print("  - auto:  自动检测目录结构，选择合适的模式")
    print("\n安装要求：")
    print("  1. 安装 Mono: sudo apt-get install mono-complete")
    print("  2. 下载 SplitCap.exe: https://github.com/clay584/SplitCap")
    print("  3. 将 SplitCap.exe 放置在脚本同目录或 /usr/local/bin/")
    print("  4. 安装 scapy: pip install scapy")
    print("=" * 70)
    
    # 执行批量处理
    process_all_datasets(
        datasets_root=DATASETS_ROOT,
        output_root=OUTPUT_ROOT,
        dataset_names=DATASETS,
        num_workers=NUM_WORKERS,
        processing_mode=PROCESSING_MODE
    )
    
    print("\n💡 后续步骤:")
    print(f"  1. 查看拆分结果: {OUTPUT_ROOT}")
    print(f"  2. 使用拆分后的单流pcap文件进行MFR矩阵生成")
    print(f"  3. 更新data_process.py中的输入路径为: {OUTPUT_ROOT}")
    print("")