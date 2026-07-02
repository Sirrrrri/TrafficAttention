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
import scapy.all as scapy


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
            return False
        
        # 保存修改后的包
        scapy.wrpcap(pcap_path, modified_packets)
        return True
        
    except Exception as e:
        # 处理失败，返回False
        return False

def split_pcap_with_splitcap_wrapper(args):
    """
    多进程包装函数
    """
    pcap_path, output_dir, splitcap_exe = args
    return split_pcap_with_splitcap(pcap_path, output_dir, splitcap_exe)


def split_pcap_with_splitcap(pcap_path, output_dir, splitcap_exe):
    """
    使用 SplitCap.exe (通过 Mono) 将一个pcap文件按流拆分
    
    参数:
        pcap_path: 输入pcap文件路径
        output_dir: 输出目录
        splitcap_exe: SplitCap.exe 的路径
    
    返回:
        flows_count: 拆分出的流数量
    """
    temp_output = None
    debug_log = os.path.join(output_dir, 'splitcap_debug.log')
    
    try:
        # 创建临时工作目录（SplitCap会在工作目录中生成文件）
        base_name = os.path.splitext(os.path.basename(pcap_path))[0]
        temp_output = os.path.join(output_dir, f'temp_{base_name}')
        os.makedirs(temp_output, exist_ok=True)
        
        # 将pcap文件复制到临时目录（SplitCap.exe和pcap要在同一目录下）
        temp_pcap = os.path.join(temp_output, os.path.basename(pcap_path))
        shutil.copy2(pcap_path, temp_pcap)
        
        # 将SplitCap.exe也复制到临时目录（确保在同一目录）
        temp_splitcap = os.path.join(temp_output, 'SplitCap.exe')
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
            
            # 如果是目录，可能是SplitCap创建的输出目录，进入查找文件
            if os.path.isdir(full_path):
                # 在子目录中查找所有文件（包括没有.pcap扩展名的）
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if os.path.isfile(item_path):
                        flow_files.append(item_path)
            # 如果是文件，直接添加（可能是没有扩展名的pcap文件）
            elif os.path.isfile(full_path):
                flow_files.append(full_path)
        
        flows_count = len(flow_files)
        
        # 记录文件统计
        with open(debug_log, 'a') as f:
            f.write(f"All files: {all_files}\n")
            f.write(f"Flow files found: {flows_count}\n")
            if flows_count > 0:
                f.write(f"Flow file list: {[os.path.basename(ff) for ff in flow_files[:5]]}\n")
        
        if flows_count == 0:
            shutil.rmtree(temp_output, ignore_errors=True)
            return 0
        
        # 将拆分出的流文件移动到最终输出目录并进行匿名化
        for idx, flow_file in enumerate(flow_files, 1):
            # 生成新的流文件名（添加.pcap扩展名）
            flow_basename = os.path.basename(flow_file)
            # 如果文件没有.pcap扩展名，添加上
            if not flow_basename.endswith('.pcap'):
                new_filename = f"{base_name}_flow{idx:04d}_{flow_basename}.pcap"
            else:
                new_filename = f"{base_name}_flow{idx:04d}_{flow_basename}"
            new_path = os.path.join(output_dir, new_filename)
            
            # 移动文件
            shutil.move(flow_file, new_path)
            
            # 对单流pcap文件进行IP地址随机化和端口清零
            anonymize_flow_pcap(new_path)
        
        # 清理临时目录
        shutil.rmtree(temp_output, ignore_errors=True)
        
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


def process_dataset_directory(dataset_path, output_root, num_workers=None):
    """
    处理整个数据集目录（使用多进程 + splitcap）
    
    参数:
        dataset_path: 数据集目录（包含多个类别子目录）
        output_root: 输出根目录
        num_workers: 并行进程数（None=自动检测）
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
    print(f"并行进程数: {num_workers}\n")
    
    # 查找所有类别目录
    class_dirs = [d for d in glob.glob(os.path.join(dataset_path, '*')) if os.path.isdir(d)]
    
    if len(class_dirs) == 0:
        print(f"❌ 未找到类别目录: {dataset_path}")
        return
    
    print(f"找到 {len(class_dirs)} 个类别目录\n")
    
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
        
        if len(pcap_files) == 0:
            pcap_files = glob.glob(os.path.join(class_dir, '**/*.pcap'), recursive=True)
        
        if len(pcap_files) == 0:
            print(f"  ⚠️  未找到pcap文件")
            continue
        
        print(f"  找到 {len(pcap_files)} 个pcap文件，使用 {num_workers} 个进程并行处理...")
        
        # 创建输出目录
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


def process_all_datasets(datasets_root, output_root, dataset_names, num_workers=None):
    """
    批量处理多个数据集（使用多进程 + SplitCap.exe）
    
    参数:
        datasets_root: 数据集根目录
        output_root: 输出根目录
        dataset_names: 数据集名称列表
        num_workers: 并行进程数（None=自动检测）
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
    
    print("\n" + "🌊" * 35)
    print("批量流拆分工具（SplitCap.exe + Mono 多进程版本）")
    print("🌊" * 35)
    print(f"\n数据集根目录: {datasets_root}")
    print(f"输出根目录: {output_root}")
    print(f"待处理数据集: {', '.join(dataset_names)}")
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
                num_workers=num_workers
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
        from config import DATASETS_ROOT, DATASETS
        print("✓ 从config.py加载配置")
    except ImportError:
        print("⚠️  config.py未找到，使用默认配置")
        DATASETS_ROOT = "/root/datasets_new/original_pcap_datasets/"
        DATASETS = ["CICIoT2023", "ISCXVPN2016", "USTC-TFC2016"]
    
    # 输出目录（拆分后的单流pcap文件）
    OUTPUT_ROOT = "/root/datasets_new/单流Pcap数据集"
    
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
    print("\n功能说明：")
    print("  1. 使用 SplitCap.exe 按会话拆分pcap文件为单流文件")
    print("  2. 对每条流进行IP地址随机化（每条流内所有包使用相同的随机IP）")
    print("  3. 将TCP/UDP端口号清零（设置为0）")
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
        num_workers=NUM_WORKERS
    )
    
    print("\n💡 后续步骤:")
    print(f"  1. 查看拆分结果: {OUTPUT_ROOT}")
    print(f"  2. 使用拆分后的单流pcap文件进行MFR矩阵生成")
    print(f"  3. 更新data_process.py中的输入路径为: {OUTPUT_ROOT}")
    print("")

