#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
扫描已生成的单流文件，统计每个应用的流数量，初始化流计数器文件
"""

import os
import glob

def count_existing_flows(output_root):
    """
    扫描输出目录，统计每个应用已有的流数量
    
    参数:
        output_root: 输出根目录
    
    返回:
        dict: {应用名称: 流数量}
    """
    flow_counts = {}
    
    if not os.path.exists(output_root):
        print(f"❌ 输出目录不存在: {output_root}")
        return flow_counts
    
    print(f"正在扫描输出目录: {output_root}")
    
    # 查找所有应用目录
    app_dirs = [d for d in glob.glob(os.path.join(output_root, '*')) if os.path.isdir(d)]
    
    for app_dir in app_dirs:
        app_name = os.path.basename(app_dir)
        
        # 统计该应用的pcap文件数量
        pcap_files = glob.glob(os.path.join(app_dir, '*.pcap'))
        count = len(pcap_files)
        
        if count > 0:
            flow_counts[app_name] = count
            print(f"  {app_name}: {count} 条流")
    
    return flow_counts


def write_flow_counter_file(flow_counter_file, flow_counts):
    """
    将流计数写入计数器文件
    
    参数:
        flow_counter_file: 流计数器文件路径
        flow_counts: {应用名称: 流数量}
    """
    try:
        with open(flow_counter_file, 'w') as f:
            for app_name, count in sorted(flow_counts.items()):
                f.write(f"{app_name}:{count}\n")
        
        print(f"\n✓ 流计数器文件已更新: {flow_counter_file}")
        print(f"  总计 {len(flow_counts)} 个应用")
        print(f"  总计 {sum(flow_counts.values())} 条流")
        
        # 统计已达到1000条的应用
        apps_at_limit = [app for app, count in flow_counts.items() if count >= 1000]
        if apps_at_limit:
            print(f"\n⚠️  以下 {len(apps_at_limit)} 个应用已达到1000条流限制：")
            for app in sorted(apps_at_limit):
                print(f"    - {app}: {flow_counts[app]} 条")
        
    except Exception as e:
        print(f"❌ 写入流计数器文件失败: {e}")


if __name__ == '__main__':
    OUTPUT_ROOT = "/root/datasets_new/单流Pcap数据集/NUDT_MobileTraffic Dataset_all"
    FLOW_COUNTER_FILE = os.path.join(OUTPUT_ROOT, '.flow_counters.txt')
    
    print("=" * 70)
    print("初始化流计数器 - 从已有文件统计")
    print("=" * 70)
    
    # 统计已有的流
    flow_counts = count_existing_flows(OUTPUT_ROOT)
    
    if not flow_counts:
        print("\n⚠️  未找到任何已生成的流文件")
    else:
        # 写入流计数器文件
        write_flow_counter_file(FLOW_COUNTER_FILE, flow_counts)
    
    print("\n✓ 完成!")
