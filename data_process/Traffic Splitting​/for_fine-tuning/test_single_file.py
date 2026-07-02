#!/usr/bin/python3
#-*- coding:utf-8 -*-

"""
测试单个txt和pcap文件对的流提取
"""

import os
import sys

# 导入主脚本的函数
from split_flows_Mobile import parse_flow_info, extract_flow_from_pcap, load_target_apps

# 测试文件
TEST_TXT = "/root/datasets_new/original_pcap_datasets/NUDT_MobileTraffic Dataset/NetlogTraffic_0626/83/20200626/netlog20200626_23_08_24_log.txt"
TEST_PCAP = "/root/datasets_new/original_pcap_datasets/NUDT_MobileTraffic Dataset/NetlogTraffic_0626/83/20200626/netlog20200626_23_08_24.pcap"
OUTPUT_DIR = "/tmp/test_flows"
CSV_FILE = "/root/我的代码和数据集/Mine/data_process/Traffic Splitting​/NUDT_dataset_statistics.csv"

print("=" * 80)
print("单文件流提取测试")
print("=" * 80)

# 加载目标应用
target_apps = load_target_apps(CSV_FILE, min_flows=2500)
print(f"\n加载了 {len(target_apps)} 个目标应用")

# 读取txt文件
print(f"\n读取测试文件: {TEST_TXT}")
if not os.path.exists(TEST_TXT):
    print(f"❌ 文件不存在: {TEST_TXT}")
    sys.exit(1)

with open(TEST_TXT, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print(f"总行数: {len(lines)}")
print(f"\n前5行内容:")
for i, line in enumerate(lines[:5]):
    print(f"  {i+1}. {line.strip()}")

# 解析流信息
print(f"\n解析流信息...")
flows_by_app = {}
parse_errors = 0

for line in lines:
    flow_info = parse_flow_info(line)
    if flow_info:
        app_name = flow_info['app_name']
        if app_name in target_apps:
            if app_name not in flows_by_app:
                flows_by_app[app_name] = []
            flows_by_app[app_name].append(flow_info)
    else:
        parse_errors += 1

print(f"解析成功: {sum(len(v) for v in flows_by_app.values())} 条流")
print(f"解析失败: {parse_errors} 条")
print(f"目标应用数: {len(flows_by_app)}")

# 显示每个应用的流数量
print(f"\n各应用流数量:")
for app_name, flows in sorted(flows_by_app.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {app_name}: {len(flows)} 条")

# 测试提取第一条流
print(f"\n测试提取流...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_count = 0
success_count = 0
fail_count = 0

for app_name, flows in flows_by_app.items():
    if test_count >= 5:  # 只测试前5个应用
        break
    
    # 测试该应用的第一条流
    flow_info = flows[0]
    output_file = os.path.join(OUTPUT_DIR, f"{app_name}_test.pcap")
    
    print(f"\n测试提取: {app_name}")
    print(f"  协议: {flow_info['protocol']}")
    print(f"  源: {flow_info['src_ip']}:{flow_info['src_port']}")
    print(f"  目标: {flow_info['dst_ip']}:{flow_info['dst_port']}")
    print(f"  IPv6: {flow_info['is_ipv6']}")
    
    success = extract_flow_from_pcap(TEST_PCAP, flow_info, output_file)
    
    if success:
        file_size = os.path.getsize(output_file)
        print(f"  ✓ 提取成功! 文件大小: {file_size} 字节")
        success_count += 1
    else:
        print(f"  ✗ 提取失败")
        fail_count += 1
    
    test_count += 1

print(f"\n" + "=" * 80)
print(f"测试结果: 成功 {success_count}/{test_count}, 失败 {fail_count}/{test_count}")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 80)
