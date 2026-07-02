# MFR数据集flow_mfr文件数量统计

统计时间：2025年11月4日

数据集路径：`/root/datasets/MFR数据集`

---

## 总体统计

四个子数据集的`flow_mfr`文件夹内所有文件总数：**2,179,958 个文件**

- **CICIoT2022_MFR**: 1,671,347 个文件
- **ISCXTor2016_MFR**: 13,476 个文件
- **ISCXVPN2016_MFR**: 13,520 个文件
- **USTC-TFC2016_MFR**: 481,615 个文件

---

## 1. CICIoT2022_MFR（总计：1,671,347 个文件）

| 类别 | 文件数量 | 占比 |
|------|---------|------|
| Idle | 1,405,324 | 84.08% |
| Attacks_Flood | 249,352 | 14.92% |
| Attacks_Nmap | 11,994 | 0.72% |
| Power_Cameras | 1,340 | 0.08% |
| Interactions_Cameras | 1,207 | 0.07% |
| Power_Audio | 1,183 | 0.07% |
| Interactions_Audio | 506 | 0.03% |
| Power_Other | 226 | 0.01% |
| Attacks_Hydra | 126 | 0.01% |
| Interactions_Other | 89 | 0.01% |

**说明**：该数据集主要包含IoT设备的流量数据，包括攻击流量（Flood、Nmap、Hydra）、空闲流量（Idle）、设备交互流量和电源相关流量。

---

## 2. ISCXTor2016_MFR（总计：13,476 个文件）

| 类别 | 文件数量 | 占比 |
|------|---------|------|
| VOIP | 4,271 | 31.69% |
| BROWSING | 2,399 | 17.80% |
| P2P | 2,063 | 15.31% |
| FILE-TRANSFER | 1,602 | 11.89% |
| VIDEO | 1,450 | 10.76% |
| AUDIO | 788 | 5.85% |
| MAIL | 469 | 3.48% |
| CHAT | 434 | 3.22% |

**说明**：该数据集包含Tor网络中的不同应用流量类型，主要用于加密流量分类研究。

---

## 3. ISCXVPN2016_MFR（总计：13,520 个文件）

| 类别 | 文件数量 | 占比 |
|------|---------|------|
| VOIP | 5,419 | 40.08% |
| CHAT | 4,029 | 29.79% |
| BROWSING | 1,928 | 14.26% |
| FTP | 1,020 | 7.54% |
| p2p | 477 | 3.53% |
| Streaming | 349 | 2.58% |
| MAIL | 298 | 2.20% |

**说明**：该数据集包含VPN加密流量，涵盖多种网络应用类型。

---

## 4. USTC-TFC2016_MFR（总计：481,615 个文件）

| 类别 | 文件数量 | 占比 |
|------|---------|------|
| FTP | 101,037 | 20.98% |
| MySQL | 86,089 | 17.88% |
| Geodo | 40,947 | 8.50% |
| Weibo | 39,950 | 8.29% |
| SMB | 38,937 | 8.08% |
| Neris | 33,791 | 7.02% |
| Virut | 33,103 | 6.87% |
| Cridex | 16,386 | 3.40% |
| Miuref | 13,481 | 2.80% |
| Zeus | 10,970 | 2.28% |
| Shifu | 9,634 | 2.00% |
| Gmail | 8,629 | 1.79% |
| Tinba | 8,504 | 1.77% |
| WorldOfWarcraft | 7,883 | 1.64% |
| BitTorrent | 7,517 | 1.56% |
| Htbot | 6,367 | 1.32% |
| Skype | 6,321 | 1.31% |
| Nsis-ay | 6,069 | 1.26% |
| Facetime | 6,000 | 1.25% |

**说明**：该数据集包含正常应用流量（FTP、MySQL、Gmail、Skype、Facetime、Weibo、BitTorrent、WorldOfWarcraft等）和恶意软件流量（Geodo、Neris、Virut、Cridex、Miuref、Zeus、Shifu、Tinba、Htbot、Nsis-ay等），适用于流量分类和恶意流量检测研究。

---

## 数据集结构说明

每个子数据集下的结构为：
```
数据集名称/
├── 类别1/
│   └── flow_mfr/
│       ├── train/    # 训练集
│       ├── valid/    # 验证集
│       └── test/     # 测试集
├── 类别2/
│   └── flow_mfr/
│       ├── train/
│       ├── valid/
│       └── test/
...
```

每个`flow_mfr`文件夹下包含三个子文件夹（train、valid、test），存储着对应的MFR（Multi-scale Flow Representation）特征数据。

---

## 数据集应用

这些数据集可用于：
1. **加密流量分类**：ISCXTor2016、ISCXVPN2016
2. **IoT设备流量分析**：CICIoT2022
3. **恶意流量检测**：USTC-TFC2016
4. **网络应用识别**：所有数据集
5. **预训练和微调模型训练**：使用MFR特征进行深度学习模型训练

