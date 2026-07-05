"""
整合两个预训练任务的脚本：

数据共享策略：
- 每个step加载N个流样本（N = batch_size，例如8个）
- 这N个流同时被两个任务处理，生成2N个样本（每个流产生2个样本）
- 两个任务共享同一批流数据，样本数始终相等

两个预训练任务：
1. NSP_Indicating (Packet Masking) - 50%
   - 使用流级别Packet Masking策略（h-h-h-h-h和p-p-p-p-p关系版本）
   - 预训练任务: MLM (仅Masked Language Model)
   - 输入: 流MFR矩阵（前4个数据包，每个80 tokens）
   - Sequence Length: 325 tokens (包含CLS、SEP、padding)
   - 每个流产生1个样本
   - 两种操作，随机选择：
     - 操作1（h-h-h-h-h关系）：将流的四个数据包MFR矩阵的payload都设置为CNS，随机mask某一个数据包MFR矩阵的header，mask率90%
     - 操作2（p-p-p-p-p关系）：将流的四个数据包MFR矩阵的header都设置为CNS，随机mask某一个数据包MFR矩阵的payload，mask率90%

2. MFR-Masking (Header/Payload Masking) - 50%
   - 对每条流，对前4个数据包MFR矩阵分别进行处理
   - 每个数据包：80 tokens（Header 40 + Payload 40）
   - 单个目标数据包mask后放入4包显式边界槽位，其它包位用PAD占位
   - Sequence Length: 325 tokens (CLS + 4 * (80 + SEP))以匹配NSP_Indicating
   - 每个流产生4个样本（每个数据包1个样本）
   - 三种关系masking策略（按所有数据包划分）：
     - h-h关系 (30%)：Payload用[CNS]替代，对Header随机mask 80%
     - p-p关系 (30%)：Header用[CNS]替代，对Payload随机mask 80%
     - h-p关系 (40%)：对Header随机mask 95%，用Payload预测Header

输入数据格式：
- 原始数据：流MFR矩阵（前5个数据包，每个160 tokens）
- 处理后：流MFR矩阵（前4个数据包，每个80 tokens）
- 预训练任务使用处理后的流MFR矩阵
"""

import argparse
# PyTorch深度学习框架
import torch
# UER，即，Universal Encoder Representations
# UER是一个用于预训练和微调Transformer模型的工具包
import sys
import os
import random
import time
# 修改路径设置：使用当前目录下的UER模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 添加当前目录到Python路径（包含uer文件夹）
# 检查当前目录下是否存在uer模块
uer_path = os.path.join(current_dir, 'uer')
if not os.path.exists(uer_path):
    raise ImportError(f"未找到uer模块，请确保当前目录({current_dir})下存在uer文件夹")
# 现在尝试导入UER模块
import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from uer.opts import *


def find_latest_checkpoint(model_dir, base_model_path):
    """
    查找最新的模型检查点

    Args:
        model_dir: 模型保存目录
        base_model_path: 基础模型路径（用于提取基础名称）

    Returns:
        最新检查点的完整路径，如果没有找到则返回None
    """
    if not os.path.exists(model_dir):
        return None

    # 获取基础模型名称（不包含步数）
    base_name = os.path.basename(base_model_path)

    # 查找所有匹配的检查点文件
    checkpoints = []
    for filename in os.listdir(model_dir):
        # 检查文件是否以基础名称开头且包含步数
        if filename.startswith(base_name) and '-' in filename:
            try:
                # 尝试从文件名提取步数
                step = int(filename.split('-')[-1])
                full_path = os.path.join(model_dir, filename)
                if os.path.isfile(full_path):
                    checkpoints.append((step, full_path))
            except (ValueError, IndexError):
                continue

    if checkpoints:
        # 返回步数最大的检查点
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]

    return None


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options. 
    # 指定各种path
    parser.add_argument("--dataset_path", type=str, default="/root/datasets_new/pt_packet_datasets/dataset_pretrain.pt",
                        help="Path of the preprocessed dataset (same format as NSP_Indicating).")
    parser.add_argument("--vocab_path", default="/root/datasets_new/pt_packet_datasets/vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                        help="Config file of model hyper-parameters.")

    # Training and saving options. 
    # 总steps
    # 在一个 Epoch 中，Step 的数量等于训练数据集的总样本数除以 Batch Size
    # 总steps = epoch数 * 单个epoch的steps
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    # accumulation_steps：指定梯度积累的步数，实际有效批次大小 = batch_size × world_size × accumulation_steps
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    # batch size：指定每个GPU上的批次大小
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")
    # instances_buffer_size：指定内存实例缓冲区的大小
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--labels_num", type=int, required=False,
                        help="Number of prediction labels.")
    # 指定dropout率，防止过拟合
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    # 指定随机种子，确保实验的可重复性。相同的种子会产生相同的随机数序列，从而使训练过程确定化。
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Preprocess options.
    # 指定要使用的tokenizer类型，将原始文本分割成模型可以处理的token 
    # bert/char/space 类型
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Model options.
    # 向参数解析器添加一系列模型架构相关的参数，如隐藏层大小、注意力头数、层数等。具体的参数取决于 model_opts函数的实现。
    model_opts(parser)
    # 指定目标嵌入类型
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word_pos_seg",
                        help="Target embedding type.")
    # 指定解码器类型
    parser.add_argument("--decoder", choices=["transformer"], default="transformer", help="Decoder type.")
    # 指定池化策略
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    # 指定预训练目标
    # 注意：multi_task模式下此参数不会被使用，因为我们直接构建自定义的多任务模型
    parser.add_argument("--target", choices=["bert", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm", "multi_task"], default="mlm",
                        help="The training target of the pretraining model (not used in multi_task mode).")
    # 选择是否绑定词嵌入和softmax层的权重（共享参数），以减少模型参数数量
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    # 选择是否在语言模型目标的输出层添加偏置项
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")

    # Masking options.
    # 选择掩码策略（标准mask：mask单个token）
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    # mask连续的token序列
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")
    
    # Sequence length option
    parser.add_argument("--seq_length", type=int, default=325,
                        help="Sequence length for flow MFR matrix (default: 325 tokens = CLS + 4 * (80 tokens + SEP)).")
    
    # NSP_Indicating任务参数（Packet Masking）
    parser.add_argument("--nsp_indicating_weight", type=float, default=0.50,
                        help="Weight for NSP Indicating task (default: 0.50, i.e., 50%%).")
    parser.add_argument("--packet_selection_prob", type=float, default=0.25,
                        help="Probability of selecting each packet for masking (default: 0.25, i.e., 25%%).")
    parser.add_argument("--packet_mask_ratio", type=float, default=0.90,
                        help="Masking ratio for selected packet (default: 0.90, i.e., 90%% of tokens in the selected packet).")
    parser.add_argument("--tokens_per_packet", type=int, default=80,
                        help="Number of tokens per packet (default: 80).")
    
    # MFR-Masking任务参数
    parser.add_argument("--mfr_masking_weight", type=float, default=0.50,
                        help="Weight for MFR Masking task (default: 0.50, i.e., 50%%).")
    parser.add_argument("--mfr_header_mask_ratio", type=float, default=0.80,
                        help="Masking ratio for header region (default: 0.80, i.e., mask 80%% of header tokens when header is selected).")
    parser.add_argument("--mfr_payload_mask_ratio", type=float, default=0.90,
                        help="Masking ratio for payload region (default: 0.90, i.e., mask 90%% of payload tokens when payload is selected).")

    # 两级注意力机制参数
    parser.add_argument("--use_packet_attention", action='store_true', default=True,
                        help="Enable packet-level attention mechanism.")
    parser.add_argument("--use_flow_attention", action='store_true', default=True,
                        help="Enable flow-level attention mechanism.")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Window size for payload-payload window attention in PacketAttention.")
    parser.add_argument("--header_size", type=int, default=40,
                        help="Number of tokens in packet header.")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads for both packet and flow attention.")

    # DataLoader options.
    # 数据加载器参数：启用多进程异步加载
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of worker processes for data loading (default: 4). Set to 0 to disable multiprocessing.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches loaded in advance by each worker (default: 2).")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use pinned memory for faster CPU-to-GPU transfer.")
    
    # Optimizer options.
    # 向参数解析器添加optimizer相关的配置选项
    optimization_opts(parser)

    # GPU options.
    # world_size：进程数量
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    # 指定各个进程分别使用什么GPU，例如，--gpu_ranks 0 1 2 3表示使用4个GPU，排名分别为0、1、2、3；如果列表为空，则表示使用CPU进行训练
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")
    # 分布式训练指定主节点的IP地址
    # 使用时间戳生成唯一端口避免冲突
    unique_port = 12345 + (int(time.time()) % 10000)
    parser.add_argument("--master_ip", default=f"tcp://localhost:{unique_port}", type=str, help="IP-Port of master for training.")
    # 分布式训练指定后端通信库
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")



    # 解析用户在命令行中提供的所有参数，并将其存储在 args对象中
    args = parser.parse_args()

    # 验证任务权重之和为1.0
    total_weight = args.nsp_indicating_weight + args.mfr_masking_weight
    assert abs(total_weight - 1.0) < 1e-6, f"任务权重之和必须为1.0，当前为: {total_weight}"

    # 如果训练目标是分类任务（cls），则必须提供labels_num（类别数量）
    if args.target == "cls":
        assert args.labels_num is not None, "Cls target needs the denotation of the number of labels."

    # Load hyper-parameters from config file. 
    if args.config_path:
        load_hyperparam(args)

    # 通过 gpu_ranks列表的长度确定用户指定了多少个GPU用于训练
    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        # Multiprocessing distributed mode.
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit."
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs."
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        # Single GPU mode.
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."
        args.dist_train = False
        args.single_gpu = True
        print("Using GPU %d for training." % args.gpu_id)
    else:
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    # 检测并创建模型保存目录
    model_save_dir = os.path.dirname(args.output_model_path)
    if not model_save_dir:
        model_save_dir = "/root/models/pre-trained_model_two_tasks_with_two-attention"
    if not os.path.exists(model_save_dir):
        print(f"创建模型保存目录: {model_save_dir}")
        os.makedirs(model_save_dir, exist_ok=True)
    else:
        print(f"模型保存目录已存在: {model_save_dir}")

    # 自动查找最新的模型检查点
    latest_checkpoint = find_latest_checkpoint(model_save_dir, args.output_model_path)
    if latest_checkpoint:
        print(f"发现最新的模型检查点: {latest_checkpoint}")
        args.pretrained_model_path = latest_checkpoint
        # 从检查点名称提取步数
        try:
            checkpoint_step = int(latest_checkpoint.split('-')[-1])
            args.checkpoint_step = checkpoint_step
            print(f"将从第 {checkpoint_step} 步继续训练")
        except (ValueError, IndexError):
            args.checkpoint_step = 0
            print("无法从检查点名称提取步数，将从头开始训练")
    else:
        args.checkpoint_step = 0
        print("未发现模型检查点，将从头开始训练")
    
    # 使用多任务预训练trainer（优化版：分批次加载）
    print("=" * 80)
    print("多任务预训练 - 整合两个预训练任务 (分批次加载 + 两级注意力)")
    print("=" * 80)
    print(f"数据处理流程:")
    print(f"  - 原始数据: 流MFR矩阵（前5个数据包 × 160 tokens = 800 tokens）")
    print(f"  - 处理策略: 取前4个数据包，每个数据包只取前{args.tokens_per_packet} tokens")
    print(f"  - 处理后数据: 流MFR矩阵（前4个数据包 × {args.tokens_per_packet} tokens = {args.tokens_per_packet * 4} tokens）")
    print(f"  - Sequence Length: {args.seq_length} tokens (包含CLS、SEP、padding)")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - 内存优化: 从800 tokens降低到320 tokens (减少60%)")
    print("")
    print(f"数据加载策略 (分批次):")
    print(f"  - 使用小buffer快速启动训练")
    print(f"  - 边加载边训练，GPU不等待")
    print(f"  - 内存友好，适合大规模数据")
    print("")
    print(f"数据共享策略:")
    print(f"  - 每个step加载 {args.batch_size} 个流样本")
    print(f"  - 这 {args.batch_size} 个流同时被两个任务处理")
    print(f"  - NSP_Indicating: {args.batch_size} 个样本（每流1个）")
    print(f"  - MFR-Masking: {args.batch_size * 4} 个样本（每流4个数据包）")
    print(f"  - 每个step产生 {args.batch_size * 5} 个训练样本 ({args.batch_size} + {args.batch_size * 4})")
    print("")
    print("=" * 80)
    print("两级注意力机制配置")
    print("=" * 80)
    print(f"Packet size: {args.tokens_per_packet} tokens (Header: {args.header_size}, Payload: {args.tokens_per_packet - args.header_size})")
    print(f"Number of attention heads: {args.num_attention_heads}")
    print("")
    
    # 包级注意力配置
    print("【Level 1: Packet-Level Attention】")
    print(f"  Status: {'✓ Enabled' if args.use_packet_attention else '✗ Disabled'}")
    if args.use_packet_attention:
        print(f"  Purpose: 对每个数据包内部的header-payload关系建模")
        print(f"  Mechanisms:")
        print(f"    (1) h-h关系: Header内部全局注意力 ({args.header_size} tokens)")
        print(f"    (2) h-p关系: Payload作q，Header作k&v的交叉注意力")
        print(f"    (3) p-p关系: Payload内部窗口注意力 (window={args.window_size})")
    print("")
    
    # 流级注意力配置
    print("【Level 2: Flow-Level Attention】")
    print(f"  Status: {'✓ Enabled' if args.use_flow_attention else '✗ Disabled'}")
    if args.use_flow_attention:
        print(f"  Purpose: 对4个数据包之间的跨包依赖关系建模")
        print(f"  Input: 包级注意力的输出（池化后）")
        print(f"  Payload池化: 40 tokens -> 4 tokens (每10个tokens平均池化)")
        print(f"  Mechanisms:")
        print(f"    (1) 数据重组: 拼接4个数据包的header和池化后的payload")
        print(f"    (2) 滑动窗口注意力:")
        print(f"        - 包1和包4: 窗口大小=2（attend到1个相邻包）")
        print(f"        - 包2和包3: 窗口大小=3（attend到2个相邻包）")
        print(f"    (3) h查h: Header的tokens作q，相邻包的header作k&v")
        print(f"    (4) p查p: Payload的tokens作q，相邻包的payload作k&v")
    print("=" * 80)
    print("")
    print("两个预训练任务及其权重:")
    print(f"  1. NSP_Indicating (Packet Masking): {args.nsp_indicating_weight*100:.0f}%")
    print(f"     - 使用流级别Packet Masking策略（h-h-h-h-h和p-p-p-p-p关系版本）")
    print(f"     - 输入: 流MFR矩阵（前4个数据包，每个80 tokens）")
    print(f"     - Sequence Length: 325 tokens")
    print(f"     - 两种操作，随机选择：")
    print(f"       * 操作1（h-h-h-h-h关系）：payload设置为CNS，随机mask一个header（90%）")
    print(f"       * 操作2（p-p-p-p-p关系）：header设置为CNS，随机mask一个payload（90%）")
    print("")
    print(f"  2. MFR-Masking (Header/Payload Masking): {args.mfr_masking_weight*100:.0f}%")
    print(f"     - 对每条流，对前4个数据包MFR矩阵分别进行处理")
    print(f"     - 每个数据包：80 tokens（Header 40 + Payload 40）")
    print(f"     - 单个目标数据包mask后放入4包显式边界槽位，其它包位用PAD占位")
    print(f"     - Sequence Length: 325 tokens (CLS + 4 * (80 + SEP))")
    print(f"     - 每个流产生4个样本（每个数据包1个样本）")
    print(f"     - 三种关系masking策略（按所有数据包划分）：")
    print(f"       * h-h关系 (30%)：Payload用[CNS]替代，对Header随机mask 80%")
    print(f"       * p-p关系 (30%)：Header用[CNS]替代，对Payload随机mask 80%")
    print(f"       * h-p关系 (40%)：对Header随机mask 95%，用Payload预测Header")
    print("")
    print(f"总损失函数: Loss = {args.nsp_indicating_weight}*Loss_Indicating + {args.mfr_masking_weight}*Loss_MFR")
    print("=" * 80)
    
    # 使用原始trainer（稳定版）
    from multi_task_trainer import train_and_validate_multi_task
    train_and_validate_multi_task(args)


if __name__ == "__main__":
    main()
