import argparse
import torch
import sys
import os
import random
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  

uer_path = os.path.join(current_dir, 'uer')
if not os.path.exists(uer_path):
    raise ImportError(f"未找到uer模块，请确保当前目录({current_dir})下存在uer文件夹")

import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from uer.opts import *


def find_latest_checkpoint(model_dir, base_model_path):
    
    if not os.path.exists(model_dir):
        return None

    
    base_name = os.path.basename(base_model_path)

    
    checkpoints = []
    for filename in os.listdir(model_dir):
        
        if filename.startswith(base_name) and '-' in filename:
            try:
                
                step = int(filename.split('-')[-1])
                full_path = os.path.join(model_dir, filename)
                if os.path.isfile(full_path):
                    checkpoints.append((step, full_path))
            except (ValueError, IndexError):
                continue

    if checkpoints:
        
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]

    return None


def extract_checkpoint_step(checkpoint_path):
    if not checkpoint_path:
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'step' in checkpoint:
            return int(checkpoint['step'])
    except Exception:
        pass

    try:
        return int(str(checkpoint_path).split('-')[-1])
    except (ValueError, IndexError):
        return 0


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path of the preprocessed dataset (same format as NSP_Indicating).")
    parser.add_argument("--vocab_path", default="", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default="", type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default="", type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default="", type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="Path of the pretrained model weights used for initialization.")
    parser.add_argument("--resume_checkpoint_path", type=str, default="",
                        help="Path of a full training checkpoint to resume from. This has priority over --pretrained_model_path.")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Automatically resume from the latest checkpoint under output_model_path directory when no explicit checkpoint is given.")
    parser.add_argument("--output_model_path", type=str, default="",
                        help="Path of the output model.")
    parser.add_argument("--config_path", type=str, default="",
                        help="Config file of model hyper-parameters.")
    
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")
    
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--labels_num", type=int, required=False,
                        help="Number of prediction labels.")
    
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    
    
    model_opts(parser)
    
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word_pos_seg",
                        help="Target embedding type.")
    
    parser.add_argument("--decoder", choices=["transformer"], default="transformer", help="Decoder type.")
    
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    
    
    parser.add_argument("--target", choices=["bert", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm", "multi_task"], default="mlm",
                        help="The training target of the pretraining model (not used in multi_task mode).")
    
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")
    
    parser.add_argument("--seq_length", type=int, default=410,
                        help="Sequence length for flow MFR matrix (default: 410 tokens = CLS + 5 * (80 tokens + SEP) plus padding).")
    
    parser.add_argument("--nsp_indicating_weight", type=float, default=0.50,
                        help="Weight for NSP Indicating task (default: 0.50, i.e., 50%%).")
    parser.add_argument("--packet_selection_prob", type=float, default=0.25,
                        help="Probability of selecting each packet for masking (default: 0.25, i.e., 25%%).")
    parser.add_argument("--packet_mask_ratio", type=float, default=0.90,
                        help="Masking ratio for selected packet (default: 0.90, i.e., 90%% of tokens in the selected packet).")
    parser.add_argument("--tokens_per_packet", type=int, default=80,
                        help="Number of tokens per packet (default: 80).")
    
    
    parser.add_argument("--mfr_masking_weight", type=float, default=0.50,
                        help="Weight for MFR Masking task (default: 0.50, i.e., 50%%).")
    parser.add_argument("--mfr_header_mask_ratio", type=float, default=0.80,
                        help="Masking ratio for header region (default: 0.80, i.e., mask 80%% of header tokens when header is selected).")
    parser.add_argument("--mfr_payload_mask_ratio", type=float, default=0.90,
                        help="Masking ratio for payload region (default: 0.90, i.e., mask 90%% of payload tokens when payload is selected).")

    
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

    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of worker processes for data loading (default: 4). Set to 0 to disable multiprocessing.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches loaded in advance by each worker (default: 2).")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use pinned memory for faster CPU-to-GPU transfer.")
    
    optimization_opts(parser)
    
    
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")
    
    
    unique_port = 12345 + (int(time.time()) % 10000)
    parser.add_argument("--master_ip", default=f"tcp://localhost:{unique_port}", type=str, help="IP-Port of master for training.")
    
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")
    
    args = parser.parse_args()

    
    total_weight = args.nsp_indicating_weight + args.mfr_masking_weight
    assert abs(total_weight - 1.0) < 1e-6, f"任务权重之和必须为1.0，当前为: {total_weight}"

    
    if args.target == "cls":
        assert args.labels_num is not None, "Cls target needs the denotation of the number of labels."

    
    if args.config_path:
        load_hyperparam(args)

    
    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        
        assert torch.cuda.is_available(), "No available GPUs."
        cuda_device_count = torch.cuda.device_count()
        assert ranks_num == args.world_size, (
            f"world_size ({args.world_size}) must match the number of gpu_ranks ({ranks_num}) "
            "for single-node distributed training."
        )
        invalid_gpu_ranks = [gpu_id for gpu_id in args.gpu_ranks if gpu_id < 0 or gpu_id >= cuda_device_count]
        assert not invalid_gpu_ranks, (
            f"Invalid GPU id(s): {invalid_gpu_ranks}. Available GPU ids are 0..{cuda_device_count - 1}."
        )
        args.dist_train = True
        args.ranks_num = ranks_num
        pass
    elif args.world_size == 1 and ranks_num == 1:
        
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."
        args.dist_train = False
        args.single_gpu = True
        pass
    else:
        
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        pass

    
    model_save_dir = os.path.dirname(args.output_model_path)
    if not model_save_dir:
        model_save_dir = ""
    if model_save_dir and not os.path.exists(model_save_dir):
        pass
        os.makedirs(model_save_dir, exist_ok=True)
    else:
        pass

    
    args.checkpoint_step = 0
    if args.resume_checkpoint_path:
        if not os.path.isfile(args.resume_checkpoint_path):
            raise FileNotFoundError(f"指定的checkpoint不存在: {args.resume_checkpoint_path}")
        args.pretrained_model_path = args.resume_checkpoint_path
        args.checkpoint_step = extract_checkpoint_step(args.resume_checkpoint_path)
        pass
        if args.checkpoint_step > 0:
            pass
        else:
            pass
    elif args.auto_resume:
        latest_checkpoint = find_latest_checkpoint(model_save_dir, args.output_model_path)
        if latest_checkpoint:
            args.pretrained_model_path = latest_checkpoint
            args.checkpoint_step = extract_checkpoint_step(latest_checkpoint)
            if args.checkpoint_step > 0:
                pass
            else:
                pass
        else:
            pass
    elif args.pretrained_model_path:
        pass
    else:
        pass
    
    if args.use_packet_attention:
        pass
    

    if args.use_flow_attention:
        pass

    
    
    from multi_task_trainer import train_and_validate_multi_task
    train_and_validate_multi_task(args)


if __name__ == "__main__":
    main()
