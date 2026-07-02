import argparse
import torch
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
uer_path = os.path.join(current_dir, 'uer')
if not os.path.exists(uer_path):
    raise ImportError(f"UER module not found. Please ensure the 'uer' folder exists in {current_dir}")

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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the preprocessed dataset (.pt file).")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path to the SentencePiece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path to the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path to the target SentencePiece model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to a pretrained model to resume from.")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path to save the output model.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the model config JSON file.")

    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000)
    parser.add_argument("--report_steps", type=int, default=100)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--instances_buffer_size", type=int, default=25600)
    parser.add_argument("--labels_num", type=int, required=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space")

    model_opts(parser)
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word_pos_seg")
    parser.add_argument("--decoder", choices=["transformer"], default="transformer")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first")
    parser.add_argument("--target", choices=["bert", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm", "multi_task"], default="mlm")
    parser.add_argument("--tie_weights", action="store_true")
    parser.add_argument("--has_lmtarget_bias", action="store_true")

    parser.add_argument("--whole_word_masking", action="store_true")
    parser.add_argument("--span_masking", action="store_true")
    parser.add_argument("--span_geo_prob", type=float, default=0.2)
    parser.add_argument("--span_max_length", type=int, default=10)

    parser.add_argument("--seq_length", type=int, default=330)

    parser.add_argument("--nsp_indicating_weight", type=float, default=0.50)
    parser.add_argument("--packet_selection_prob", type=float, default=0.25)
    parser.add_argument("--packet_mask_ratio", type=float, default=0.90)
    parser.add_argument("--tokens_per_packet", type=int, default=80)

    parser.add_argument("--mfr_masking_weight", type=float, default=0.50)
    parser.add_argument("--mfr_header_mask_ratio", type=float, default=0.80)
    parser.add_argument("--mfr_payload_mask_ratio", type=float, default=0.90)

    parser.add_argument("--use_packet_attention", action='store_true', default=True)
    parser.add_argument("--use_flow_attention", action='store_true', default=True)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--header_size", type=int, default=40)
    parser.add_argument("--num_attention_heads", type=int, default=8)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")

    optimization_opts(parser)

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int)
    unique_port = 12345 + (int(time.time()) % 10000)
    parser.add_argument("--master_ip", default=f"tcp://localhost:{unique_port}", type=str)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str)

    args = parser.parse_args()

    total_weight = args.nsp_indicating_weight + args.mfr_masking_weight
    assert abs(total_weight - 1.0) < 1e-6, f"Task weights must sum to 1.0, got: {total_weight}"

    if args.target == "cls":
        assert args.labels_num is not None, "cls target requires --labels_num"

    if args.config_path:
        load_hyperparam(args)

    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size
        assert ranks_num <= torch.cuda.device_count()
        args.dist_train = True
        args.ranks_num = ranks_num
    elif args.world_size == 1 and ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count(), "Invalid GPU device."
        args.dist_train = False
        args.single_gpu = True
    else:
        assert ranks_num == 0, "GPUs specified but world_size is 1 with no gpu_ranks."
        args.dist_train = False
        args.single_gpu = False

    model_save_dir = os.path.dirname(args.output_model_path)
    if not model_save_dir:
        model_save_dir = "."
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    latest_checkpoint = find_latest_checkpoint(model_save_dir, args.output_model_path)
    if latest_checkpoint:
        args.pretrained_model_path = latest_checkpoint
        try:
            args.checkpoint_step = int(latest_checkpoint.split('-')[-1])
        except (ValueError, IndexError):
            args.checkpoint_step = 0
    else:
        args.checkpoint_step = 0

    from multi_task_trainer import train_and_validate_multi_task
    train_and_validate_multi_task(args)


if __name__ == "__main__":
    main()
