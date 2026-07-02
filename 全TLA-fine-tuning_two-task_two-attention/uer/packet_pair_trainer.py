"""
数据包对预测任务 (Packet Pair Prediction Task) 的训练器
"""
import os
import pickle
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.utils.packet_pair_data import PacketPairDataLoader, PacketPairDataset
from uer.targets.packet_pair_target import PacketPairTarget
from tqdm import tqdm


def train_and_validate_packet_pair(args):
    """
    数据包对预测任务的训练函数
    """
    set_seed(args.seed)
    
    # 加载词汇表
    print("=" * 80)
    print("Loading vocabulary...")
    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "You need to install SentencePiece: "
                "pip install sentencepiece"
            )
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i
                      in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
    
    print(f"Vocabulary size: {len(args.vocab)}")
    
    # 构建模型
    print("Building model...")
    model = build_model(args)
    
    # 加载或初始化参数
    if args.pretrained_model_path is not None:
        print(f"Loading pretrained model from {args.pretrained_model_path}...")
        model = load_model(model, args.pretrained_model_path)
    else:
        print("Initializing model with normal distribution...")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    
    # 替换输出层为PacketPairTarget
    print("Replacing output target with PacketPairTarget...")
    args.labels_num = 3  # 3分类
    model.target = PacketPairTarget(args, len(args.vocab))
    
    print("=" * 80)
    
    # 进行分布式训练或单机训练
    if args.dist_train:
        # 多进程分布式模式
        print("Starting distributed training...")
        mp.spawn(
            worker_packet_pair,
            nprocs=args.ranks_num,
            args=(args.gpu_ranks, args, model),
            daemon=False
        )
    elif args.single_gpu:
        # 单GPU模式
        print(f"Starting training on GPU {args.gpu_id}...")
        worker_packet_pair(args.gpu_id, None, args, model)
    else:
        # CPU模式
        print("Starting training on CPU...")
        worker_packet_pair(None, None, args, model)


class PacketPairTrainer(object):
    """数据包对预测任务的训练器类"""
    
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.output_model_path = args.output_model_path
        
        self.start_time = time.time()
        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0
        
        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
    
    def forward_propagation(self, batch, model):
        """
        前向传播
        
        Args:
            batch: (src, tgt_label)
            model: 模型
        
        Returns:
            loss: 损失
        """
        src, tgt_label = batch
        
        # 将段ID设置为全0（因为我们处理的是两个数据包的拼接）
        # [CLS] + packet1 + [SEP] + packet2 + [SEP]
        seg = torch.zeros_like(src)
        
        # 模型前向传播
        # model输出为 (loss, correct)
        loss_info = model(src, tgt_label, seg)
        loss, correct = loss_info
        
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)  # batch_size
        
        loss = loss / self.accumulation_steps
        return loss
    
    def report_and_reset_stats(self):
        """报告和重置统计信息"""
        done_tokens = self.batch_size * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        
        accuracy = self.total_correct / max(1, self.total_instances)
        print(
            f"| {self.current_step:8d}/{self.total_steps:8d} steps"
            f"| {done_tokens / (time.time() - self.start_time):8.2f} instances/s"
            f"| loss {self.total_loss / self.report_steps:7.2f}"
            f"| acc: {accuracy:3.3f}"
        )
        
        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0
    
    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        """
        训练循环
        """
        model.train()
        loader_iter = iter(loader)
        
        while True:
            if self.current_step == self.total_steps + 1:
                break
            
            try:
                batch = next(loader_iter)
            except StopIteration:
                # 数据加载器已完成
                break
            
            batch = list(batch)
            self.seq_length = batch[0].size(1)
            
            # 转移到GPU
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)
            
            # 前向传播
            loss = self.forward_propagation(batch, model)
            
            # 反向传播
            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            # 梯度积累
            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            # 报告统计信息
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()
            
            # 保存检查点
            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))
            
            self.current_step += 1


def worker_packet_pair(gpu_id, gpu_ranks, args, model):
    """
    工作进程函数
    """
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    
    rank = 0
    if args.dist_train:
        rank = gpu_ranks.index(gpu_id)
        
        # 使用环境变量初始化分布式训练
        import os
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        
        # 初始化分布式进程组
        dist.init_process_group(
            backend=args.backend if args.backend else 'nccl',
            init_method='env://',  # 使用环境变量而不是tcp://
            world_size=args.world_size,
            rank=rank
        )
        print(f"[Rank {rank}] Distributed training initialized")
    
    # 加载数据
    print("Loading data...")
    
    # 首先检查数据集是否存在
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}")
        print("Building dataset...")
        
        # 构建数据集
        dataset = PacketPairDataset(args, args.vocab, args.tokenizer)
        dataset.build_and_save(
            workers_num=args.world_size if args.dist_train else 1
        )
        
        # 合并数据集
        dataset_writer = open(args.dataset_path, "wb")
        for i in range(args.world_size if args.dist_train else 1):
            tmp_path = f"/tmp/dataset-packet-pair-{i}.pt"
            if os.path.exists(tmp_path):
                with open(tmp_path, "rb") as tmp_reader:
                    while True:
                        try:
                            instance = pickle.load(tmp_reader)
                            pickle.dump(instance, dataset_writer)
                        except EOFError:
                            break
                os.remove(tmp_path)
        dataset_writer.close()
        print(f"Dataset saved to {args.dataset_path}")
    
    # 创建数据加载器
    loader = PacketPairDataLoader(
        args,
        args.dataset_path,
        args.batch_size,
        rank if args.dist_train else 0,
        args.world_size if args.dist_train else 1,
        shuffle=True
    )
    
    # 先将模型移到GPU
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    
    # 设置模型为分布式模式（如果需要）
    if args.dist_train:
        model = DistributedDataParallel(model, device_ids=[gpu_id])
    
    # 设置优化器和学习率调度器
    print("Setting up optimizer and scheduler...")
    optimizer = str2optimizer[args.optimizer](model.parameters(), lr=args.learning_rate)
    scheduler = str2scheduler[args.scheduler](optimizer, args.warmup, args.total_steps)
    
    # 创建训练器
    trainer = PacketPairTrainer(args)
    
    # 开始训练
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    trainer.train(args, gpu_id, rank, loader, model, optimizer, scheduler)
    
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

