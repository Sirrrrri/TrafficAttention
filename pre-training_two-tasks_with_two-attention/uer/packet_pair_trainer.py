import os
import pickle
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from threading import Thread
from queue import Queue
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
    set_seed(args.seed)

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
        args.vocab = {
            sp_model.IdToPiece(i): i
            for i in range(sp_model.GetPieceSize())
        }
        args.tokenizer = str2tokenizer[args.tokenizer](args)
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab

    model = build_model(args)

    if args.pretrained_model_path is not None:
        model = load_model(model, args.pretrained_model_path)
    else:
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    args.labels_num = 3
    model.target = PacketPairTarget(args, len(args.vocab))

    if args.dist_train:
        mp.spawn(
            worker_packet_pair,
            nprocs=args.ranks_num,
            args=(args.gpu_ranks, args, model),
            daemon=False
        )
    elif args.single_gpu:
        worker_packet_pair(args.gpu_id, None, args, model)
    else:
        worker_packet_pair(None, None, args, model)


class PrefetchDataLoader(object):
    def __init__(self, data_loader, prefetch_queue_size=2):
        self.data_loader = data_loader
        self.prefetch_queue_size = prefetch_queue_size
        self.queue = Queue(maxsize=prefetch_queue_size)
        self.thread = None

    def _prefetch_worker(self):
        try:
            for batch in self.data_loader:
                self.queue.put(batch, timeout=60)
        except Exception:
            pass
        finally:
            self.queue.put(None)

    def __iter__(self):
        self.thread = Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

        if self.thread is not None:
            self.thread.join()


class PacketPairTrainer(object):
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
        src, tgt_label = batch
        seg = torch.zeros_like(src)
        loss, correct = model(src, tgt_label, seg)

        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)

        return loss / self.accumulation_steps

    def report_and_reset_stats(self):
        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        prefetch_loader = PrefetchDataLoader(loader, prefetch_queue_size=2)
        loader_iter = iter(prefetch_loader)

        while True:
            if self.current_step == self.total_steps + 1:
                break

            try:
                batch = next(loader_iter)
            except (StopIteration, Exception):
                break

            batch = list(batch)
            self.seq_length = batch[0].size(1)

            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id, non_blocking=True)

            loss = self.forward_propagation(batch, model)

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()

            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                save_model(
                    model,
                    self.output_model_path + "-" + str(self.current_step)
                )

            self.current_step += 1


def worker_packet_pair(gpu_id, gpu_ranks, args, model):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    rank = 0
    if args.dist_train:
        rank = gpu_ranks.index(gpu_id)

        import os
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        dist.init_process_group(
            backend=args.backend if args.backend else 'nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )

    if not os.path.exists(args.dataset_path):
        dataset = PacketPairDataset(args, args.vocab, args.tokenizer)
        dataset.build_and_save(
            workers_num=args.world_size if args.dist_train else 1
        )

        dataset_writer = open(args.dataset_path, "wb")
        for i in range(args.world_size if args.dist_train else 1):
            tmp_path = ""
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

    loader = PacketPairDataLoader(
        args,
        args.dataset_path,
        args.batch_size,
        rank if args.dist_train else 0,
        args.world_size if args.dist_train else 1,
        shuffle=True
    )

    if gpu_id is not None:
        model = model.cuda(gpu_id)

    if args.dist_train:
        model = DistributedDataParallel(model, device_ids=[gpu_id])

    optimizer = str2optimizer[args.optimizer](
        model.parameters(),
        lr=args.learning_rate
    )
    scheduler = str2scheduler[args.scheduler](
        optimizer,
        args.warmup,
        args.total_steps
    )
    trainer = PacketPairTrainer(args)
    trainer.train(args, gpu_id, rank, loader, model, optimizer, scheduler)
