"""
数据包对预测任务 (Packet Pair Prediction Task)
预测两个数据包是否属于同一流以及其时间顺序

Label定义：
- 0: 不同流（来自两条不同的流）
- 1: 同一流，packet1在前（根据时间戳，packet1先出现）
- 2: 同一流，packet1在后（根据时间戳，packet1后出现）
"""
import torch
import torch.nn as nn
from uer.utils import *


class PacketPairTarget(nn.Module):
    """
    数据包对预测任务目标模块
    使用CLS token的表示进行3分类：
    1. 不同流（label=0）
    2. 同一流，packet1在前（label=1）
    3. 同一流，packet1在后（label=2）
    """
    
    def __init__(self, args, vocab_size):
        super(PacketPairTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.num_labels = 3  # 3分类：不同流、packet1在前、packet1在后
        
        # 预测层
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activation = nn.Tanh()
        self.linear_2 = nn.Linear(args.hidden_size, self.num_labels)
        
        # 损失函数
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
    
    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
                         由Transformer encoder输出的所有token的表示
            tgt: [batch_size] 
                 每个样本的标签（0, 1, 或 2）
        
        Returns:
            loss: 预测损失
            correct: 预测正确的数量
        """
        
        # 使用CLS token的表示（第一个token）进行预测
        # memory_bank[:, 0, :] shape: [batch_size x hidden_size]
        cls_output = memory_bank[:, 0, :]
        
        # 通过线性层和激活函数
        output = self.linear_1(cls_output)
        output = self.activation(output)
        output = self.linear_2(output)
        
        # 计算损失
        output_log_probs = self.softmax(output)
        loss = self.criterion(output_log_probs, tgt)
        
        # 计算准确率
        predictions = output_log_probs.argmax(dim=-1)
        correct = (predictions == tgt).sum()
        
        return loss, correct

