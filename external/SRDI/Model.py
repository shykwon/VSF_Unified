import torch
import numpy as np
import argparse
import time
import torch.nn as nn
from util import *
# Lazy import to avoid circular dependency with diff_models
# from main_model import *
import torch.nn.functional as F

class dis_forc(nn.Module):
    def __init__(self, config, device, model2, target_dim = 140):
        super(dis_forc,self).__init__()
        self.config = config
        self.device = device
        self.target_dim = target_dim
        # Lazy import to avoid circular dependency
        from main_model import CSDI_Forecasting
        self.model= CSDI_Forecasting(config = config, device = device, target_dim = target_dim).to(device) #difussion module


    def forward(self,task,idx=None, args=None, mask_remaining=False, test_idx_subset=None):


        #mask掉一部分
        batch = data_processing(task,0.85,self.target_dim)
        loss1,loss_d= self.model(batch)

        samples = self.model.evaluate(batch,1)
        samples = samples.permute(0, 1, 3, 2)# (1,nsample,L,K)
        samples_median = samples.median(dim=1).values #(B,L,K)

        return loss1,loss_d, samples_median


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.fc2(out)
        invariant = out
        return invariant


class dispatcher(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, dropout=0.1):
        super(dispatcher,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.residual_blocks = ResidualBlock(input_size, hidden_size, dropout)

    def forward(self,x):#x(64,12,140)
        next = x
        invariant = self.residual_blocks(next)
        temp = invariant.view(-1,self.input_size)
        loss = compute(self.input_size,self.input_size,temp)
        next = next - invariant
        invariant_total = torch.zeros_like(invariant)
        invariant_total = invariant_total + invariant
        for _ in range(self.num_layers):
            invariant_new = self.residual_blocks(next)
            invariant_total = invariant_total + invariant_new
            z = invariant_new.view(-1,self.input_size)
            loss += compute(self.input_size,self.input_size,z)
            next = next - invariant_new
        #next就是variant
        return invariant_total,next,loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # 两个输入合并后的维度
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_invariant, x_variant):
        # 将不变模式和可变模式沿着特征维度拼接
        x = torch.cat((x_invariant, x_variant), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#x_invariant和x_variant是两个时间序列数据，形状为(batch_size, seq_len, input_dim)
# 整合两个时间序列数据
def integrate_patterns(x_invariant, x_variant, mlp_model):
    # 将两个时间序列数据沿着特征维度拼接
    x_combined = torch.cat((x_invariant, x_variant), dim=-1)
    # 使用MLP模型整合数据
    x_final = mlp_model(x_invariant, x_variant)
    return x_final

def compute(input_dim, output_dim, input_data):
    """
    Compute temporal consistency loss for invariant patterns.

    MODIFIED for memory efficiency (GTX 1080 Ti 11GB):
    - Original: stored full (batch*seq, dim, dim) correlation_matrix (~10GB for 207 nodes)
    - Modified: only stores previous (dim, dim) matrix, computes diff on-the-fly
    - TODO: Revert to original when running on larger GPU (32GB+)
    """
    device = input_data.device
    layer = nn.Linear(input_dim, output_dim).to(device)
    output_data = layer(input_data)

    flattened_tensor = output_data

    x = flattened_tensor.shape[0]
    y = output_dim

    # Memory-efficient version: only keep previous correlation for diff calculation
    # Instead of: correlation_matrix = torch.zeros(x, y, y).to(device)  # OOM!
    loss = torch.zeros(y, y).to(device)
    prev_corr = None

    for i in range(x):
        # Compute current correlation matrix (y, y)
        curr_corr = torch.nn.functional.cosine_similarity(
            flattened_tensor[i].unsqueeze(1),
            flattened_tensor[i].unsqueeze(0)
        )

        # Accumulate loss from difference with previous timestep
        if prev_corr is not None:
            loss += torch.abs(curr_corr - prev_corr)

        prev_corr = curr_corr

    return loss
