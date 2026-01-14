import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from Model import *
from torch import softmax
import numpy as np
import Model


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, target_dim, device, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.target_dim = target_dim
        self.device = device
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    target_dim = self.target_dim,
                    device =  self.device,
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        loss_z = []
        num = 0
        for layer in self.residual_layers:
            num += 1
            x, skip_connection,loss= layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
            loss_z.append(loss)
        
        loss_d = sum(loss_z)/num
        loss_d = loss_d.sum(dim = 0)
        loss_d = loss_d.sum(dim = 0)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x, loss_d


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,target_dim,device,is_linear=False):
        super().__init__()
        self.device = device  # Store device for later use
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.pool = nn.Linear(target_dim,1).to(device)
        #=================================
        self.dispatcher = Model.dispatcher(target_dim, 64, 3).to(device) #noise diapatcher
        self.mlp = Model.MLP(target_dim,64,target_dim).to(device)
        #=================================
        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)


        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        y = y.squeeze(dim = 0)
        y = y.reshape(channel*B, K ,L)
        y = y.transpose(1, 2)
        z = att_dot_var(y,self.pool)
        gcn = GraphConvolutionalNetwork(64,K).to(self.device)
        out = gcn(z)
        out = out.reshape(B, channel, K * L)

        return out

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        #dispatcher=========================
        y = y.squeeze(dim = 0)
        y = y.reshape(channel*B, K ,L)
        y = y.transpose(1, 2)
        invariant,variant,loss_d = self.dispatcher(y)
        print("invariant----",invariant.shape)
        print("variant----",invariant.shape)
        invariant = invariant.transpose(1, 2)
        invariant = invariant.unsqueeze(dim = 0)
        variant = variant.transpose(1, 2)
        variant = variant.unsqueeze(dim = 0)

        y_inv = self.forward_time(invariant, base_shape)
        y_inv = self.forward_feature(y_inv, base_shape)  # (B,channel,K*L)

        y_v = self.forward_time(variant, base_shape)
        y_v = self.forward_feature(y_v, base_shape)  # (B,channel,K*L)

        y_inv = y_inv.squeeze(dim = 0)
        y_inv = y_inv.reshape(channel*B, K, L)
        y_inv = y_inv.transpose(1, 2)

        y_v = y_v.squeeze(dim = 0)
        y_v = y_v.reshape(channel*B, K, L)
        y_v = y_v.transpose(1, 2)

        res = Model.integrate_patterns(y_inv,y_v,self.mlp)
        res = res.transpose(1, 2)
        res = res.unsqueeze(dim = 0)
        res = y_inv.reshape(B, channel, K * L)
        #==================================
        y = self.mid_projection(res)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip, loss_d


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super(GraphConvolutionalNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.W = nn.Linear(num_nodes, num_nodes)
        self.E = self.embedding

    def forward(self, z_delta):
        z_delta = z_delta.reshape(-1,self.num_nodes)
        A = torch.matmul(self.E, self.E.T)
        A = F.softmax(A, dim=-1)
        z = torch.matmul(z_delta, A)
        z = self.W(z)
        return z


def att_dot_var(x,pool):
    B,L,K = x.shape
    zt = pool(x)
    zt = zt.reshape(-1,1)
    zt = zt.transpose(0, 1)
    temp = x.reshape(-1,K)
    e = torch.matmul(zt, temp)
    e = e / np.sqrt(K)
    attention = F.softmax(e,dim = -1) #(1,140)
    out = temp* attention
    out = out.reshape(B,L,K)
    out = F.relu(out)
    return out