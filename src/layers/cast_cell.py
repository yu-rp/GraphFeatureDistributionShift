import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat

import numpy as np

from src.layers.dilated_conv import DilatedConvEncoder
from src.layers.vq_functions import vq, vq_st

import os.path as osp
from torch_geometric.data import Dataset, download_url, Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_undirected, dense_to_sparse, coalesce
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dense_to_sparse

import dgl
from dgl.nn import EdgeWeightNorm


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)





class BandedFourierLayer(nn.Module):
# class CauDisentangler(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        # print(input.shape) # [64, 17, 34]
        # print(input[:, self.start:self.end].shape) # [64, 13, 34]
        # print(self.weight.shape) # [13, 16, 8]
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class TempDisentangler(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims, depth, dropout):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.repr_dropout = nn.Dropout(dropout)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(input_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(input_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x):  # x: B x T x input_dims

        # x: B x Co x T

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)

def vector_aug(vector, noise_level=0.1):
    """
    Adds Gaussian noise to a latent vector.
    :param vector: The latent vector to add noise to.
    :param noise_level: The standard deviation of the Gaussian noise to add.
    :return: The new latent vector with added noise.
    """
    noise = torch.normal(0, noise_level, vector.shape).to(vector.device)
    return vector + noise

class EnvEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # codebook
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, x): # x [b, l, h_d]
        # x_ = x.permute(0, 2, 1).contiguous()
        x_ = x.contiguous()
        latents = vq(x_, self.embedding.weight)
        return latents
    
    def straight_through(self, z_e_x):# x [b, h_d]
        '''
        z_e_x: the latent vectors for environments
        '''
        z_e_x_ = z_e_x.contiguous()
        # get the feature from the codebook and its index
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach()) # z_q_x_: [b, h_d]    indices:[b]
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar, indices

    def straight_through_test(self, z_e_x):# the index is soft
        inputs = z_e_x.contiguous()
        codebook = self.embedding.weight.detach()

        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0) 
            # get the index
            indices = torch.softmax(distances, dim=1)    
            # compute the env vector
            codes_flatten = torch.mm(indices, codebook)
            codes = codes_flatten.view_as(inputs)

            return codes.contiguous(), None

## spatial
class HodgeLaguerreConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                    weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.reshape(xshape[0],-1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')

# write a GCN class with K-hop message passing


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.K = K

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, input, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print(edge_index.shape)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(input)
        x_res = x
        for _ in range(self.K):
            # Step 3-5: Start propagating messages.
            x = self.propagate(edge_index, x=x, norm=edge_weight)
            x = F.relu(x)

        # Step 6: Apply a final bias vector.
        out = self.bias + x_res + x

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



