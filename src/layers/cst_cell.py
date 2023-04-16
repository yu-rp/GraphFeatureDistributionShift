
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

from src.layers.cell import *
from src.layers.cst_cell import *
from numpy.fft import fft, ifft
from scipy.fftpack import fftfreq
from math import sqrt



class FourierDecomp(nn.Module):
    def __init__(self, hist_len, in_dim, hid_dim, out_dim, n_nodes, freq_type, freq_threshold, device):
        super(FourierDecomp, self).__init__()
        '''
        input time series: [b, n, l, d]
        output c/e time seires: [b, n, l, d]
        '''
        self.fourier_layer = FourierCell(in_dim * n_nodes, out_dim* n_nodes, hist_len, freq_type, freq_threshold, device)
        self.mlp = nn.Sequential(nn.Conv1d(in_dim* n_nodes, hid_dim *2, 1),
                                        nn.GELU(),
                                        nn.Conv1d(hid_dim *2, out_dim* n_nodes, 1)
                                        ).to(device)
        
    def forward(self, ts_in):
        b, n, l, d = ts_in.shape
        ts_in = ts_in.permute(0,1,3,2).reshape(b, n*d, l)
        x1 = self.fourier_layer(ts_in)
        x2 = self.mlp(ts_in)
        x = x1 + x2
        ts_out = F.relu(x)
        return ts_out.reshape(b, n, d, l).permute(0,1,3,2)
    
    
class FourierCell(nn.Module):
    def __init__(self, in_channels, out_channels, t_len, freq_type, freq_threshold, device):
        super(FourierCell, self).__init__()
        """
        FFT -> linear transform -> Inverse FFT   
        """
        self.in_channels = in_channels # n * in_d
        self.out_channels = out_channels # n * out_d
        self.freq_type = freq_type
        self.freq_threshold = int(freq_threshold)
        self.scale = (1 / (in_channels*out_channels))
        if freq_type == 'high':
            self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, t_len//2 + 1 - self.freq_threshold, dtype=torch.cfloat)).to(device)
        elif freq_type == 'low':
            self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.freq_threshold, dtype=torch.cfloat)).to(device)
    def compl_mul(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, ts_in):
        x_ft = torch.fft.rfft(ts_in) # [1024, 34, 13]
        ts_out = torch.zeros(ts_in.shape[0], self.out_channels, ts_in.size(-1)//2 + 1, device=ts_in.device, dtype=torch.cfloat)
        if self.freq_type == 'low':
            ts_out[:, :, :self.freq_threshold] = self.compl_mul(x_ft[:, :, :self.freq_threshold], self.weights)
        elif self.freq_type == 'high':
            ts_out[:, :, self.freq_threshold:] = self.compl_mul(x_ft[:, :, self.freq_threshold:], self.weights)
        ts_out = torch.fft.irfft(ts_out, n=ts_in.shape[-1])
        return ts_out



#################### tcn    
class TemporalEnc(nn.Module):
    '''
    input (b, n, t, d)
    output (b, n, l-2, hid_dim)
    '''
    def __init__(self, in_dim, hid_dim, kernel_size=3):
        super(TemporalEnc, self).__init__()
        self.c_in = in_dim
        self.c_out = hid_dim
        self.res_conv = nn.Conv2d(in_dim, hid_dim, 1)
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_dim, hid_dim, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_dim, hid_dim, (1, kernel_size))

    def forward(self, x):
        # x [b, n, t, d]
        x = x.permute(0, 3, 1, 2)  # [b, c, num_nodes, t]
        x_input = self.res_conv(x)
        x_input = x_input[:, :, :, self.kernel_size - 1:]
        out = self.conv1(x) + x_input
        out = out * torch.sigmoid(self.conv2(x))
        out = out.permute(0, 2, 3, 1) #[b, n, l-2, d]
        return out


class SpatialAtten(nn.Module):
    '''
    input (b, n, l, d)
    output: a_c, a_e (b, n, n) *2
    '''
    def __init__(self, hid_dim, embed_dim, wo_causal_s, attn_threshold, block_num, seq_len, input_dim):
        super(SpatialAtten, self).__init__()
        self.attn_threshold = attn_threshold
        self.wo_causal_s = wo_causal_s
        self.embed_dim = embed_dim
        # layer
        l_t = seq_len - (block_num+1) * 2
        self.ts_proj = nn.Linear(l_t * hid_dim, embed_dim)
        
    def get_norm_attn(self, attn):
        '''
        attn: (b, n, n)
        '''
        # np.save('/home/yutong/STCausal/notebook/attn_0.npy', attn.cpu().detach().numpy())
        mask_c_bool = torch.ge(attn, self.attn_threshold)
        mask_c = mask_c_bool.int() * -np.inf
        mask_c = torch.where(torch.isnan(mask_c), torch.full_like(mask_c,1),mask_c).requires_grad_(False)
        maske_attn_c = attn * mask_c
        a_c = F.softmax(maske_attn_c, -1).detach()
        a_c = torch.where(torch.isnan(a_c), torch.full_like(a_c,0),a_c).requires_grad_(False)
        
        if not self.wo_causal_s:
            mask_e_bool = ~mask_c_bool
            mask_e = mask_e_bool.int() * -np.inf
            mask_e = torch.where(torch.isnan(mask_e), torch.full_like(mask_e,1),mask_e).requires_grad_(False)
            neg_attn = (torch.ones(attn.shape).to(attn.device) / attn)
            maske_attn_e = neg_attn * mask_e
            a_e = F.softmax(maske_attn_e, -1).detach()
            a_e = torch.where(torch.isnan(a_e), torch.full_like(a_e,0),a_e).requires_grad_(False)
        else: a_e = None
        
        return a_c, a_e
    
    def attention(self, queries, keys):
        scale = 1. / sqrt(self.embed_dim)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        A = torch.softmax(scale * scores, dim=-1)
        return A
        
    def forward(self, ts_enc):
        # ts_enc #[b, n, l_, d]
        b, n, l_, d = ts_enc.shape
        # print(ts_enc.shape)
        # input()
        ts_enc = self.ts_proj(ts_enc.reshape(b * n, -1)).reshape(b, n, self.embed_dim) # [b, n, 512]
        attn = self.attention(ts_enc, ts_enc) # [b, n, n]
        a_c, a_e = self.get_norm_attn(attn)
        return a_c, a_e
        # return attn, a_e
    
    
# class Prediction(nn.Module):
#     def __init__(self, pred_len, hid_dim, n_nodes, hid_dim_pred, output_dim):
#         super(Prediction, self).__init__()
#         self.output_dim = output_dim
#         self.pred_len = pred_len
#         self.hid_dim_pred = hid_dim_pred
#         self.fc_in = nn.Linear(hid_dim, hid_dim_pred)
#         self.fc_out = nn.Linear(hid_dim_pred, pred_len * output_dim)
#         self.ln = nn.LayerNorm([n_nodes ,hid_dim_pred])

#     def forward(self, hn):
#         # h: (b, n, hid_dim)
#         b, n, d = hn.shape
#         h = F.relu(self.fc_in(hn.reshape(b * n, d)))  
#         h = self.ln(h.reshape(b, n, self.hid_dim_pred))    
#         pred = self.fc_out(h.reshape(b * n, self.hid_dim_pred))
#         return pred.reshape(b, n, self.pred_len, self.output_dim).permute(0,2,1,3)



class CSTBlock(nn.Module):
    def __init__(self, seq_len, pred_len, hid_dim, n_nodes, input_dim, output_dim, pred_block, device, wo_causal_s, attn_threshold, spatial_embed_dim, block_num):
        super(CSTBlock, self).__init__()
        self.pred_block = pred_block
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.block_num = block_num
        self.wo_causal_s = wo_causal_s
        
        # layer
        if block_num == 0:
            self.temporal= TemporalEnc(input_dim, hid_dim, kernel_size=3)
        else:
            self.temporal= TemporalEnc(hid_dim, hid_dim, kernel_size=3)
        self.get_attention = SpatialAtten(hid_dim, spatial_embed_dim, wo_causal_s, attn_threshold, block_num, seq_len, input_dim)
        self.ln = nn.LayerNorm([n_nodes ,hid_dim])
        
        
    def spatial_agg(self, att, v):
        return torch.einsum("bln,bnsd->blsd", att, v)

    def forward(self, h):
        # h: [b, n, s, d]
        # b, n, s, d = h.shape
        h_ = self.temporal(h) # [512, n, l, d?]
        a_c, a_e = self.get_attention(h_)
        c = self.spatial_agg(a_c , h_)
        c = c + h_
        c = self.ln(c.permute(0, 2, 1, 3))# [b, l_, n, d]
        c = c.permute(0, 2, 1, 3)
        
        if not self.wo_causal_s:
            e = self.spatial_agg(a_e, h_)
            e = e.reshape(-1, e.shape[-1])
        else: e = None
        return c, e, a_c
