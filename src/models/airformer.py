import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from src.base.model import BaseModel
from src.layers.embedding import AirEmbedding
import numpy as np
import time


class LatentLayer(nn.Module):
    def __init__(self, hidden_dim, latent_dim_in, latent_dim_out, middle_dim, num_layers=2):
        super(LatentLayer, self).__init__()

        self.num_layers = num_layers

        # self.enc_in = nn.Sequential(nn.Conv2d(hidden_dim+latent_dim_in, middle_dim, 1),
        #                             nn.BatchNorm2d(middle_dim))
        self.enc_in = nn.Sequential(
            nn.Conv2d(hidden_dim+latent_dim_in, middle_dim, 1))

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(middle_dim, middle_dim, 1))
            # layers.append(nn.BatchNorm2d(middle_dim))
            layers.append(nn.ReLU(inplace=True))
        self.enc_hidden = nn.Sequential(*layers)
        self.enc_out_1 = nn.Conv2d(middle_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv2d(middle_dim, latent_dim_out, 1)

    def forward(self, x):
        # x: [b, c, n, t]
        h = self.enc_in(x)
        for i in range(self.num_layers):
            h = self.enc_hidden[i](h)
        mu = torch.minimum(self.enc_out_1(h), torch.ones_like(h)*10)
        sigma = torch.minimum(self.enc_out_2(h), torch.ones_like(h)*10)
        return mu, sigma


class GenerativeModel(nn.Module):
    def __init__(self, hidden_channels, latent_channels, num_blocks=4):
        super(GenerativeModel, self).__init__()

        self.layers = nn.ModuleList()

        # the top n-1 layers
        for _ in range(num_blocks-1):
            self.layers.append(
                LatentLayer(hidden_channels,
                            latent_channels,
                            latent_channels,
                            latent_channels,
                            2))
        # the button layer
        self.layers.append(
            LatentLayer(hidden_channels,
                        0,
                        latent_channels,
                        latent_channels,
                        2))

    def reparameterize(self, mu, sigma):
        # std = torch.exp(logsigma)
        # eps = torch.randn_like(std)
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps*sigma

    def forward(self, d):
        # d: [num_blocks, b, c, n, t]

        # top-down
        _mu, _logsigma = self.layers[-1](d[-1])  # [b, ]
        _sigma = torch.exp(_logsigma) + 1e-3
        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]

        for i in reversed(range(len(self.layers)-1)):
            _mu, _logsigma = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            _sigma = torch.exp(_logsigma) + 1e-3
            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))

        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        # print(mus.mean())
        # print(sigmas.mean())
        return z, mus, sigmas


class InferenceModel(GenerativeModel):
    def __init__(self, hidden_channels, latent_channels, num_blocks):
        super(InferenceModel, self).__init__(
            hidden_channels, latent_channels, num_blocks)

    def forward(self, d, mu_p, sigma_p):
        # d: [num_blocks, b, c, n, t]

        # top-down
        mu_q_hat, logsigma_q_hat = self.layers[-1](d[-1])
        sigma_q_hat = torch.exp(logsigma_q_hat) + 1e-3
        sigma_q = 1 / (torch.pow(sigma_q_hat, -2) +
                       torch.pow(sigma_p[-1], -2) + 1e-3)
        mu_q = sigma_q*(mu_q_hat * torch.pow(sigma_q_hat, -2) +
                        mu_p[-1] * torch.pow(sigma_p[-1], -2))
        sigmas = [sigma_q]
        mus = [mu_q]
        z = [self.reparameterize(mu_q, sigma_q)]

        for i in reversed(range(len(self.layers)-1)):
            mu_q_hat, logsigma_q_hat = self.layers[i](
                torch.cat((d[i], z[-1]), dim=1))
            sigma_q_hat = torch.exp(logsigma_q_hat) + 1e-3
            sigma_q = 1 / (torch.pow(sigma_q_hat, -2) +
                           torch.pow(sigma_p[i], -2) + 1e-3)
            mu_q = sigma_q*(mu_q_hat * torch.pow(sigma_q_hat, -2) +
                            mu_p[i] * torch.pow(sigma_p[i], -2))
            mus.append(mu_q)
            sigmas.append(sigma_q)
            z.append(self.reparameterize(mu_q, sigma_q))

        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return z, mus, sigmas


class AirFormer(BaseModel):
    def __init__(self,
                 dropout=0.3,
                 supports_len=2,
                 spatial_flag=True,
                 stochastic_flag=True,
                 hidden_channels=32,
                 end_channels=512,
                 blocks=4,
                 mlp_expansion=2,
                 path_assignment='data/local_partition/assignment.npy',
                 path_mask='data/local_partition/mask.npy',
                 **args):
        super(AirFormer, self).__init__(**args)
        self.dropout = dropout
        self.blocks = blocks
        self.transformer_flag = spatial_flag
        self.stochastic_flag = stochastic_flag
        self.assignment = torch.from_numpy(
            np.load(path_assignment)).float().to(self.device)
        self.mask = torch.from_numpy(
            np.load(path_mask)).bool().to(self.device)

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.embedding_air = AirEmbedding()

        self.start_conv = nn.Conv2d(in_channels=6,  # air quality attributes
                                    out_channels=hidden_channels,
                                    kernel_size=(1, 1))

        self.supports_len = supports_len
        for b in range(blocks):
            window_size = self.seq_len // 2 ** (blocks - b - 1)
            print('ws=', window_size)
            self.t_modules.append(TemporalTransformer(hidden_channels, depth=1, heads=2,
                                                      window_size=window_size,
                                                      mlp_dim=hidden_channels*mlp_expansion,
                                                      num_time=self.seq_len))

            if self.transformer_flag:
                self.s_modules.append(
                    SpatialTransformer(hidden_channels, 1, 2, hidden_channels*mlp_expansion, self.num_nodes,
                                       self.assignment, self.mask, dropout=dropout))
            else:
                self.residual_convs.append(nn.Conv1d(in_channels=hidden_channels,
                                                     out_channels=hidden_channels,
                                                     kernel_size=(1, 1)))

            self.bn.append(nn.BatchNorm2d(hidden_channels))

        # self.E = nn.Parameter(torch.randn(hidden_channels, self.num_nodes))  # spatial embedding
        # self.E_conv2d = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels*2, 1, 1, 0),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(hidden_channels*2, hidden_channels*blocks*2, 1, 1, 0))

        if self.stochastic_flag:
            self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels*blocks*2,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)
        else:
            self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels*blocks,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.horizon*self.output_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        if stochastic_flag:
            self.generative_model = GenerativeModel(
                hidden_channels, hidden_channels, blocks)
            # self.inference_model = InferenceModel(
            #     hidden_channels, hidden_channels, blocks)
            self.inference_model = GenerativeModel(
                hidden_channels, hidden_channels, blocks)

            # self.time_compression = nn.Sequential(nn.Linear(self.seq_len, 64),
            #                                       nn.ReLU(),
            #                                       nn.Dropout(dropout),
            #                                       nn.Linear(64, self.output_dim))

            self.reconstruction_model = \
                nn.Sequential(nn.Conv2d(in_channels=hidden_channels*blocks,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(in_channels=end_channels,
                                        out_channels=self.input_dim,
                                        kernel_size=(1, 1),
                                        bias=True)
                              )

    def forward(self, inputs, supports=None):
        x_embed = self.embedding_air(inputs[..., 11:15].long())
        inputs = torch.cat((inputs[..., :11], x_embed, inputs[..., 15:]), -1)
        x = inputs[..., :6]  # [b, t, n, c]
        ext = inputs[..., 6:]

        x = x.permute(0, 3, 2, 1)  # [b, c, n, t]
        x = self.start_conv(x)
        d = []  # deterministic states
        for i in range(self.blocks):
            # residual = x

            if self.transformer_flag:
                x = self.s_modules[i](x, ext)
            else:
                x = self.residual_convs[i](x)

            x = self.t_modules[i](x)  # [b, c, n, t]

            # x = x + residual

            x = self.bn[i](x)
            d.append(x)

        d = torch.stack(d)  # [num_blocks, b, c, n, t]
        if self.stochastic_flag:
            # generatation and inference
            d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[..., :-1])
                       for i in range(len(d))]
            d_shift = torch.stack(d_shift)  # [num_blocks, b, c, n, t]

            z_p, mu_p, sigma_p = self.generative_model(d_shift)
            z_q, mu_q, sigma_q = self.inference_model(d)
            # z_q, mu_q, sigma_q = self.inference_model(d, mu_p, sigma_p)

            # compute KL divergence
            # if (sigma_p <= 0).sum() > 0:
            #     print('problem: ', sigma_p[sigma_p < 0])
            #     print('problem: ', sigma_p[sigma_p == 0])
            #     input()

            # if (torch.isnan(sigma_p)).sum() > 0:
            #     print('problem: ', (torch.isnan(sigma_p)).sum())
            #     input()
            logf = open("error_v5.log", "w")
            try:
                p = torch.distributions.Normal(mu_p, sigma_p)
                q = torch.distributions.Normal(mu_q, sigma_q)
            except Exception as e:  # most generic exception you can catch
                logf.write("Failed to run: {0}\n".format(str(e)))
                torch.save(d, 'd.pt')
                torch.save(z_p, 'z_p.pt')
                torch.save(mu_p, 'mu_p.pt')
                torch.save(sigma_p, 'sigma_p.pt')
                torch.save(z_q, 'z_q.pt')
                torch.save(mu_q, 'mu_q.pt')
                torch.save(sigma_q, 'sigma_q.pt')
                print(self.state_dict())
                torch.save(self.state_dict(), 'model.pt')
            finally:
                pass

            kl_loss = torch.distributions.kl_divergence(q, p).mean()  # take care of the order

            # reshaping
            num_blocks, B, C, N, T = d.shape
            z_p = z_p.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]
            z_q = z_q.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]

            # reconstruction
            x_rec = self.reconstruction_model(z_p)  # [b, c, n, t]
            x_rec = x_rec.permute(0, 3, 2, 1)

            # make prediction
            # (b, 256, 1085, 24) --> (b, 256, 1085, 1)
            # x_hat = self.time_compression(z_q)
            # x_hat = F.relu(x_hat)
            # x_hat = F.relu(self.end_conv_1(x_hat))
            # x_hat = self.end_conv_2(x_hat)

            # make prediction
            # (b, 256, 1085, 24) --> (b, 256, 1085, 1)
            # x_hat = z_q[..., -1:]

            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]
            # tmp = self.E_conv2d(self.E.unsqueeze(0).unsqueeze(-1))
            x_hat = torch.cat([d[..., -1:], z_q[..., -1:]], dim=1)
            x_hat = F.relu(self.end_conv_1(x_hat))
            # x_hat = F.relu(self.end_conv_1(x_hat + tmp))
            x_hat = self.end_conv_2(x_hat)

            return x_hat, x_rec, kl_loss

        else:
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]
            x_hat = F.relu(d[..., -1:])
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout=0.,
                 num_sector=17, assignment=None, mask=None, ext_dim=21, ext_flag=False):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sector = num_sector
        self.assignment = assignment
        self.mask = mask

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_sector, dim))
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sector))
        self.ext_flag = ext_flag
        self.beta = 1e-2
        if ext_flag:
            self.ext2attn = nn.Sequential(nn.Linear(ext_dim, 64),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(0.5),
                                          nn.Linear(64, heads*num_sector),)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, ext):
        # x: [b, n, c]
        # ext: [b, n, d]
        # assignment: [n, n, num_sector]
        # mask: [n, num_sector]

        B, N, C = x.shape

        # query: [bn, 1, c]
        # key/value target: [bn, num_sector, c]
        # [b, n, num_sector, c]

        pre_kv = torch.einsum('bnc,mnr->bmrc', x, self.assignment)

        # Matmul version
        # pre_kv = x.permute(0, 2, 1) # [b, 1, c, n]
        # a = self.assignment.unsqueeze(1) # [n, n, num_sector]
        # pre_kv = torch.matmul(pre_kv, a) # [b, n, c, num_sector]
        # pre_kv = pre_kv.permute(0, 1, 3, 2) # [b, n, num_sector, c]

        # # sparse version
        # tmp = x.permute(1, 2, 0).reshape(N, -1) # [n, cb]
        # # assignment is [n*num_sector, n]
        # pre_kv = torch.sparse.mm(self.assignment, tmp) # [n*num_sector, cb]
        # pre_kv = pre_kv.reshape(N, self.num_sector, C, B).permute(3, 0, 1, 2) # [b, n, num_sector, c]

        pre_kv = pre_kv.reshape(-1, self.num_sector, C)  # [bn, num_sector, c]
        # pre_kv += self.pos_embedding # [bn, num_sector, c]
        pre_q = x.reshape(-1, 1, C)  # [bn, 1, c]

        q = self.q_linear(pre_q).reshape(B*N, -1, self.num_heads, C //
                                         self.num_heads).permute(0, 2, 1, 3)  # [bn, num_heads, 1, c//num_heads]
        kv = self.kv_linear(pre_kv).reshape(B*N, -1, 2, self.num_heads,
                                            C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [bn, num_heads, num_sector, c//num_heads]

        # merge key padding and attention masks
        # [bn, num_heads, 1, num_sector]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # non-broadcast version
        # mask = self.mask.reshape(1, N, 1, 1, self.num_sector).repeat(
        #     B, 1, self.num_heads, 1, 1)
        # # [bn, num_heads, 1, num_sector]
        # mask = mask.reshape(-1, self.num_heads, 1, self.num_sector)
        # # [bn, num_heads, 1, num_sector]

        # broadcast version
        attn = attn.reshape(B, N, self.num_heads, 1,
                            self.num_sector) + self.relative_bias
        if self.ext_flag:
            ext_impact = self.ext2attn(ext).reshape(*attn.shape)
            attn += self.beta * ext_impact
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)

        # masking
        attn = attn.masked_fill_(mask, float(
            "-inf")).reshape(B * N, self.num_heads, 1, self.num_sector)

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, num_nodes, assignment, mask, dropout=0.):
        super().__init__()
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_nodes, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim, heads=heads, dropout=dropout,
                                 assignment=assignment, mask=mask,
                                 ext_dim=21, ext_flag=True),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, ext):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)  # [b*t, n, c]
        # x = x + self.pos_embedding  # [b*t, n, c]
        for attn, ff in self.layers:
            x = attn(x, ext) + x
            x = ff(x) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # self.mask = torch.tril(torch.ones(window_size, window_size))
        self.mask = torch.tril(torch.ones(window_size, window_size)).cuda()

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads,
                                  window_size=window_size, dropout=dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)  # [b*n, t, c]
        x = x + self.pos_embedding  # [b*n, t, c]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x

# For testing


def Air():
    return AirFormer(dropout=0.3,
                     supports_len=2,
                     spatial_flag=True,
                     hidden_channels=32,
                     end_channels=32 * 8,
                     name='airformer',
                     dataset='AIR_72',
                     device=None,
                     num_nodes=1085,
                     seq_len=24,
                     horizon=24,
                     input_dim=27,
                     output_dim=1)


if __name__ == '__main__':
    x = torch.rand(8, 24, 1085, 16).fill_(1)
    model = Air()
    y = model(x)
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    flops = FlopCountAnalysis(model, x)
    print('flops: {:.3f} G'.format(flops.total() / 1e9))
    counter = flops.by_module().most_common()
    for c in counter:
        print('{} FLOPs: {} G'.format(c[0], c[1] / 1e9))
    print('Params: {}'.format(parameter_count_table(model)))
    print(y.shape)
