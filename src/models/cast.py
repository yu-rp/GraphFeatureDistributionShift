
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

from src.layers.cell import *
from src.layers.cast_cell import *
from numpy.fft import fft, ifft
from scipy.fftpack import fftfreq
from math import sqrt
from src.layers.dilated_conv import DilatedConvEncoder
import math
from torch_geometric.utils import normalized_cut

class CaST(BaseModel):
    def __init__(self,
                 num_edges,
                 dropout=0.3,
                 hid_dim = 16,
                 wo_env = False,
                 wo_env_aug = False,
                 wo_s_node = False,
                 wo_s_edge = False,
                 edge_feat_flag = True,
                 K = 3,
                 depth = 10,
                 bias = True,
                 n_envs = 10,
                 time_delay_scaler = 6,
                 aug_magnitude = 0.1,
                 **args):
        super(CaST, self).__init__(**args)
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.wo_env_aug = wo_env_aug
        self.wo_env = wo_env
        self.wo_s_node = wo_s_node
        self.wo_s_edge = wo_s_edge
        self.edge_feat_flag = edge_feat_flag
        self.n_node = self.num_nodes
        self.n_edge = num_edges
        self.n_envs = n_envs
        self.aug_magnitude = aug_magnitude
            
        self.start_mlp_edge = nn.Linear(2 + self.seq_len//time_delay_scaler, hid_dim)
        
        ## temperoal 
        self.start_encoder = DilatedConvEncoder(in_channels=self.input_dim, channels=[self.input_dim] * depth + [self.hid_dim], kernel_size=3)
        
        t_kernels = [2**i for i in range(int(math.log2(self.seq_len//2)))]
        self.temporal = TempDisentangler(input_dims =self.hid_dim, 
                                         output_dims =self.hid_dim*2,
                                            kernels = t_kernels,
                                            length = self.seq_len,
                                            hidden_dims=self.hid_dim,
                                            depth= depth,
                                            dropout = dropout)
        
        self.codebook = EnvEmbedding(n_envs, hid_dim)
        
        self.t_proj_env = nn.Linear(self.seq_len, 1)
        self.t_proj_cau = nn.Linear(self.seq_len, 1)
        
        self.spatial_edge = HodgeLaguerreConv(in_channels=hid_dim, out_channels=hid_dim, K=K, bias=bias)
        self.edge_causal = nn.Linear(hid_dim, 1)
        self.edge_proj = nn.Linear(self.n_edge, self.n_node)

        self.spatial_node = GCNConv(in_channels=hid_dim, out_channels=hid_dim, K=K)

        # mutual info regulization
        self.env_cla = nn.Sequential(nn.Linear(hid_dim, self.hid_dim),
                                    nn.ReLU(),
                                    # nn.Dropout(dropout),
                                    nn.Linear(self.hid_dim, n_envs),
                                    nn.Softmax(dim = 1)
                                    )
        
        if wo_env or wo_s_edge:
            end_indim = hid_dim * 2
        else:
            if self.edge_feat_flag:
                end_indim = hid_dim * 3
            else:
                end_indim = hid_dim * 2
        self.end_mlp = nn.Sequential(nn.Linear(end_indim, end_indim*2),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(end_indim*2, self.horizon * self.output_dim)
                                    )


    
    def forward(self, X, test_flag=False):
        '''
        input:  #### edge #####
                graph.x_s                [5248, 6] [64 * 82, 6]
                graph.edge_index_s       [2, 98688] 
                graph.edge_weight_s      [98688]

                #### node #####
                graph.x_t                [2176, 24, 1]
                graph.edge_index_t       [2, 12160]
                graph.edge_weight_t      [12160]
        '''

        
        x_link, edge_index_link, edge_weight_link = X.x_s, X.edge_index_s, X.edge_weight_s # edge
        x_node, edge_index_node, edge_weight_node = X.x_t, X.edge_index_t, X.edge_weight_t # node

        edge_index = X.edge_index # [2, 5248]

        b, l, d = x_node.shape
        batch_size = b//self.n_node
        
        # h_node = self.start_mlp_node(x_node.float()) # [2176, 24, 1] --> [2176, 24, 32]
        
        #############   edge
        if self.wo_s_edge:
            norm_causal_score = torch.ones((edge_index.size(1)), device=edge_index.device)
        else:
            # update the edge feature to recogenize the causal score 
            h_link = self.start_mlp_edge(x_link.float()) # [4352, 6] --> [4352, 32]
            h_link_updated = self.spatial_edge(h_link, edge_index_link, edge_weight_link) #[5248, 32] --> [5248, 32]
            norm_causal_score = self.edge_causal(h_link_updated).squeeze()
            if self.edge_feat_flag:
                # project the link feature for prediction
                h_link_proj = self.edge_proj(h_link_updated.reshape(batch_size, self.n_edge, -1).permute(0,2,1)).permute(0,2,1) #[b, n_nodes, 32]

        #############   node
        # if self.wo_causal_t:
        #     h_node = self.start_mlp_node(x_node.float().reshape(b, l*d)) # [2176, 24 * 1] --> [2176, 32]
        #     h_node_cau = self.spatial_node(h_node_cau, edge_index, norm_causal_score) #[2176, 32] -> [2176, 32]
            
        # else:
        h_node = self.start_encoder(x_node.float().permute(0,2,1)) #[2176, 24, 1] --> [2176, 1, 24] --> [2176, 32, 24]
        h_node_env, h_node_cau = self.temporal(h_node) # [2176, 32, 24] --> [2176, 24, 16] *2 
        
        ## enviroment
        h_node_env = self.t_proj_env(h_node_env.permute(0,2,1)).squeeze() # [2176, 24, 32] --> [2176, 32]
        if not test_flag: 
            h_node_env_st, h_node_env_q, env_ind = self.codebook.straight_through(h_node_env) # [2176, 32] --> [2176, 32]*2
        else: 
            h_node_env_st, h_node_env_q = self.codebook.straight_through_test(h_node_env)
            env_ind = None
        if not self.wo_env_aug:
            h_node_env_st = vector_aug(h_node_env_st, self.aug_magnitude)
        
        ## causal
        # reduce the time dimension
        h_node_cau = self.t_proj_cau(h_node_cau.permute(0,2,1)).squeeze() # [2176, 24, 32] --> [2176, 32]
        # update the node representation based on the causal score
        h_node_cau = self.spatial_node(h_node_cau, edge_index, norm_causal_score) #[2176, 32] -> [2176, 32]

        ## get the final node representation
        if self.wo_env:
            h_out_node = torch.cat([h_node_cau.reshape(batch_size, self.n_node, -1), h_link_proj],dim=-1) 
        elif self.wo_s_edge:
            h_out_node = torch.cat([h_node_env_st.reshape(batch_size, self.n_node, -1), h_node_cau.reshape(batch_size, self.n_node, -1)],dim=-1)
        else:
            if self.edge_feat_flag:
                h_out_node = torch.cat([h_node_env_st.reshape(batch_size, self.n_node, -1), h_node_cau.reshape(batch_size, self.n_node, -1), h_link_proj],dim=-1)  # [2176, 64]
            else: 
                h_out_node = torch.cat([h_node_env_st.reshape(batch_size, self.n_node, -1), h_node_cau.reshape(batch_size, self.n_node, -1)],dim=-1)  # [2176, 64]
        pred = self.end_mlp(h_out_node) #[64, 34, 32*2] --> #[64, 34, 24]

        ## get the mutual information regulization
        env_cla_pred = self.env_cla(h_node_cau)
        
        return pred.permute(0, 2, 1).reshape(batch_size,l,self.n_node,self.output_dim), h_node_env, h_node_env_q, env_ind, env_cla_pred
    
    
    
