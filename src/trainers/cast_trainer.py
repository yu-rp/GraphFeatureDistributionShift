import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
import torch.nn.functional as F

from src.utils.logging import get_logger
from src.base.trainer import BaseTrainer
from src.utils import graph_algo
from src.utils.metrics import masked_rmse
from src.utils import metrics as mc
# from src.utils.helper import mutual_info
# from sklearn.metrics import mutual_info_score

class CaST_Trainer(BaseTrainer):
    def __init__(self,  
                #  wo_causal_t,
                 wo_env,
                 beta1,
                 beta2,
                 **args):
        super(CaST_Trainer, self).__init__(**args)
        
        # self.wo_causal_t = wo_causal_t
        # self.wo_env = wo_env
        self.beta1 = beta1
        self.beta2 = beta2
        self.mi_regulization = nn.CrossEntropyLoss()

    def _calculate_supports(self, adj_mat, filter_type):

        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        if filter_type == "identity":
            supports = np.diag(np.ones(new_adj.shape[0])).astype(np.float32)
            supports = Tensor(supports).cuda()
        else:
            scaled_adj = graph_algo.calculate_scaled_laplacian(new_adj).todense()
            cheb_poly_adj = graph_algo.calculate_cheb_poly(scaled_adj, 3)
            supports = Tensor(cheb_poly_adj).cuda()
        return supports
    

    def train_batch(self, X, label, iter):
        self.optimizer.zero_grad()
        ###### need to figure out why have one more dimension
        label = label.squeeze(1) 
        ###################################################
        # if self.wo_env:
        #     pred = self.model(X)
        #     pred, label = self._inverse_transform([pred, label])
        #     loss = self.loss_fn(pred, label, 0.0)
        # else:

        pred, h_node_env, h_node_env_q, env_ind, env_cla_pred = self.model(X)
        pred, label = self._inverse_transform([pred, label])
        # prediction loss
        loss_pred = self.loss_fn(pred, label, 0.0)
        # Vector quantization objective
        loss_vq = F.mse_loss(h_node_env,h_node_env_q)
        # Commitment objective
        loss_commit = F.mse_loss(h_node_env_q, h_node_env)
        # mutual info for env and causal
        loss_mi = - self.mi_regulization(env_cla_pred, env_ind)
        
        # loss = loss_pred + loss_vq + self.beta * loss_commit + 0.5 * loss_mi
        loss = loss_pred + loss_vq + self.beta1 * loss_commit + self.beta2 * loss_mi
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()
    
    def test_batch(self, X, label):
        ###### need to figure out why have one more dimension
        label = label.squeeze(1) 
        ###################################################
        # if self.wo_env:
        #     pred = self.model(X, test_flag=True)
        # else:
        pred, _, _, _, _ = self.model(X, test_flag=True)
        pred, label = self._inverse_transform([pred, label])
        return pred, label
    