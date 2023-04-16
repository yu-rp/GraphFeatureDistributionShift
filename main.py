import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader_cast, check_device, get_num_nodes, get_num_edges, setup_seed
from src.utils.metrics import masked_mae
from src.models.cast import CaST
from src.trainers.cast_trainer import CaST_Trainer
from src.utils.args import get_public_config, str_to_bool
from src.utils.graph_algo import load_graph_data

def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='cast')
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--filter_type', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--wo_env_aug', type=str_to_bool, default=False, help='w/o apply augmentation on the environment representation')
    parser.add_argument('--wo_env', type=str_to_bool, default=False, help='w/o use the environment representation')
    parser.add_argument('--wo_s_edge', type=str_to_bool, default=False, help='w/o edge convolution')

    parser.add_argument('--edge_feat_flag', type=str_to_bool, default=True, help='use the edge feature to do the prediction?')
    
    # loss
    parser.add_argument('--beta1', type=float, default=1.0, help='contribution of commitment loss, between 0.1 and 2.0')
    parser.add_argument('--beta2', type=float, default=1.0, help='contribution of mutual information regulization loss')

    # temporal
    parser.add_argument('--depth', type=int, default=10, help='hp for temporal block')
    parser.add_argument('--n_envs', type=int, default=10, help='the number of environments')
    parser.add_argument('--aug_magnitude', type=float, default=0.1, help='the noise level of augmentation of the environment representation')
    
    # spatial
    parser.add_argument('--K', type=int, default=2, help='num of neigborhood')
    parser.add_argument('--bias', type=str_to_bool, default=True, help='whether to use bias')

    # training
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.7)
    
    # dataset
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--train_ratio',type=float ,default=8/12, help='The training set ratio')
    parser.add_argument('--val_ratio',type=float ,default=2/12, help='The validation set ratio')
    parser.add_argument('--train_val',type=str ,default='8_2_2', help='just for the name of the results')
    
    parser.add_argument('--time_delay_scaler', type=int, default=6, help='the rolling step on time series when calculating the time delay similarity')
    

    args = parser.parse_args()
    args.steps = [5,10,30,50]
    


    print(args)
    
    args.folder_name = 'woenv{}_woenva{}_wose{}_edgefeat{}/hid{}_dropout{}_lr{}_K{}_seed{}'.format(
                                                                args.wo_env,
                                                                args.wo_env_aug,
                                                                args.wo_s_edge,
                                                                args.edge_feat_flag,
                                                                args.hid_dim, 
                                                                args.dropout, 
                                                                args.base_lr,
                                                                args.K,
                                                                args.seed)
    args.log_dir = './logs/{}/{}_{}_{}_{}/{}/{}/'.format(args.dataset+ '_' + args.train_val,
                                             args.seq_len, args.horizon, args.input_dim, args.output_dim,
                                             args.model_name,
                                             args.folder_name)
    args.num_nodes = get_num_nodes(args.dataset)  
    args.num_edges = get_num_edges(args.dataset)  
                                       

    if args.filter_type == 'identity':
        args.support_len = 1
    else:
        args.support_len = 3

    args.datapath = os.path.join('./data', args.dataset)
    if args.dataset[:3] == 'AIR':
        args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower()[:6])
    else:
        args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower())
    if args.seed != 0:
        setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args

def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)
    
    model = CaST(name = args.model_name,
                dataset = args.dataset,
                device = device,
                num_nodes = args.num_nodes,
                num_edges = args.num_edges,
                seq_len = args.seq_len,
                horizon=args.horizon,
                input_dim=args.input_dim,
                output_dim=args.output_dim, 
                dropout=args.dropout,
                hid_dim = args.hid_dim,
                # wo_causal_t = args.wo_causal_t,
                wo_env_aug = args.wo_env_aug,
                wo_env = args.wo_env,
                # wo_s_node = args.wo_s_node,
                wo_s_edge = args.wo_s_edge,
                edge_feat_flag = args.edge_feat_flag,
                K = args.K, # num of neigborhood
                depth = args.depth,
                bias = args.bias,
                time_delay_scaler = args.time_delay_scaler,
                aug_magnitude = args.aug_magnitude,
                n_envs = args.n_envs,
                )

    dataloader = get_dataloader_cast(datapath = args.datapath,
                          batch_size = args.batch_size,
                          input_dim = args.input_dim,
                          output_dim = args.output_dim,
                          seq_length_x = args.seq_len,
                          seq_length_y = args.horizon,
                          interval = args.interval,
                          time_delay_scaler = args.time_delay_scaler,
                          train_ratio = args.train_ratio,
                          val_ratio = args.val_ratio,
                          )

    result_path = args.result_path + '/' + args.dataset + '_' + args.train_val + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    trainer = CaST_Trainer(model=model,
                            adj_mat=adj_mat,
                            filter_type=args.filter_type,
                            data=dataloader,
                            base_lr=args.base_lr,
                            lr_decay_ratio=args.lr_decay_ratio,
                            log_dir=args.log_dir,
                            n_exp=args.n_exp,
                            save_iter=args.save_iter,
                            clip_grad_value=args.max_grad_norm,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            device=device,
                            aug=args.aug,
                            steps=args.steps,
                            model_name = args.model_name,
                            result_path = result_path,
                            hp = args.folder_name,
                            # wo_causal_t = args.wo_causal_t,
                            wo_env = args.wo_env,
                            beta1 = args.beta1,
                            beta2 = args.beta2,

                            )

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')


if __name__ == "__main__":
    main()