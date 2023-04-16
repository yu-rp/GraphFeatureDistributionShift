from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.data import DataLoader
from src.utils.dataset import Dataset_CaST, Dataset_CaST_processed
import torch
from torch import Tensor
import logging
import numpy as np
import pandas as pd
import os
import sys
import pickle
import random

import torch_geometric

from src.utils.scaler import StandardScaler

def get_dataloader(datapath, batch_size,input_dim, output_dim, mode='train'):
    data = {}
    processed = {}
    results = {}
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scalers = []
    for i in range(output_dim):
        scalers.append(StandardScaler(mean=data['x_train'][..., i].mean(),
                                      std=data['x_train'][..., i].std()))

    # Data format
    for category in ['train', 'val', 'test']:
        # normalize the target series (generally, one kind of series)
        
        for i in range(output_dim):
            data['x_' + category][..., i] = scalers[i].transform(data['x_' + category][..., i])
            data['y_' + category][..., i] = scalers[i].transform(data['y_' + category][..., i])

        new_x = Tensor(data['x_' + category])[..., :input_dim]
        new_y = Tensor(data['y_' + category])[..., :output_dim]

        processed[category] = TensorDataset(new_x, new_y)

    results['train_loader'] = DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = DataLoader(processed['test'], batch_size, shuffle=False)

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers
    return results


def get_dataloader_cast(datapath, batch_size, input_dim, output_dim, seq_length_x, seq_length_y, interval, time_delay_scaler, train_ratio, val_ratio):
    processed = {}
    results = {}
    
    #### scaler
    scaler_dir = os.path.join(datapath, 'scaler.pkl')
    if not os.path.exists(scaler_dir):
        scalers = []
        # data = np.load(os.path.join(datapath, 'train.npz')['data'])
        data = np.load(os.path.join(datapath, 'dataset.npy'))
        for i in range(output_dim):
            scalers.append(StandardScaler(mean=data[..., i].mean(),
                                        std=data[..., i].std()))
        with open(scaler_dir, 'wb') as f:
            pickle.dump(scalers, f)
    else:
        with open(scaler_dir, 'rb') as f:
            scalers = pickle.load(f)
    
    
    #### dataset
    processed_dir = os.path.join(datapath, 'processed')
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
        dataFile = os.path.join(datapath, 'dataset.npy')
        processed_dataset = Dataset_CaST(dataFile, datapath, scalers, input_dim, output_dim, seq_length_x, seq_length_y, interval, time_delay_scaler)
        dataloader = torch_geometric.loader.DataLoader(processed_dataset, batch_size = 1, shuffle=False)
        
        for i, data in enumerate(dataloader):
            [graph,  y] = data
            data_zip = {'graph':graph,
                        'y': y}
            torch.save(data_zip, os.path.join(processed_dir, 'Graph'+str(i)+'.pt'))
    
    num_samples = len(os.listdir(processed_dir)) - 3 # minus three label.csv
    print('num_samples: {}'.format(num_samples))

    idx_train = round(num_samples * train_ratio)
    idx_val = round(num_samples * (val_ratio+train_ratio))
    # _test = num_samples - num_train - num_val
    idx_list = [0, idx_train, idx_val, num_samples]
    
    for i, category in enumerate(['train', 'val', 'test']):
        df_ind = pd.DataFrame(columns=['sample_name'])
        df_ind['sample_name'] = ['Graph'+str(fileidx)+'.pt' for fileidx in range(idx_list[i], idx_list[i+1])]
        indexFile = os.path.join(processed_dir, '{}_index.csv'.format(category))
        df_ind.to_csv(indexFile)
        processed[category] = Dataset_CaST_processed(processed_dir, indexFile)
        
    # # Data format
    # for category in ['train', 'val', 'test']:
    #     dataFile = os.path.join(datapath, category + '.npz')
    #     adjFile = datapath
    #     processed[category] = Dataset_CaST(dataFile, adjFile, scalers, input_dim, output_dim, seq_length_x, seq_length_y, interval, time_delay_scaler)

    results['train_loader'] = torch_geometric.loader.DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = torch_geometric.loader.DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = torch_geometric.loader.DataLoader(processed['test'], batch_size, shuffle=False)

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers
    return results

def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def get_num_nodes(dataset):
    print(dataset)
    # d = {'AIR_BJ': 34, 'AIR_BJ_CO': 34, 'AIR_BJ_10_1_1': 34, 'AIR_BJ_CO_10_1_1': 34, 
    #      'AIR_GZ': 41, 'AIR_GZ_CO': 41, 'AIR_GZ_10_1_1': 41, 'AIR_GZ_CO_10_1_1': 41, }
    d = {'AIR_BJ': 34,  
        'AIR_GZ': 41 }
    assert dataset[:6] in d.keys()
    return d[dataset[:6]]

def get_num_edges(dataset):
    print(dataset)
    # d = {'AIR_BJ': 34, 'AIR_BJ_CO': 34, 'AIR_BJ_10_1_1': 34, 'AIR_BJ_CO_10_1_1': 34, 
    #      'AIR_GZ': 41, 'AIR_GZ_CO': 41, 'AIR_GZ_10_1_1': 41, 'AIR_GZ_CO_10_1_1': 41, }
    d = {'AIR_BJ': 82,  
        'AIR_GZ': 41 }
    assert dataset[:6] in d.keys()
    return d[dataset[:6]]






