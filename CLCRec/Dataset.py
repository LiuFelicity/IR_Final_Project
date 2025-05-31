import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_load(dataset, has_v=True, has_a=True, has_t=True):
    dir_str = './Data/' + dataset

    # Check for required files
    required_files = [
        'train.npy', 'val_full.npy', 'val_warm.npy', 'val_cold.npy',
        'test_full.npy', 'test_warm.npy', 'test_cold.npy'
    ]
    if has_v:
        required_files.append('feat_v.npy')
    if has_a:
        required_files.append('feat_a.npy')
    if has_t:
        required_files.append('feat_t.npy')

    for file in required_files:
        if not os.path.exists(os.path.join(dir_str, file)):
            if file in ['feat_a.npy', 'feat_t.npy', 'feat_v.npy']:
                print(f"Optional file '{file}' is missing in directory '{dir_str}', skipping.")
                continue
            raise FileNotFoundError(f"Required file '{file}' is missing in directory '{dir_str}'")

    train_data = np.load(dir_str+'/train.npy', allow_pickle=True)
    val_data = np.load(dir_str+'/val_full.npy', allow_pickle=True)
    val_warm_data = np.load(dir_str+'/val_warm.npy', allow_pickle=True)
    val_cold_data = np.load(dir_str+'/val_cold.npy', allow_pickle=True)
    test_data = np.load(dir_str+'/test_full.npy', allow_pickle=True)
    test_warm_data = np.load(dir_str+'/test_warm.npy', allow_pickle=True)
    test_cold_data = np.load(dir_str+'/test_cold.npy', allow_pickle=True)

    v_feat = None
    a_feat = None
    t_feat = None

    if dataset == 'movielens':
        num_user = 55485
        num_item = 5986
        num_warm_item = 5119
        if has_v and os.path.exists(os.path.join(dir_str, 'feat_v.npy')):
            v_feat = torch.tensor(np.load(dir_str+'/feat_v.npy', allow_pickle=True), dtype=torch.float).to(device)
        if has_a and os.path.exists(os.path.join(dir_str, 'feat_a.npy')):
            a_feat = torch.tensor(np.load(dir_str+'/feat_a.npy', allow_pickle=True), dtype=torch.float).to(device)
        if has_t and os.path.exists(os.path.join(dir_str, 'feat_t.npy')):
            t_feat = torch.tensor(np.load(dir_str+'/feat_t.npy', allow_pickle=True), dtype=torch.float).to(device)

    elif dataset == 'amazon':
        num_user = 27044
        num_item = 86506
        num_warm_item = 68810
        if has_v and os.path.exists(os.path.join(dir_str, 'feat_v.pt')):
            v_feat = torch.load(dir_str+'/feat_v.pt').to(device)

    elif dataset == 'tiktok':
        num_user = 32309
        num_item = 57832+8624
        num_warm_item = 57832
        if has_v and os.path.exists(os.path.join(dir_str, 'feat_v.pt')):
            v_feat = torch.load(dir_str+'/feat_v.pt').to(device)
        if has_a and os.path.exists(os.path.join(dir_str, 'feat_a.pt')):
            a_feat = torch.load(dir_str+'/feat_a.pt').to(device)
        if has_t and os.path.exists(os.path.join(dir_str, 'feat_t.pt')):
            t_feat = torch.load(dir_str+'/feat_t.pt').to(device)

    elif dataset == 'kwai':
        num_user = 7010
        num_item = 86483
        num_warm_item = 74470
        if has_v and os.path.exists(os.path.join(dir_str, 'feat_v.npy')):
            v_feat = torch.tensor(np.load(dir_str+'/feat_v.npy', allow_pickle=True), dtype=torch.float).to(device)

    return num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, dataset, train_data, num_neg):
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = set(np.load('./Data/'+dataset+'/cold_set.npy'))
        self.all_set = set(range(num_user, num_user+num_item))-self.cold_set

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]
        neg_item = random.sample(self.all_set-set(self.user_item_dict[user]), self.num_neg)
        user_tensor = torch.LongTensor([user]*(self.num_neg+1))
        item_tensor = torch.LongTensor([pos_item] + neg_item)
        return user_tensor, item_tensor

