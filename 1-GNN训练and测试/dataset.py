import torch
from torch.utils.data import DataLoader, Dataset
import os
import random
from transformers import AutoTokenizer
import pandas as pd
import pickle
import numpy as np

def load_data(args):
    
    # 图结构
    neighbors, node_graph_feat = pickle.load(open('graph_data.pkl','rb'))
    args.neighbors = [[],[]]
    for tail, heads in enumerate(neighbors):
        for head in heads:
            args.neighbors[0].append(head)
            args.neighbors[1].append(tail)
            
    sen_embs = pickle.load(open('sen_embs.pkl','rb'))
    # args.node_graph_feat = node_graph_feat
    args.node_graph_feat = sen_embs
    
    # 训练&测试集
    X_train, X_test, y_train, y_test = pickle.load(open('dataset.pkl','rb'))
    args.n_class = max(y_train)+1


    train_set = GraphDataSet(X_train, y_train)
    test_set = GraphDataSet(X_test, y_test)

    train_loader = DataLoader(train_set, args.batch_size,  num_workers=0,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size,  num_workers=0)
    return train_loader, test_loader

class GraphDataSet(Dataset):
    def __init__(self, node, label):
        super().__init__()
        self.node = node
        self.label = label
        
    def __len__(self):
        return len(self.node)
    
    def __getitem__(self, idx): 
        return self.node[idx], self.label[idx]

