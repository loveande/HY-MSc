import torch
from torch import nn
import os
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # gnn
        self.dim = args.node_graph_feat.shape[-1]
        self.gnn = nn.ModuleList(
            GATConv(self.dim, self.dim,dropout=args.dropout) # GAT
            for _ in range(args.gnn_layers)
        )
        # 分类层
        self.cls = nn.Linear(self.dim, args.n_class)
        # 参数初始化
        for p in self.parameters():
            nn.init.normal_(p.data, 0, 0.1)
        
        # 图结构和节点特征矩阵        
        self.neighbors = nn.Parameter(torch.LongTensor(args.neighbors), requires_grad=False)
        self.node_graph_feat = nn.Parameter(torch.FloatTensor(args.node_graph_feat), requires_grad=False)
        # 损失函数
        self.loss_func = nn.CrossEntropyLoss()
            
    def forward(self, node):
        node_emb = self.node_graph_feat
        for gnn in self.gnn:
            node_emb = gnn(node_emb, self.neighbors)
        
        cls_feat = node_emb[node.long()]
        pred = self.cls(cls_feat)
        
        return pred
