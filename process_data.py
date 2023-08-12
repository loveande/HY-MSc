import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
# 图特征提取
import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
import pickle

def cosine_similarity(vector, matrix):
    # 计算向量的范数
    vector_norm = np.linalg.norm(vector)
    # 计算矩阵每个样本的范数
    matrix_norm = np.linalg.norm(matrix, axis=1)
    # 计算向量和矩阵每个样本的内积
    dot_product = np.dot(matrix, vector)
    # 计算余弦相似度
    similarity = dot_product / (matrix_norm * vector_norm)
    return similarity


if __name__=="__main__":
    # 新的数据集
    # https://www.kaggle.com/datasets/honyuu/chnsenticorp-htl-all?resource=download
    df = pd.read_csv('dataset.csv')
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')
    device = 'cuda'
    model = model.to(device)
        
        
        
    sen_embs = []
    labels = []

    for sen, lab in tqdm(df.to_numpy(),ncols=80):
        # 文本预处理
        if isinstance(lab,int) and isinstance(sen, str):
            tokens, mask, t_type = tokenizer(sen, 
                    padding="max_length", 
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                    ).values()
            # 抽取文本语义特征
            sen_emb = model(
                            input_ids=tokens.to(device),
                            token_type_ids=mask.to(device),
                            attention_mask=t_type.to(device)
                            )['pooler_output']
            sen_embs.append(sen_emb.detach().cpu().numpy())
            labels.append(lab)

    sen_embs = np.concatenate(sen_embs, 0) 
        
    # 根据语义相似度构建图
    k = 3 # 每个节点的邻居数量
    # 为每个节点找到最相似的k个节点作为图中的邻居节点
    neighbors = [] 
        
    for sen in tqdm(sen_embs, ncols=80):
        # 计算语义相似度
        similarity = cosine_similarity(sen, sen_embs) 
        # 找到节点的邻居
        neis = similarity.argsort()[-k-1:]
        # 保存邻居信息
        neighbors.append(neis)
        
        
        
    # 创建图结构
    G = nx.Graph()
    for tail, neis in enumerate(neighbors):
        for head in neis:
            G.add_edge(head, tail)
    # 使用 Node2Vec 训练节点嵌入向量
    node2vec = Node2Vec(G, dimensions=32, walk_length=30, num_walks=200, workers=4)
    # 拟合模型（生成随机游走序列）
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # 获取节点的嵌入向量
    node_embeddings = model.wv


    # 将图特征和图结构保存
    node_graph_feat = []
    for i in range(len(sen_embs)):
        node_graph_feat.append(node_embeddings[i])
    node_graph_feat = np.stack(node_graph_feat, 0)

    nodes = np.arange(len(node_graph_feat)).reshape((-1,1))

        
        
    # 将label数字化
    label2int = dict((l,idx) for idx, l in enumerate(np.unique((labels))))
    label_int = [label2int[l] for l in labels]    
        
    # 使用train_test_split函数划分数据集
    X_train, X_test, y_train, y_test = train_test_split(nodes, label_int, test_size=0.2, stratify=label_int, random_state=42)
    X_train = X_train.reshape(-1)
    X_test = X_test.reshape(-1)

    pickle.dump((neighbors,node_graph_feat), open('graph_data.pkl','wb'))
    pickle.dump((X_train, X_test, y_train, y_test), open('dataset.pkl','wb'))
    pickle.dump(sen_embs, open('sen_embs.pkl','wb'))
    pickle.dump((label2int,labels), open('label2int.pkl','wb'))