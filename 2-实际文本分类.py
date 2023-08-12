import torch
import os
import pickle
from transformers import AutoTokenizer, AutoModel
import numpy as np

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



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')
    device = 'cpu'
    model = model.to(device)
    
    sen_embs = pickle.load(open('sen_embs.pkl','rb'))
    label2int,labels = pickle.load(open('label2int.pkl','rb'))
    
    word = input("请输入一句话(输入0结束):")
    
    while word != '0':
        
        # 文本预处理
        tokens, mask, t_type = tokenizer(word, 
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
                        )['pooler_output'].detach().cpu().numpy().reshape(-1)
        # 计算语义相似度
        similarity = cosine_similarity(sen_emb, sen_embs) 
        # 找到最相似的节点
        sim_node = similarity.argsort()[-1]
        
        print('识别的情绪为：'+labels[sim_node])
        
        word = input("请输入一句话(输入0结束):")
    print('=======System Out========')
        