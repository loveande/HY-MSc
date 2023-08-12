from tqdm import tqdm
from dataset import load_data
import torch
from parse import get_parse
from utils import fix_seed
from model import Model
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc

SHOW = True

def main(args):
    fix_seed()

    train_loader, test_loader = load_data(args)

    model = Model(args)
    model.to(args.device)

    # 记录各种最好的指标
    best_acc = 0
    best_recall = 0
    best_precision = 0
    best_f1 = 0
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
    for e in range(args.epoch):
        model.train()
        all_loss = 0.0
        all_num = 0
        
        if SHOW:
            bar = tqdm(train_loader, total=len(train_loader),ncols=100)
        else:
            bar = train_loader
        for node, label in bar:
            pred = model(node.to(args.device))
            optimizer.zero_grad() 
            loss = model.loss_func(pred, label.to(args.device))
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            all_num += len(node)
            if SHOW:
                bar.set_postfix(Epoch=e, LR=optimizer.param_groups[0]['lr'], Train_Loss=all_loss/all_num)
        if SHOW:
            print('epoch%d - loss%f'%(e,all_loss/len(train_loader)))  
              
        # 验证模型
        with torch.no_grad():
            model.eval()
            all_pred = []
            all_true = []
            
            for node, label in test_loader:
                pred= model(node.to(args.device))
                all_pred.append(torch.softmax(pred,-1).detach().cpu().numpy())
                all_true.append(label.numpy())
                
            all_pred = np.concatenate(all_pred, 0)
            all_true = np.concatenate(all_true, 0)
            all_pred_labels = np.argmax(all_pred, -1)

            acc = accuracy_score(all_true, all_pred_labels)
            recall = recall_score(all_true, all_pred_labels,average='macro')
            precision = precision_score(all_true, all_pred_labels,average='macro')
            f1 = f1_score(all_true, all_pred_labels,average='macro')
            
            print("ACC-%.2f%%"%(acc*100))
            print("RECALL-%.2f%%"%(recall*100))
            print("PRECISION-%.2f%%"%(precision*100))
            print("F1-%.2f%%"%(f1*100))

            if best_acc < acc:
                best_acc = acc
            if best_recall < recall:
                best_recall = recall
            if best_precision < precision:
                best_precision = precision
            if best_f1 < f1:
                best_f1 = f1   
                torch.save(model, 'model.pt')
                
    return best_acc, best_recall, best_precision, best_f1      

if __name__ == '__main__':
    args = get_parse()
    best_acc, best_recall, best_precision, best_f1 = main(args)
    print("============")
    print("BEST-ACC = %.2f%%"%(best_acc*100))
    print("BEST-RECALL = %.2f%%"%(best_recall*100))
    print("BEST-PRECISION = %.2f%%"%(best_precision*100))
    print("BEST-F1 = %.2f%%"%(best_f1*100))