import argparse
import torch

def get_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_layers', type=int, default=2)   

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='learning rate') 
    
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
    parser.add_argument('--device', default='cuda', type=str,help='cuda or cpu')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate') 
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty') 
    
    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: args.device = torch.device('cpu')
    return args
