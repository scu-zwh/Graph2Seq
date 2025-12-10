import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import torch.nn.functional as F
import numpy as np

from cadlib.macro import SOL_IDX, PAD_IDX, EOS_IDX
from params import *
from graph_encoders import GNN
from attention_decoder import Graph2SeqTransformer
from train import train
from utils import *
from data_proc import cad_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser("Interface for Graph2Seq")

    # General
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--root_path', type=str, default='./data/deepcad/cad_graph1')
    parser.add_argument('--train_list', type=str, default='/mnt/data/zhengwenhao/workspace/cad_graph/ProxBFN-main/data/deepcad/train_all_graphs.txt')
    parser.add_argument('--val_list', type=str, default='/mnt/data/zhengwenhao/workspace/cad_graph/ProxBFN-main/data/deepcad/val_all_graphs.txt')
    parser.add_argument('--test_list', type=str, default='/mnt/data/zhengwenhao/workspace/cad_graph/ProxBFN-main/data/deepcad/test_all_graphs.txt')  
    
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
       
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio.')

    # The Encoder
    parser.add_argument('--gnn', type=str, default='gin', choices=['gcn', 'gin', 'gat'], help='The GNN architecture.')
    parser.add_argument('--gnn_hidden_channels', type=int, default=128, help='Number of GNN hidden channels.')
    parser.add_argument('--gnn_num_layers', type=int, default=4, help='Number of hidden layers for the GNN.')
    # parser.add_argument('--use_gdc', action='store_true', help='Whether to use GDC.')

    # The Decoder
    parser.add_argument('--dec_hidden_state_size', type=int, default=256, help='Number of decoder hidden channels.')

    # General Training
    parser.add_argument('--vocab_size', type=int, default=263, help='Vocabulary size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    # parser.add_argument('--wandb', action='store_true', help='Track experiment')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

def main():
    """
    The main flow of Graph2Seq
    """
    args, sys_argv = get_args()
    
    # ====================================================================================
    # ===================================== Data Loading & Processing
    """
    Assumption on Data Processing:
        1. Data contains three splits (i.e., train, validation, & test) covering the whole datasets
        2. Each data split (i.e., train, validation, & test) consists of a list of pairs (graph, sentence).
           The length of the list specifies the number of instances in the split. 
           Each instance corresponds to one 'graph (of a SQL query)' that maps to one 'interpretation'.
        3. There should be a 'graph_lang' (~input_lang) that maps each node of the graph to its node id.
        4. There should be a 'output_lang' that maps each word in a sentence to its corresponding id.
        5. 'graph': it includes the input graph in the convention format accepted by PyTorch Geometric.
                    The node ids comes from 'graph_lang'.
        6. 'sentence': this is a English sentence. The word ids come from 'output_lang'.
    """

    datamodule = cad_dataset.CADGraphDataModule(args)
    
    data_train, data_val, data_test = datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()

    # ====================================================================================
    # ===================================== Model Definition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ====== wandb 初始化 ======
    wandb.init(
        project="Graph2Seq-CAD",     
        name="gnn_transformer_run4",   
        config={
            "lr": args.lr,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "d_model": args.dec_hidden_state_size,
            "gnn_hidden": args.gnn_hidden_channels,
            "batch_size": args.batch_size,
        }
    )

    save_dir = "./checkpoints/scheduler1_ls"
    os.makedirs(save_dir, exist_ok=True)

    # graph encoder
    graph_encoder = GNN(
        in_channels=num_node_init_feats,
        hidden_channels=args.gnn_hidden_channels,
        out_channels=args.dec_hidden_state_size,
        num_layers=args.gnn_num_layers,
        dropout=args.dropout,
        gnn_mode=args.gnn
    )

    model = Graph2SeqTransformer(
        gnn=graph_encoder,
        vocab_size=args.vocab_size,
        d_model=args.dec_hidden_state_size,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        max_len=MAX_LENGTH,        # 你的 CAD 序列上限
        dropout=args.dropout,
        pad_idx=PAD_IDX
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),   # Transformer 标配
        weight_decay=0.01
    )

    min_lr = 1e-5

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50,
        min_lr=min_lr,
        verbose=True
    )

    # criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.0)
    
    # ========= 打印参数量 =========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ====================================================================================
    # ===================================== Train & Validation   
    train(
        train_loader=data_train,    
        val_loader=data_val,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=args.epochs,
        device=device,
        pad_idx=262,
        max_length=MAX_LENGTH,
        save_dir=save_dir,
        save_every=50,                    # 每 50 epoch 存一次普通 checkpoint
    )
    
    wandb.finish()

if __name__ == '__main__':
    main()