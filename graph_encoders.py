"""
Graph Encoder of Graph2Seq Architecture

Date:
    - Jan. 28, 2023
"""
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.logging import log

from torch_geometric.datasets import Planetoid

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATv2Conv,
    global_mean_pool,
    BatchNorm,
    JumpingKnowledge,
)

class GNN(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=4,
                 dropout=0.1,
                 gnn_mode='gin'):  # 'gcn' / 'gin' / 'gat'
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_mode = gnn_mode

        convs = []
        norms = []

        in_ch = in_channels
        for layer in range(num_layers):
            # 这里你可以继续让最后一层 out_ch = out_channels，其它层 = hidden_channels
            if layer == num_layers - 1:
                out_ch = out_channels
            else:
                out_ch = hidden_channels

            if gnn_mode == 'gcn':
                conv = GCNConv(in_ch, out_ch, cached=False, normalize=True)
            elif gnn_mode == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_ch, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_ch),
                )
                conv = GINConv(mlp)
            elif gnn_mode == 'gat':
                heads = 4
                conv = GATv2Conv(in_ch, out_ch // heads, heads=heads, dropout=dropout)
            else:
                raise ValueError(f"Unknown gnn_mode: {gnn_mode}")

            convs.append(conv)
            norms.append(BatchNorm(out_ch))
            in_ch = out_ch

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x, edge_index, batch_vec, edge_weight=None):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_in = h

            if self.gnn_mode == 'gcn':
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)

            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # 维度一致时做残差
            if h_in.shape == h.shape:
                h = h + h_in

        node_encs = h                               # [N_total, out_channels]
        graph_emb = global_mean_pool(node_encs, batch_vec)  # [B, out_channels]

        return node_encs, graph_emb

def train_gnn(model, optimizer, data, epochs):
    """
    train a GNN object
    NOTE: this is for local testing
    """
    best_val_acc = 0
    for epoch in range(epochs):
        # train
        model.train()
        optimizer.zero_grad()
        out, pooled_ge = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # evaluate
        train_acc, val_acc = eval_gnn(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc)



def eval_gnn(model, data):
    with torch.no_grad():
        model.eval()
        pred, pooled_ge = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def test_gnn(model, data):
    # final test
    with torch.no_grad():
        model.eval()
        pred, pooled_ge = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.argmax(dim=-1)
        test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

    # print("DEBUG: Test: pooled_ge:", pooled_ge)

    return test_acc



def main():
    """
    To test the functionality of graph encoder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--gnn', type=str, default='gcn', choices=['gcn', 'bi_gcn'], help='The GNN architecture.')
    parser.add_argument('--gnn_hidden_channels', type=int, default=80, help='Number of GNN hidden channels.')
    parser.add_argument('--gnn_num_layers', type=int, default=7, help='Number of hidden layers for the GNN.')
    args = parser.parse_args()
    print("DEBUG: args:\n", args)

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    print(data)

    # if args.use_gdc:
    #     transform = T.GDC(
    #         self_loop_weight=1,
    #         normalization_in='sym',
    #         normalization_out='col',
    #         diffusion_kwargs=dict(method='ppr', alpha=0.05),
    #         sparsification_kwargs=dict(method='topk', k=128, dim=0),
    #         exact=True,
    #     )
    #     data = transform(data)

    # define model
    model = GNN(dataset.num_features, args.gnn_hidden_channels, dataset.num_classes, num_layers=args.gnn_num_layers,
                dropout=args.dropout, use_gdc=args.use_gdc, gnn_mode=args.gnn)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train & validation
    train_gnn(model, optimizer, data, args.epochs)

    # test
    test_acc = test_gnn(model, data)
    log(Final_Test_ACC=test_acc)


if __name__ == '__main__':
    main()