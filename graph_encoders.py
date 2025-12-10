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

def node_render(X):
    """
    X: (n, d)
    first 3 dims = one-hot node type
    P0 = X[..., 3:6]
    P1 = X[..., 6:9]
    C  = X[..., 9:12]
    output: (n, 100, 3)
    """
    device = X.device
    n, _ = X.shape
    num_points = 100

    # node type = [LINE, ARC, CIRCLE]
    node_types = X[..., 0]     # (n)

    # geometry
    P0 = X[..., 3:6]          # (n, 3)
    P1 = X[..., 6:9]          # (n, 3)
    C  = X[..., 9:12]         # (n, 3)

    # output
    X_render = torch.zeros((n, num_points, 3), device=device)

    # t for line interpolation
    t = torch.linspace(0, 1, num_points, device=device)           # (100,)
    t = t.view(1, num_points, 1)                               # (1,100,1)

    # =========================
    # LINE
    # =========================
    mask_line = (node_types == 0).view(n, 1, 1)               # (n,1,1)
    if mask_line.any():
        P0e = P0.unsqueeze(1)                                     # (n,1,3)
        P1e = P1.unsqueeze(1)                                     # (n,1,3)
        line_points = P0e + t * (P1e - P0e)                       # (n,100,3)
        X_render = X_render.where(~mask_line, line_points)

    # =========================
    # ARC
    # =========================
    mask_arc = (node_types == 1).view(n, 1, 1)
    if mask_arc.any():
        vec0 = P0 - C                                             # (n,3)
        vec1 = P1 - C
        angle0 = torch.atan2(vec0[..., 1], vec0[..., 0])            # (n)
        angle1 = torch.atan2(vec1[..., 1], vec1[..., 0])
        angle1 = angle1 + (angle1 < angle0) * (2 * torch.pi)

        # expand to (n, 100)
        angles = angle0.unsqueeze(-1) + t.squeeze(-1) * (angle1 - angle0).unsqueeze(-1)

        r = torch.norm(vec0, dim=-1).unsqueeze(-1).unsqueeze(-1)  # (n,1,1)
        Ce = C.unsqueeze(1)  # (n,1,3)

        arc_points = torch.cat([
            (Ce[..., 0:1] + torch.cos(angles).unsqueeze(-1) * r),
            (Ce[..., 1:2] + torch.sin(angles).unsqueeze(-1) * r),
            Ce[..., 2:3].expand(n, num_points, 1)
        ], dim=-1)

        X_render = X_render.where(~mask_arc, arc_points)

    # =========================
    # CIRCLE
    # =========================
    mask_circle = (node_types == 2).view(n, 1, 1)
    if mask_circle.any():
        # angles must be (n,100)
        base_angles = torch.linspace(0, 2 * torch.pi, num_points, device=device)   # (100,)
        base_angles = base_angles.view(1, num_points).expand(n, num_points)

        r = torch.norm(P0 - C, dim=-1).unsqueeze(-1)          # (n,1)
        Ce = C.unsqueeze(1)                                    # (n,1,3)

        circle_points = torch.cat([
            Ce[..., 0:1] + torch.cos(base_angles).unsqueeze(-1) * r.unsqueeze(-1),
            Ce[..., 1:2] + torch.sin(base_angles).unsqueeze(-1) * r.unsqueeze(-1),
            Ce[..., 2:3].expand(n, num_points, 1)
        ], dim=-1)

        X_render = X_render.where(~mask_circle, circle_points)

    return X_render


class CNNEncoder(nn.Module):
    def __init__(self, output_dim):
        super(CNNEncoder, self).__init__()

        # 输入: (bs*n, 100, 3)  → reshape → (bs*n, 3, 100)
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        x: [n, 100, 3]
        return: [n, output_dim]
        """
        n, point_num, _ = x.shape

        # Reshape for CNN
        x = x.permute(0, 2, 1)                # (bs*n, 3, 100)

        # Conv1D blocks
        x = F.relu(self.conv1(x))             # (bs*n, 32, 100)
        x = F.relu(self.conv2(x))             # (bs*n, 64, 100)
        x = F.relu(self.conv3(x))             # (bs*n, 128, 100)

        # Global max pooling over points
        x = torch.max(x, dim=-1)[0]           # (bs*n, 128)

        # FC to output_dim
        x = self.fc(x)                        # (bs*n, output_dim)
        
        return x


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
        
        self.point_encoder = CNNEncoder(output_dim=64)

    def forward(self, x, edge_index, batch_vec, edge_weight=None):
        x_render = node_render(x)
        x_encoding = self.point_encoder(x_render)
        x = torch.cat([x, x_encoding], dim=-1).float()
        
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