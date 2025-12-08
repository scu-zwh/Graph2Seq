
"""
Attention-based Decoder

Date:
    - Jan. 28, 2023
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils import *
from params import MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Graph2SeqTransformer(torch.nn.Module):
    def __init__(self, gnn, vocab_size, d_model, nhead=8, num_layers=4,
                 dim_feedforward=1024, max_len=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.gnn = gnn
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = torch.nn.Embedding(max_len, d_model)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # 我下面用 [T, B, C] 格式
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, batch, y_in):
        # 1. Graph encoder：得到每个图的 graph embedding
        node_encs, graph_emb = self.gnn(batch.x, batch.edge_index, batch.batch, None)
        # graph_emb: [B, d_model]
        memory = graph_emb.unsqueeze(0)  # [1, B, d_model]

        # 2. token + position embedding
        B, T = y_in.shape
        pos = torch.arange(T, device=y_in.device).unsqueeze(0).expand(B, T)
        tgt = self.tok_emb(y_in) + self.pos_emb(pos)  # [B, T, d_model]
        tgt = tgt.transpose(0, 1)  # [T, B, d_model]

        # 3. causal mask（防止看见未来）
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(y_in.device)

        # 4. padding mask（可选，如果有 pad_idx）
        tgt_key_padding_mask = (y_in == self.pad_idx)  # [B, T]

        decoded = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [T, B, d_model]

        logits = self.out_proj(decoded)  # [T, B, vocab_size]
        return logits