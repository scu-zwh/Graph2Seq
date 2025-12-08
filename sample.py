import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F

from tqdm import tqdm
from cadlib.cad_transfer import get_output_metrics
from params import *
from graph_encoders import GNN
from attention_decoder import Graph2SeqTransformer
from train import train
from utils import *
from data_proc import cad_dataset

from main import get_args
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

log_file = open("out.txt", "a", encoding="utf-8") 

def log(*args, **kwargs):
    print(*args, **kwargs)                    
    print(*args, **kwargs, file=log_file)     

def sample():
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
    )
    
    # ============ load checkpoint ======================================================
    checkpoint_path = "./checkpoints/scheduler/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # ========= 打印参数量 =========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ====================================================================================
    count = 0
    
    epoch_mse_sum    = 0.0   
    epoch_mse_count  = 0    

    epoch_valid_ok   = 0    
    epoch_valid_all  = 0       
    
    all_gt = []
    all_pred = []

    for idx, batch in tqdm(enumerate(data_test)):
        batch = batch.to(device)

        B = batch.idx.size(0)
        seq_len = batch.y.size(0) // B
        y = batch.y.view(B, seq_len).long()

        y_in  = y[:, :-1]
        y_out = y[:, 1:]

        logits = model(batch, y_in)
        Tm1, B_, V = logits.shape
        logits = logits.view(-1, V)
        
        y_pred = torch.argmax(logits, dim=-1)
        y_out  = y_out.transpose(0, 1).contiguous().view(-1)

        # 将SOL拼接回来
        SOL_IDX = 260
        y_pred = torch.cat(
            [torch.full((B_,), SOL_IDX, dtype=torch.long, device=device), y_pred],
            dim=0
        )
        y_out = torch.cat( 
            [torch.full((B_,), SOL_IDX, dtype=torch.long, device=device), y_out],
            dim=0
        )
        
        
        metrics, valid_count = get_output_metrics(y_pred)
        r_metrics, r_valid_count = get_output_metrics(y_out)

        epoch_valid_ok   += valid_count
        epoch_valid_all  += r_valid_count
        
        if valid_count:       
            gt = torch.as_tensor(r_metrics, device=device)
            pred = torch.as_tensor(metrics, device=device)  # (B, 6)
            
            all_gt.append(gt)          # (B, 6)
            all_pred.append(pred)

            mask = pred.ne(-1) & gt.ne(-1)
            diff = (pred - gt).masked_select(mask)
            epoch_mse_sum   += (diff ** 2).sum().item()
            epoch_mse_count += mask.sum().item()   

    all_gt = torch.cat(all_gt, dim=0).cpu().numpy()
    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()       

    if epoch_mse_count:
        avg_mse  = epoch_mse_sum / epoch_mse_count
    else:
        avg_mse  = float("nan")
    valid_ratio = epoch_valid_ok / max(epoch_valid_all, 1)
    
    log("BFN Sample Results:")
    log("-" * 30)
    log(f"epoch_valid_ok = {epoch_valid_ok}, all_count = {epoch_valid_all}")
    log(f"Valid p: {valid_ratio:.4f}")
    log(f"MSE: {avg_mse:.4f}")

    for i in range(6):
        mask = (all_pred[:, i] != -1) & (all_gt[:, i] != -1)
        if i < 2:
            if mask.sum() > 1:
                y_pred = all_pred[mask, i]
                y_true = all_gt[mask, i]

                r, _ = pearsonr(y_pred, y_true)

                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)

                log(f"Column {i} → Pearson r = {r:.4f} | MSE = {mse:.4f} | MAE = {mae:.4f}")
            else:
                log(f"Column {i} → Pearson r / MSE / MAE = NaN (insufficient valid samples)")
        else:
            if mask.sum() > 0:
                acc = (all_pred[mask, i] == all_gt[mask, i]).sum() / mask.sum()
                log(f"Column {i} Accuracy = {acc:.4f}")
            else:
                log(f"Column {i} Accuracy = NaN (insufficient valid samples)")

    log("-" * 30)

    log_file.close()

if __name__ == '__main__':
    sample()