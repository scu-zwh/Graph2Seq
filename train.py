"""
Graph2Seq: Training procedure

Date:
    - Jan. 28, 2023
"""
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import math
import time
import random
import numpy as np

from utils import *
from params import *
import wandb

# set random seed
random.seed(RAND_SEED)

def train_epoch(train_loader, model, optimizer, criterion, device, pad_idx, max_len):
    model.train()
    all_losses = []

    for batch in train_loader:
        batch = batch.to(device)

        # 恢复 [B, T]
        B = batch.idx.size(0)
        seq_len = batch.y.size(0) // B
        y = batch.y.view(B, seq_len).long()  # [B, T]

        # 这里假设 y 已经是 [SOS, ..., EOS, PAD...]
        y_in  = y[:, :-1]  # 输入给 decoder
        y_out = y[:, 1:]   # 用来算 loss

        optimizer.zero_grad()
        logits = model(batch, y_in)  # [T-1, B, vocab]

        # 调整形状算 CrossEntropy
        Tm1, B, V = logits.shape
        logits = logits.view(-1, V)           # [(T-1)*B, V]
        y_out  = y_out.transpose(0, 1).reshape(-1)  # [(T-1)*B]

        loss = criterion(logits, y_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        wandb.log({"train/step_loss": loss.item()})
        all_losses.append(loss.item())

    return np.mean(all_losses)


def val_epoch(val_loader, model, criterion, device, pad_idx, max_len):
    model.eval()
    all_losses = []

    with torch.no_grad():   # ✅ 禁用梯度
        for batch in val_loader:
            batch = batch.to(device)

            B = batch.idx.size(0)
            seq_len = batch.y.size(0) // B
            y = batch.y.view(B, seq_len).long()

            y_in  = y[:, :-1]
            y_out = y[:, 1:]

            logits = model(batch, y_in)
            Tm1, B_, V = logits.shape
            logits = logits.view(-1, V)
            y_out  = y_out.transpose(0, 1).contiguous().view(-1)

            loss = criterion(logits, y_out)

            wandb.log({"val/step_loss": loss.item()})
            all_losses.append(loss.item())

    return float(np.mean(all_losses))


def train(train_loader, val_loader, model, optimizer, scheduler, criterion,
          epochs, device, pad_idx, max_length, save_dir, save_every=5):

    print("INFO: Number of batches per epoch: {}".format(len(train_loader)))

    best_val_loss = float("inf")
    best_ckpt_path = None

    for epoch_idx in range(epochs):
        start_time_epoch = time.time()

        # 1. 训练一个 epoch
        train_loss = train_epoch(
            train_loader, model, optimizer,
            criterion, device, pad_idx=pad_idx, max_len=max_length
        )
        
        val_loss = val_epoch(
            val_loader, model, criterion,
            device, pad_idx=pad_idx, max_len=max_length
        )
        
        scheduler.step(val_loss)

        elapsed = time.time() - start_time_epoch
        
        # 当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 2. 打印日志
        print("INFO: Epoch: {}, Elapsed time: {:.2f}s.".format(epoch_idx, elapsed))
        print("INFO: \tTrain loss: {:.4f}".format(train_loss))
        print("INFO: \tVal   loss: {:.4f}".format(val_loss))
        print("INFO: \tLR        : {:.6f}".format(current_lr))

        # 3. wandb 记录（epoch 级）
        wandb.log({
            "epoch": epoch_idx,
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "lr": current_lr,
            "time/epoch": elapsed,
        })

        # 4. 无论 save_every 与否，都检查是否是最好模型 ✅
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_ckpt_path)
            print(f"INFO: New best model saved to {best_ckpt_path} (val_loss={val_loss:.4f})")

        # 5. 间隔保存普通 checkpoint
        if (epoch_idx + 1) % save_every == 0:           
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch_idx+1}.pt")
            torch.save({
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"INFO: Saved checkpoint to {ckpt_path}")

    print(f"INFO: Training done. Best val_loss={best_val_loss:.4f}, best_model={best_ckpt_path}")