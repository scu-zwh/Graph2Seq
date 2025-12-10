import os
import h5py
import torch
import pathlib
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

from data_proc.abstract_dataset import AbstractDataModule

def matrix_to_vector(matrix):
    flattened_array = matrix.flatten()
    result_vector = flattened_array[flattened_array != -1]
    return result_vector

class CADGraphDataset(InMemoryDataset):
    """
    本地 CAD Graph 数据集
    每个 .pt 文件都是一个 torch_geometric.data.Data 对象
    不需要下载
    """
    def __init__(self, split, root, file_list_path, transform=None, pre_transform=None, pre_filter=None):
        self.split = split  # train / val / test
        self.file_list_path = file_list_path  # 例如 connected_graphs.txt
        
        self.raw_data = '/mnt/data/zhengwenhao/workspace/cad_graph/ProxBFN-main/data/deepcad/cad_vec'
        self.seq_len = 64  # CAD 序列长度上限
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # 不用实际下载，只是占位
        return [f"{self.split}.txt"]

    @property
    def processed_file_names(self):
        # 每个 split 会保存为单独的 .pt
        return [f"{self.split}.pt"]

    def download(self):
        # 本地数据集，不需要下载
        pass

    def process(self):
        print(f"Processing local CADGraphDataset split = {self.split}")

        # 读取包含 .pt 文件路径列表
        with open(self.file_list_path, "r") as f:
            graph_rel_paths = [line.strip() for line in f if line.strip()]

        data_list = []
        
        for i, rel_path in enumerate(tqdm(graph_rel_paths, desc=f"Loading {self.split} graphs")):
            full_path = os.path.join(self.root, f"{rel_path}.pt")
            
            # 找到CAD序列
            h5_path = os.path.join(self.raw_data, rel_path + ".h5")
            with h5py.File(h5_path, "r") as fp:
                cad_mat = fp["vec"][:] # (len, 1 + N_ARGS)
            cad_mat[:, 0] += 256
            cad_vec = matrix_to_vector(cad_mat)
                
            # self.all_data_lengths.append(len(cad_vec))
                
            cad_vec = np.pad(cad_vec, (0, self.seq_len-len(cad_vec)), mode='constant',
                            constant_values=262)
                      
            cad_vec = torch.tensor(cad_vec, dtype=torch.int32)            
                   
            if not os.path.exists(full_path):
                print(f"⚠️ 文件不存在: {full_path}")
                continue
            try:
                data = torch.load(full_path, weights_only=False)
                
                cad_geom = data.x[:, 1:].clone()  # 提取 CAD 几何信息
                edge_type_id = data.x[:, 0].long()
                data.x = F.one_hot(edge_type_id, num_classes=3).float()
                
                data.x = torch.cat([data.x, cad_geom], dim=-1)  # 拼接几何信息
                
                # 由于在创建 Graph 时缺少 edge_attr，自动补一个 dummy 特征
                if getattr(data, "edge_attr", None) is None:
                    num_edges = data.edge_index.shape[1]
                    data.edge_attr = torch.zeros((num_edges, 2), dtype=torch.float)
                    data.edge_attr[:, 1] = 1

                # ✅ 若缺少 y，则补一个空张量，shape = (1, 0)
                if getattr(data, "y", None) is None:
                    data.y = cad_vec

                # 常规预处理                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data.idx = i  # 添加索引属性
                    
                data_list.append(data)
            except Exception as e:
                print(f"⚠️ 加载失败 {full_path}: {e}")

        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"✅ 已处理 {len(data_list)} 个图, 保存至 {self.processed_paths[0]}")


class CADGraphDataModule(AbstractDataModule):
    """
    CAD Graph 数据模块
    封装 train/val/test 三个 Dataset
    """
    def __init__(self, cfg):
        self.cfg = cfg
        root_path = cfg.root_path  # 根路径，如 data/deepcad/cad_graph/
        
        # 三个 split 列表路径
        train_list = cfg.train_list
        val_list = cfg.val_list
        test_list = cfg.test_list

        datasets = {
            'train': CADGraphDataset(split='train', root=root_path, file_list_path=train_list),
            'val': CADGraphDataset(split='val', root=root_path, file_list_path=val_list),
            'test': CADGraphDataset(split='test', root=root_path, file_list_path=test_list)
        }
        
        # temp = datasets['train']
        # temp_data = temp[0]
        
        # print(temp_data)
        # for key in temp_data.keys:
        #     print(f"{key}: {temp_data[key]}")
        # exit(0)

        super().__init__(cfg, datasets)
        self.inner = datasets['train']
        self.datasets = datasets

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

    def __getitem__(self, item):
        return self.inner[item]
    
    def valency_count(self, max_n_nodes):
        """
        Compute degree (valency) distribution for CAD graphs.
        Return: tensor of length (3 * max_n_nodes - 2)
                following DiGress convention.
        """
        all_deg = []

        for data in tqdm(self.datasets['train'] + self.datasets['val'] + self.datasets['test'],
                        desc="Computing CAD valency distribution"):
            # PyG Graph data: edge_index shape (2, E)
            edge_index = data.edge_index
            deg = torch.bincount(edge_index[0], minlength=data.num_nodes)
            all_deg.append(deg)

        all_deg = torch.cat(all_deg, dim=0).float()

        # max allowed by DiGress (ensures shape compatibility)
        max_val = 3 * max_n_nodes - 2
        hist = torch.zeros(max_val)

        max_deg = int(all_deg.max().item())
        max_deg = min(max_deg, max_val - 1)

        for d in all_deg:
            idx = min(int(d.item()), max_val - 1)
            hist[idx] += 1

        hist = hist / hist.sum()  # normalize to probability

        return hist

