import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)) 

from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
from cadlib.macro import *

def matrix_to_vector(matrix):
    flattened_array = matrix.flatten()
    result_vector = flattened_array[flattened_array != -1]
    return result_vector

class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.path = os.path.join(config.data_root, config.dataid_path)
        self.metrics_path = os.path.join(config.data_root, config.metrics_path)
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]
        with open(self.metrics_path, "r") as fp:
            self.all_metrics_data = json.load(fp)[phase]

        self.seq_len = config.seq_len
        self.all_data_lengths = []
        
        self.normalization_param = {'area': [2.079852546659958, 1.813212545495623],
                                    'vol': [0.14796799684439665, 0.2250738868118991]}  # normalization parameters for area and volume, DeepCAD

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)
    
    def normalize_data(self, sv_data):
        return {
            "area": (sv_data["area"] - self.normalization_param["area"][0]) / self.normalization_param["area"][1],
            "vol": (sv_data["vol"] - self.normalization_param["vol"][0]) / self.normalization_param["vol"][1]
        }     


    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        metrics_data = self.all_metrics_data[data_id]
        sv_data = {'area': metrics_data["area"],
                   'vol': metrics_data["vol"]}
        
        init_conditions = np.array([metrics_data["area"], metrics_data["vol"], metrics_data["face"], metrics_data["edge"],
                                    metrics_data["planar_faces"], metrics_data["curved_faces"],
                                    metrics_data["linear_edges"], metrics_data["curved_edges"],], dtype=float)
        
        sv_data = self.normalize_data(sv_data)
        with h5py.File(h5_path, "r") as fp:
            cad_mat = fp["vec"][:] # (len, 1 + N_ARGS)
        cad_mat[:, 0] += 256
        cad_vec = matrix_to_vector(cad_mat)
            
        # self.all_data_lengths.append(len(cad_vec))
            
        cad_vec = np.pad(cad_vec, (0, self.seq_len-len(cad_vec)), mode='constant',
                         constant_values=PAD_IDX)
        
        conditions = np.array([sv_data["area"], sv_data["vol"], metrics_data["face"], metrics_data["edge"],
                                metrics_data["planar_faces"], metrics_data["curved_faces"],
                                metrics_data["linear_edges"], metrics_data["curved_edges"],], dtype=float)
        
        cad_vec = torch.tensor(cad_vec, dtype=torch.int32) 
        conditions = torch.tensor(conditions, dtype=torch.float32)
        init_conditions = torch.tensor(init_conditions, dtype=torch.float32)

        return {"cad_vec": cad_vec, "conditions": conditions,
                "init_conditions": init_conditions, "id": data_id}        

    def __len__(self):
        return len(self.all_data)