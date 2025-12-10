import numpy as np 
import torch
import os  
    
from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD
from cadlib.macro import *
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from occwl.solid import Solid
from typing import Dict

from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Line
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from occwl.graph import face_adjacency

def save_solid_as_step(solid, filename):
    if solid.IsNull():
        print("solid 对象无效！")
        return
    
    step_writer = STEPControl_Writer()
    step_writer.Transfer(solid, STEPControl_AsIs)
    status = step_writer.Write(filename)
    
    if status == 1:  # 1 表示成功
        print("STEP 文件保存成功：", filename)
    else:
        print("STEP 文件保存失败！")

def save_steps(vec, path):
    os.makedirs(path, exist_ok=True)
    
    bs, length = vec.shape
    for i in range(bs):
        item = vec[i]
        
        item = cut_at_eos(item, EOS_IDX, PAD_IDX)
        
        mat = vector_to_matrix(item)
        if mat is None:
            continue
        try:
            cad = CADSequence.from_vector(mat, is_numerical=True, n=256)
            cad3d = create_CAD(cad)
        except:
            continue
        
        save_solid_as_step(cad3d, path + f'/{i}.step')

def count_faces_edges(solid: Solid):
    graph = face_adjacency(solid)
    num_faces = len(graph.nodes)
    num_edges = len(graph.edges) // 2
    return num_faces, num_edges

def count_geom_types(shape):
    planar_faces  = 0
    curved_faces  = 0
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face = face_exp.Current()
        surf_adp = BRepAdaptor_Surface(face, True)
        if surf_adp.GetType() == GeomAbs_Plane:
            planar_faces += 1
        else:
            curved_faces += 1
        face_exp.Next()

    linear_edges = 0
    curved_edges = 0
    seen_edges   = set()

    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_exp.More():
        edge = edge_exp.Current()              
        h = edge.HashCode(1000000)        
        if h in seen_edges:
            edge_exp.Next()
            continue
        seen_edges.add(h)

        curv_adp = BRepAdaptor_Curve(edge)
        if curv_adp.GetType() == GeomAbs_Line:
            linear_edges += 1
        else:
            curved_edges += 1
        edge_exp.Next()

    return {
        "planar_faces": planar_faces,
        "curved_faces": curved_faces,
        "linear_edges": linear_edges,
        "curved_edges": curved_edges,
    }


def compute_shape_metrics(shape, solid) -> Dict[str, float]:
    try:
        face_cnt, edge_cnt = count_faces_edges(solid)
    except Exception:
        face_cnt, edge_cnt = -1, -1

    try:
        area, vol = compute_surface_area_and_volume(shape)
    except Exception:
        area, vol = -1, -1

    try:
        geom_stats = count_geom_types(shape)      

    except Exception:
        geom_stats = {
            "planar_faces": -1,
            "curved_faces": -1,
            "linear_edges": -1,
            "curved_edges": -1,
        }

    return {
        "face": face_cnt,
        "edge": edge_cnt,
        "area": area,
        "vol": vol,
        **geom_stats,    
    }

def compute_surface_area_and_volume(shape):
    surface_props = GProp_GProps()
    volume_props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, surface_props)
    surface_area = surface_props.Mass()  
    brepgprop_VolumeProperties(shape, volume_props, True, False)
    volume = volume_props.Mass()  
    return surface_area, volume


def get_row(vector, i, row_length):
    if vector[i] == SOL_IDX:
        current_row = [vector[i]]
        current_row += [-1] * (row_length - len(current_row))
        i += 1
        return i, current_row
    
    elif vector[i] == LINE_IDX:
        if i + 2 >= len(vector): 
            return None, None
        current_row = [vector[i], vector[i+1], vector[i+2]]
        i += 3
        current_row += [-1] * (row_length - len(current_row))
        return i, current_row
    
    elif vector[i] == ARC_IDX:
        if i + 4 >= len(vector): 
            return None, None
        current_row = [vector[i], vector[i+1], vector[i+2], vector[i+3], vector[i+4]]
        i += 5
        current_row += [-1] * (row_length - len(current_row))
        return i, current_row
    
    elif vector[i] == CIRCLE_IDX:
        if i + 3 >= len(vector):
            return None, None
        current_row = [vector[i], vector[i+1], vector[i+2], -1, -1, vector[i+3]]
        i += 4
        current_row += [-1] * (row_length - len(current_row))
        return i, current_row   
             
    elif vector[i] == EXT_IDX:
        if i + 11 >= len(vector):  
            return None, None
        current_row = [vector[i], -1, -1, -1, -1, -1, vector[i+1], vector[i+2], vector[i+3], vector[i+4], vector[i+5], vector[i+6],
                        vector[i+7], vector[i+8], vector[i+9], vector[i+10], vector[i+11]]
        i += 12
        return i, current_row   
    
    elif vector[i] == EOS_IDX:
        current_row = [vector[i]]
        current_row += [-1] * (row_length - len(current_row))
        i += 1
        return i, current_row           

def vector_to_matrix(vector, row_length=17):
    vector = vector[vector != PAD_IDX]
    if vector[0] != SOL_IDX or vector[-1] != EOS_IDX:
        return None
        
    rows = []
    
    i = 0
    while i < len(vector):
        if vector[i] not in [LINE_IDX, ARC_IDX, CIRCLE_IDX, EOS_IDX, SOL_IDX, EXT_IDX]:
            return None
        i, current_row = get_row(vector, i, row_length)
        if i is None:
            return None
        rows.append(current_row)        

    return np.array(rows)

def vec2sv(vec, is_mat=False):
    if is_mat:
        mat = vec
    else:
        mat = vector_to_matrix(vec)
    if mat is None:
        return -1, -1
    cad = CADSequence.from_vector(mat, is_numerical=True, n=256)
    cad3d = create_CAD(cad)
    
    return compute_surface_area_and_volume(cad3d)

def matrix_to_vector(matrix):
    flattened_array = matrix.flatten()
    result_vector = flattened_array[flattened_array != -1]
    return result_vector

def get_output_sv(samples):
    output_sv = []
    mat = samples.cpu().to(torch.int32).numpy()
    valid_count = 0
    for output_vec in mat:
        # print(output_vec)
        try:
            area, vol = vec2sv(output_vec, is_mat=False)
        except:
            area, vol = -1, -1
        if not area == vol == -1:
            valid_count += 1
            sv_data = {"area": area, "vol": vol}
            sv_data = normalize_data(sv_data)
            area, vol = sv_data["area"], sv_data["vol"]
        output_sv.append([area, vol])  
    return output_sv, valid_count

def cut_at_eos(seq, eos_idx, pad_idx):
    # seq: [T]
    out = []
    for t in seq:
        out.append(t)
        if t == eos_idx:
            break
    # 把后面全 PAD
    if len(out) < len(seq):
        out = out + [pad_idx] * (len(seq) - len(out))
    return np.array(out)

def get_output_metrics(samples: torch.Tensor):
    mat_np: np.ndarray = samples.cpu().to(torch.int32).numpy()   # (B, L)
    metrics_list = []
    valid_count, _ = samples.shape

    for output_vec in mat_np:
        try:
            output_vec = cut_at_eos(output_vec, EOS_IDX, PAD_IDX)
            mat   = vector_to_matrix(output_vec)
            cad   = CADSequence.from_vector(mat, is_numerical=True, n=256)
            shape = create_CAD(cad)                  # TopoDS_Shape
            solid = Solid(shape)

            metrics: Dict[str, float] = compute_shape_metrics(shape, solid)
            if metrics["edge"] >= 0 and metrics["curved_edges"] >= 0:
                metrics["linear_edges"] = metrics["edge"] - metrics["curved_edges"]
            else:
                metrics["linear_edges"] = -1

        except Exception: 
            metrics = {
                "area": -1,  "vol":  -1,
                "planar_faces":  -1, "curved_faces": -1,
                "linear_edges": -1, "curved_edges": -1,
            }

        if any(metrics[k] == -1 for k in ("area", "vol")):
            valid_count -= 1

        metrics_list.append([
            metrics["area"],            
            metrics["vol"],        
            metrics["planar_faces"],
            metrics["curved_faces"],
            metrics["linear_edges"],
            metrics["curved_edges"],
        ])

    if metrics_list:
        metrics_torch = torch.tensor(metrics_list, dtype=torch.float32)
    else:
        metrics_torch = torch.empty((0, 6), dtype=torch.float32)

    return metrics_torch, valid_count

    
        
