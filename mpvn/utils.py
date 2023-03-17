from typing import List, Dict
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def average_checkpoints(model: nn.modules, filenames: List[Path], device: torch.device = torch.device("cpu")) -> dict:
    n = len(filenames)

    avg = torch.load(filenames[0], map_location=device)['state_dict']

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        state_dict = torch.load(filenames[i], map_location=device)['state_dict']
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n
            
    model.load_state_dict(avg)
    
    return model
