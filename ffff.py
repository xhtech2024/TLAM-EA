import torch
from torch_geometric.utils import *
import numpy as np
edge_index = torch.tensor([
     [0, 1, 1, 2, 2, 3],[1, 0, 2, 1, 3, 2],])
adj = to_scipy_sparse_matrix(edge_index)
# `edge_index` and `edge_weight` are both returned

ss = from_scipy_sparse_matrix(adj)

ff = np.random.uniform(-0.25, 0.25, 200).round(6).tolist()
print(ss)
