#
# import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
# 在文件头部新增依赖
import gudhi
import networkx as nx
import scipy
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix

from sklearn.neighbors import kneighbors_graph

import networkx as nx
from scipy.sparse import csr_matrix


def build_alpha_complex_graph(pos, n_neighbors=10):
    """基于Alpha复合体构建几何感知空间邻接图（SpaceFlow兼容版）"""
    # Step 1: 用KNN估计图切割阈值
    A_knn = kneighbors_graph(pos, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / A_knn.count_nonzero()  # 避免除零需确保A_knn有非零元素

    # Step 2: 构建Alpha复合体骨架图
    alpha_complex = gudhi.AlphaComplex(points=pos.tolist())
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)  # 获取1-骨架（边集合）

    # Step 3: 转换为NetworkX图并移除自环
    initial_graph = nx.Graph()
    n_nodes = pos.shape[0]
    initial_graph.add_nodes_from(range(n_nodes))
    for simplex in skeleton:
        if len(simplex[0]) == 2:  # 过滤2- simplex（边）
            u, v = simplex[0]
            initial_graph.add_edge(u, v)

    # 移除自环（更高效的批量处理方式）
    initial_graph.remove_edges_from(nx.selfloop_edges(initial_graph))

    # Step 4: 转换为scipy稀疏矩阵（SpaceFlow要求的csr格式）
    sparse_array = nx.convert_matrix.to_scipy_sparse_array(initial_graph, format='csr')
    return csr_matrix(sparse_array)



##### generate n
def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    # sample_name = list(adata.uns['spatial'].keys())[0]
    # scalefactors = adata.uns['spatial'][sample_name]['scalefactors']
    # adj_mat = dist <= scalefactors['fiducial_diameter_fullres'] * (n+0.2)
    # adj_mat = adj_mat.astype(int)

    # n_neighbors = np.argpartition(dist, n+1, axis=1)[:, :(n+1)]
    # adj_mat = np.zeros((len(adata), len(adata)))
    # for i in range(len(adata)):
    #     adj_mat[i, n_neighbors[i, :]] = 1

    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### normalze graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def mask_generator(adj_label, N=1):
    idx = adj_label.indices()
    cell_num = adj_label.size()[0]

    list_non_neighbor = []
    for i in range(0, cell_num):
        neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
        n_selected = len(neighbor) * N

        # non neighbors
        total_idx = torch.range(0, cell_num-1, dtype=torch.float32)
        non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
        indices = torch.randperm(len(non_neighbor), dtype=torch.float32)
        random_non_neighbor = indices[:n_selected]
        list_non_neighbor.append(random_non_neighbor)

    x = adj_label.indices()[0]
    y = torch.concat(list_non_neighbor)

    indices = torch.stack([x, y])
    indices = torch.concat([adj_label.indices(), indices], axis=1)

    value = torch.concat([adj_label.values(), torch.zeros(len(x), dtype=torch.float32)])
    adj_mask = torch.sparse_coo_tensor(indices, value)

    return adj_mask


def graph_computing(pos, n):
    from scipy.spatial import distance
    list_x = []
    list_y = []
    list_value = []

    for node_idx in range(len(pos)):
        tmp = pos[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, pos, 'euclidean')
        res = distMat.argsort()
        # tmpdist = distMat[0, res[0][1:params.k + 1]]
        for j in np.arange(1, n + 1):
            list_x += [node_idx, res[0][j]]
            list_y += [res[0][j], node_idx]
            list_value += [1, 1]

    adj = sp.csr_matrix((list_value, (list_x, list_y)))
    adj = adj >= 1
    adj = adj.astype(np.float32)
    return adj


import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import torch
import scipy.sparse as sp
import numpy as np

def graph_construction(adata, n=6, dmax=50, mode='AlphaComplex', alpha_n_neighbors=10):
    """
    图构建函数（兼容SpaceFlow的AlphaComplex模式）
    返回包含归一化邻接矩阵、原始邻接矩阵标签和归一化值的字典
    """
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    elif mode == 'Distance':
        adj_m1 = generate_adj_mat_1(adata, dmax)
    elif mode == 'AlphaComplex':
        pos = adata.obsm['spatial']
        adj_m1 = build_alpha_complex_graph(pos, n_neighbors=alpha_n_neighbors)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # 确保邻接矩阵为csr格式（SpaceFlow要求）
    adj_m1 = adj_m1.tocsr()
    
    # 移除对角线元素（自环）
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal(), [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()
    
    # 预处理：添加自环
    adj_with_self = adj_m1 + sp.eye(adj_m1.shape[0])
    
    # 计算度矩阵的逆平方根
    degree = np.array(adj_with_self.sum(1))
    degree_inv_sqrt = np.power(degree, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    
    # 构建度矩阵的逆平方根的稀疏对角矩阵
    degree_inv_sqrt_sparse = sp.diags(degree_inv_sqrt)
    
    # 进行对称归一化
    adj_norm = degree_inv_sqrt_sparse.dot(adj_with_self).dot(degree_inv_sqrt_sparse)
    adj_norm = sp.csr_matrix(adj_norm)
    
    # 转换为PyTorch稀疏张量（COO格式）
    adj_coo = adj_m1.tocoo()
    indices = torch.stack([torch.from_numpy(adj_coo.row), torch.from_numpy(adj_coo.col)], dim=0)
    values = torch.from_numpy(adj_coo.data)
    shape = torch.Size(adj_coo.shape)
    adj_label = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    # 计算归一化值（SpaceFlow中的norm_value逻辑）
    num_edges = adj_m1.nnz
    num_nodes = adj_m1.shape[0]
    norm_value = num_nodes * num_nodes / float((num_nodes * num_nodes - num_edges) * 2)
        # 生成归一化邻接矩阵 adj_norm（假设 adj_norm 是 scipy.sparse 格式）
    adj_sparse = scipy.sparse.csr_matrix(adj_norm)
    
    # 转换为 PyTorch 稀疏张量
    adj_coo = adj_sparse.tocoo()
    indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
    values = torch.FloatTensor(adj_coo.data)
    shape = torch.Size(adj_coo.shape)
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape)
    
    graph_dict = {
        "adj_norm": adj_tensor,  # 直接传递 PyTorch 稀疏张量
        "adj_label": adj_label,
        "norm_value": norm_value
    }
    return graph_dict



def block_diag_sparse(*arrs):
        bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
        if bad_args:
            raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

        list_shapes = [a.shape for a in arrs]
        list_indices = [a.coalesce().indices().clone() for a in arrs]
        list_values = [a.coalesce().values().clone() for a in arrs]

        r_start = 0
        c_start = 0
        for i in range(len(arrs)):
            list_indices[i][0, :] += r_start
            list_indices[i][1, :] += c_start

            r_start += list_shapes[i][0]
            c_start += list_shapes[i][1]

        indices = torch.concat(list_indices, axis=1)
        values = torch.concat(list_values)
        shapes = torch.tensor(list_shapes).sum(axis=0)

        out = torch.sparse_coo_tensor(indices, values, (shapes[0], shapes[1]))

        return out


def combine_graph_dict(dict_1, dict_2):
    # TODO add adj_org
    tmp_adj_norm = block_diag_sparse(dict_1['adj_norm'], dict_2['adj_norm'])
    tmp_adj_label = block_diag_sparse(dict_1['adj_label'], dict_2['adj_label'])
    graph_dict = {
        "adj_norm": tmp_adj_norm.coalesce(),
        "adj_label": tmp_adj_label.coalesce(),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
    }
    return graph_dict



# def combine_graph_dict(dict_1, dict_2):
#     # TODO add adj_org
#     tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
#     tmp_adj_label = torch.block_diag(dict_1['adj_label'].to_dense(), dict_2['adj_label'].to_dense())
#     graph_dict = {
#         "adj_norm": tmp_adj_norm.to_sparse(),
#         "adj_label": tmp_adj_label.to_sparse(),
#         "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
#     }
#     return graph_dict