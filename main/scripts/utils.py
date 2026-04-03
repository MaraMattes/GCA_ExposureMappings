import random
import torch, os, math
import numpy as np, networkx as nx, math
from scipy.special import gamma as GammaF
from scipy.spatial import cKDTree
from torch_geometric.utils import  from_scipy_sparse_matrix
from scipy.sparse import csr_matrix

def ball_vol(d,r): # used to construct RGG
    return math.pi**(d/2) * r**d / GammaF(d/2+1)


def gen_RGG_edge_index(positions, r): 
    n = positions.shape[0]
    kdtree = cKDTree(positions)
    pairs = kdtree.query_pairs(r)  # Pairs of nodes within distance r

    # Create NetworkX graph and convert to Scipy sparse matrix
    RGG = nx.empty_graph(n=n, create_using=nx.Graph())
    RGG.add_edges_from(list(pairs))

    # Convert to sparse adjacency matrix
    A_mat = nx.to_scipy_sparse_array(RGG, nodelist=range(n), format='csc')

    # Degree sequence for normalization
    deg_seq_sim = np.squeeze(A_mat.dot(np.ones(n)[:, None]))

    # Row-normalization (avoid division by zero)
    r, c = A_mat.nonzero()
    rD_sp = csr_matrix(((1.0 / np.maximum(deg_seq_sim, 1))[r], (r, c)), shape=A_mat.shape)
    A_norm = A_mat.multiply(rD_sp)

    # Convert to torch_geometric edge_index
    edge_index, edge_weight = from_scipy_sparse_matrix(A_norm)

    return A_mat, A_norm, edge_index, edge_weight


def set_environment_variables(parameters):

    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])
    torch.cuda.manual_seed(parameters['seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    
    parameters["local-path"] = os.getcwd()

    return parameters
