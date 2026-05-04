import sys
import os
import math
import pickle as pkl
import scipy as sp
from scipy import io
import numpy as np
import networkx as nx
import torch
from numpy.linalg import eig, eigh
from torch_sparse import SparseTensor


def generate_signal_data():
    data = io.loadmat("node_raw_data/2Dgrid.mat")
    A = data["A"]
    x = data["F"].astype(np.float32)
    m = data["mask"]

    A = sp.sparse.coo_matrix(A).todense()

    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    L = np.eye(10000) - D_invsqrt_corr @ A @ D_invsqrt_corr

    e, u = eigh(L)

    y_low = u @ np.diag(np.array([math.exp(-10 * (ee - 0) ** 2) for ee in e])) @ u.T @ x
    y_high = (
        u @ np.diag(np.array([1 - math.exp(-10 * (ee - 0) ** 2) for ee in e])) @ u.T @ x
    )
    y_band = (
        u @ np.diag(np.array([math.exp(-10 * (ee - 1) ** 2) for ee in e])) @ u.T @ x
    )
    y_rej = (
        u @ np.diag(np.array([1 - math.exp(-10 * (ee - 1) ** 2) for ee in e])) @ u.T @ x
    )
    y_comb = u @ np.diag(np.array([abs(np.sin(ee * math.pi)) for ee in e])) @ u.T @ x

    e = torch.FloatTensor(e)
    u = torch.FloatTensor(u)
    x = torch.FloatTensor(x)
    m = torch.LongTensor(m).squeeze()
    y_low = torch.FloatTensor(y_low)
    y_high = torch.FloatTensor(y_high)
    y_band = torch.FloatTensor(y_band)
    y_rej = torch.FloatTensor(y_rej)
    y_comb = torch.FloatTensor(y_comb)

    torch.save([e, u, x, y_low, m], "data/signal_low.pt")
    torch.save([e, u, x, y_high, m], "data/signal_high.pt")
    torch.save([e, u, x, y_band, m], "data/signal_band.pt")
    torch.save([e, u, x, y_rej, m], "data/signal_rej.pt")
    torch.save([e, u, x, y_comb, m], "data/signal_comb.pt")


def normalize_graph(adj):
    """
    Derive symmetric normalized graph Laplace matrix
    """
    adj = np.array(adj)
    adj = adj + adj.T
    adj[adj > 0.0] = 1.0
    deg = adj.sum(axis=1).reshape(-1)
    deg[deg == 0.0] = 1.0
    deg = np.diag(deg**-0.5)
    adj_tilde = np.dot(np.dot(deg, adj), deg)
    L = np.eye(adj.shape[0]) - adj_tilde
    return L


def eigen_decompositon(adj):
    "The normalized (unit “length”) eigenvectors,"
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    L = normalize_graph(adj)  # L: symmetric normalized graph Laplace matrix
    e, u = eigh(L)
    return e, u


def parse_index_file(filepath):
    """Parse index file."""
    index = []
    try:
        with open(filepath, "r") as file:
            for line in file:
                # Attempt to convert line to integer after stripping whitespace
                try:
                    index.append(int(line.strip()))
                except ValueError:
                    # If conversion fails, raise an error indicating which line and what value caused the issue
                    raise ValueError(
                        f"Unable to convert line to integer: {line.strip()}"
                    )
    except FileNotFoundError:
        # Raise an error if the file does not exist
        raise FileNotFoundError(f"The file {filepath} was not found.")
    except Exception as e:
        # Raise any other unexpected errors
        raise Exception(f"An error occurred: {e}")

    return index


def feature_normalize_row(x):
    x = np.array(x)
    min_val = x.min(axis=1, keepdims=True)
    x -= min_val  # Subtract the minimum value from each row
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum  # Normalize by row


def feature_normalize_col(x):
    x = np.array(x)
    min_val = x.min(axis=0, keepdims=True)
    x -= min_val  # Subtract the minimum value from each column
    colsum = x.sum(axis=0, keepdims=True)
    colsum = np.clip(colsum, 1, 1e10)
    return x / colsum  # Normalize by column


def load_data(dataset_str, current_dir):
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open(
            f"{current_dir}/../../data/{dataset_str}/raw/ind.{dataset_str}.{names[i]}",
            "rb",
        ) as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        f"{current_dir}/../../data/{dataset_str}/raw/ind.{dataset_str}.test.index"
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # Convert graph dictionary to edge_index tensor
    edge_index = []
    for src, neighbors in graph.items():
        for target in neighbors:
            edge_index.append([src, target])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return adj, features, labels, edge_index


def generate_node_data(data_name, dataset, suffix=None):
    """
    decompose dataset including:
    'cora', 'citeseer', 'pubmed'
    'chameleon', 'squirrel', 'photo', 'actor'
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if data_name in ["cora", "citeseer", "pubmed"]:

        adj, x, y, edge_index = load_data(data_name, current_dir)
        adj = adj.todense()
        x = x.todense()

        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(dataset.data.y.bool().long())
        edge = torch.LongTensor(edge_index)

        if suffix is not None:
            torch.save(
                [e, u, x, y, edge],
                f"{current_dir}/../../decomposed_data/{data_name}_{suffix}_decom.pt",
            )
        else:
            torch.save(
                [e, u, x, y, edge],
                f"{current_dir}/../../decomposed_data/{data_name}_decom.pt",
            )

    elif data_name in ["weibo", "books", "reddit", "disney", "enron"]:
        data = dataset

        max_index_based_count = data.edge_index.max().item() + 1
        feature_based_count = data.x.size(0)

        num_nodes = max(max_index_based_count, feature_based_count)
        print("Estimated number of nodes:", num_nodes)

        adj = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            sparse_sizes=(num_nodes, num_nodes),
        )
        print("Type of adj:", type(adj))
        print("Size of adj:", adj.size(0), ",", adj.size(1))

        adj = adj.to_dense()

        print("Type of adj:", type(adj))
        print("Size of adj:", adj.size())

        x = data.x
        y = data.y

        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y.bool().long())
        edge = torch.LongTensor(data.edge_index)

        if suffix is not None:
            torch.save(
                [e, u, x, y, edge],
                f"{current_dir}/../../decomposed_data/{data_name}_{suffix}_decom.pt",
            )
        else:
            torch.save(
                [e, u, x, y, edge],
                f"{current_dir}/../../decomposed_data/{data_name}_decom.pt",
            )

    elif data_name in ["chameleon", "squirrel", "photo", "actor"]:
        data = dataset.data
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
        adj = adj.to_dense()

        x = data.x
        y = data.y

        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y.bool().long())
        edge = torch.LongTensor(data.edge_index)

        if suffix is not None:
            torch.save(
                [e, u, x, y, edge],
                f"{current_dir}/../../decomposed_data/{data_name}_{suffix}_decom.pt",
            )
        else:
            torch.save(
                [e, u, x, y, edge],
                f"{current_dir}/../../decomposed_data/{data_name}_decom.pt",
            )

    else:
        data = dataset
        parts = data_name.split("_")

        if len(parts) == 4 and parts[0] == "inj":
            max_index_based_count = data.edge_index.max().item() + 1
            feature_based_count = data.x.size(0)

            num_nodes = max(max_index_based_count, feature_based_count)
            print("Estimated number of nodes:", num_nodes)

            adj = SparseTensor(
                row=data.edge_index[0],
                col=data.edge_index[1],
                sparse_sizes=(num_nodes, num_nodes),
            )
            print("Type of adj:", type(adj))
            print("Size of adj:", adj.size(0), ",", adj.size(1))

            adj = adj.to_dense()

            print("Type of adj:", type(adj))
            print("Size of adj:", adj.size())

            x = data.x
            y = data.y

            e, u = eigen_decompositon(adj)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y.bool().long())
            edge = torch.LongTensor(data.edge_index)

            if suffix is not None:
                torch.save(
                    [e, u, x, y, edge],
                    f"{current_dir}/../../decomposed_data/{data_name}_{suffix}_decom.pt",
                )
            else:
                torch.save(
                    [e, u, x, y, edge],
                    f"{current_dir}/../../decomposed_data/{data_name}_decom.pt",
                )
        else:
            print(
                f"Dataset {data_name} with suffix {suffix} is unexpected. Please check the code and settings"
            )
