import os
import sys

sys.path.append("./preprocess")
import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Amazon, Actor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
from sklearn.decomposition import PCA
import pygod.utils

from preprocess.decompose_graph import generate_node_data
from preprocess.synthetic_anomaly import generate_synthetic


class Sample_data:
    """
    The class of customize of datasets
        Sample_data.name: name of dataset
        Sample_data.eigenvalues: eigenvalues
        Sample_data.vectors: vectors
        Sample_data.data: node feature
        Sample_data.num_classes: num_classes
        Sample_data.num_features: num_features
    """

    def __init__(self, root, name):
        file_path = f"{root}{name}_decom.pt"
        print(f"Loading data from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")

        tep = torch.load(file_path)
        self.eigenvalues = tep[0]
        self.vectors = tep[1]
        self.data = Data(x=tep[2], y=tep[3], edge_index=tep[4])
        self.num_classes = int(self.data.y.max() + 1)
        self.num_features = int(self.data.x.shape[1])
        self.name = f"{name}"


def walk_hop(edge_index, centernode):
    indices = torch.where(edge_index[0] == centernode)
    result = torch.stack([edge_index[0][indices], edge_index[1][indices]], dim=0)
    return result


def sample_subgraph(root, name, hop=3):
    file_path = f"{root}/processed/data.pt"
    data = torch.load(file_path)
    try:
        data = data[0]
    except:
        pass
    edge_index = data.edge_index
    start_node = torch.tensor(0)
    neighbors = walk_hop(edge_index, start_node)
    walked_nodes = torch.unique(neighbors[0])
    extended_nodes = torch.unique(neighbors[1])
    print("----Sampling-----")
    for _ in range(hop):
        for center_node in extended_nodes:
            if center_node not in walked_nodes:
                tep = walk_hop(edge_index, center_node)
                extended_nodes = torch.cat((extended_nodes, tep[1]), dim=0)
                neighbors = torch.cat([neighbors, tep], dim=1)
            walked_nodes = torch.unique(neighbors[0])
    print("----neighbors.is_undirected:-----", is_undirected(neighbors))
    if not is_undirected(neighbors):
        neighbors = to_undirected(neighbors)

    node_index = torch.unique(neighbors[0])
    node_feature = data.x[node_index]
    node_label = data.y[node_index]

    res = Data(
        x=node_feature,
        edge_index=neighbors,
        y=node_label,
    )
    torch.save(res, f"./testspace/subgraph_{name}.pt")
    print("----Done!-----", is_undirected(neighbors))


def decompose_matrix(name, dataset, suffix):
    if suffix is not None:
        print(f"Deriving decompose matrix of Dataset {name}_{suffix}...")
    else:
        print(f"Deriving decompose matrix of Dataset {name}...")
    generate_node_data(name, dataset, suffix=suffix)
    print("Process done!")


def normalize_features(data, attrs=["x"]):
    """
    Row-normalizes the attributes given in `attrs` to sum up to one.

    Args:
        data (Union[Data, HeteroData]): The graph data object.
        attrs (List[str]): The names of attributes to normalize (default: ["x"]).

    Returns:
        Union[Data, HeteroData]: The modified graph data object with normalized attributes.
    """
    for store in getattr(data, "stores", [data]):
        for attr in attrs:
            if hasattr(store, attr):
                value = getattr(store, attr)
                if value.numel() > 0:
                    value = value - value.min()
                    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.0))
                    setattr(store, attr, value)
    return data


def load_data_from_pygod(
    data_path, data_name, transform=normalize_features, suffix=None
):
    """
    Load graph data using pygod with local data check, download, and optional transformation.
    Args:
        data_path (str): Directory to store and load the dataset files.
        data_name (str): Name of the dataset to load.
        transform (callable, optional): Function to apply transformations on the data.
        suffix (str, optional): Additional suffix to specify dataset variations (e.g., 'pca_100').
    Returns:
        Graph data: Loaded and optionally transformed graph data.
    """
    os.makedirs(data_path, exist_ok=True)

    if suffix is not None:
        data_path = os.path.join(data_path, f"{data_name}_{suffix}.pt")
    else:
        data_path = os.path.join(data_path, f"{data_name}.pt")

    # Check if data already exists locally
    if not os.path.exists(data_path):
        print(
            f"No local copy of {data_name} with suffix {suffix} found, downloading..."
        )
        data = pygod.utils.load_data(data_name)  # Load the dataset
        if suffix is not None and suffix.startswith("pca_"):
            try:
                data = data.data
            except:
                data = data

            # Update data with reduced features
            feat_matrix = data.x

            # Perform PCA if specified
            target_dim = int(
                suffix.split("_")[1]
            )  # Extract target dimension from suffix
            original_dim = feat_matrix.shape[1]  # Original feature dimension
            if target_dim > original_dim:
                raise ValueError(
                    f"PCA target dimension {target_dim} is greater than the original dimension {original_dim}."
                )
            print(
                f"Performing PCA to reduce dimension from {original_dim} to {target_dim}..."
            )
            pca = PCA(n_components=target_dim)
            feat_matrix = torch.tensor(
                pca.fit_transform(feat_matrix.cpu().numpy()), dtype=torch.float32
            )

            # Update data with reduced features
            data.x = feat_matrix

        # Save the dataset locally
        print(f"Saving dataset to {data_path}...")
        torch.save(data, data_path)
    else:
        print(
            f"Dataset {data_name} with suffix {suffix}  already exists at {data_path}. Loading dataset from local storage."
        )
        # Load the dataset from local storage
        data = torch.load(data_path)

    # Apply the transformation, if provided
    if transform:
        normalize = T.NormalizeFeatures()
        data = normalize(data)

    return data


def PreDataLoader(data_name, net, suffix=None, synthetic_para=None):
    origin_data_name = data_name.lower()
    data_name = origin_data_name
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if synthetic_para is not None:
        data_name = (
            "inj_" + data_name + "_" + synthetic_para[0] + "_" + str(synthetic_para[1])
        )  # e.g., inj_cora_ctx_1

    if suffix is not None:
        path = f"{current_dir}/../../decomposed_data/{data_name}_{suffix}_decom.pt"
    else:
        path = f"{current_dir}/../../decomposed_data/{data_name}_decom.pt"

    # Make sure selected original dataset is decomposed
    # Decomposed version is stored in "../../decomposed_data"
    # Original version is stored in "../../data"
    if not os.path.exists(path):  # preprocess datasets
        data_path = f"{current_dir}/../../data"
        if origin_data_name in ["weibo", "books", "reddit", "disney", "enron"]:
            dataset = load_data_from_pygod(
                data_path, origin_data_name, transform=normalize_features, suffix=suffix
            )
        elif origin_data_name in ["cora", "citeseer", "pubmed"]:
            dataset = Planetoid(
                data_path, origin_data_name, transform=T.NormalizeFeatures()
            )
            dataset = dataset[0]
        elif origin_data_name in ["chameleon", "squirrel"]:
            dataset = WikipediaNetwork(
                data_path, origin_data_name, transform=T.NormalizeFeatures()
            )
        elif origin_data_name in ["amazon"]:
            dataset = Amazon(data_path, "computers", transform=T.NormalizeFeatures())
            dataset = dataset[0]
        elif origin_data_name in ["actor"]:
            dataset = Actor(f"{data_path}/actor")

        if synthetic_para is not None:
            if not isinstance(synthetic_para, list):
                raise TypeError(
                    f"Expected 'synthetic_para' to be a list, but got {type(synthetic_para).__name__}"
                )
            try:
                dataset = generate_synthetic(dataset, synthetic_para)
            except ValueError as e:
                raise ValueError(f"Error in generating synthetic data: {e}")

        decompose_matrix(data_name, dataset, suffix)

    # Choose decompsed dataset or original one according to the GNN to be employed
    if net == "WaveNet":
        if suffix is not None:
            name = f"{data_name}_{suffix}"
            dataset = Sample_data(
                root=f"{current_dir}/../../decomposed_data/", name=name
            )
        else:
            dataset = Sample_data(
                root=f"{current_dir}/../../decomposed_data/", name=data_name
            )
    else:
        # If WaveNet is not employed, use original dataset without decomposed
        dataset = load_dataset_wo_decomp(data_name, suffix)

    if suffix is not None:
        full_data_name = f"{data_name}_{suffix}"
    else:
        full_data_name = data_name
    return dataset, full_data_name


def load_dataset_wo_decomp(data_name, suffix):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f"{current_dir}/../../data"
    if data_name in ["weibo", "books", "reddit", "disney", "enron"]:
        dataset = load_data_from_pygod(
            data_path, data_name, transform=normalize_features, suffix=suffix
        )
    elif data_name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(data_path, data_name, transform=T.NormalizeFeatures())
    elif data_name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(
            data_path, data_name, transform=T.NormalizeFeatures()
        )
    elif data_name in ["amazon"]:
        dataset = Amazon(data_path, "computers", transform=T.NormalizeFeatures())
    elif data_name in ["actor"]:
        dataset = Actor(f"{data_path}/actor")
    return dataset
