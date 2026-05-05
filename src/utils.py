import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import Data
from pygod.utils.utility import check_parameter

import numpy as np
import math
from functools import partial
from scipy.linalg import sqrtm
from sklearn.metrics import precision_recall_curve, auc


def eval_pr_auc(targets, scores): # gives me AUC-PR score
    """
    targets: np.array
    scores: np.array

    """
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    auc_pr = auc(recall, precision)
    return auc_pr


def _normalize(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / x_max
    return x_norm


def check_for_nan(tensor, message="Tensor", detailed=True):
    if torch.isnan(tensor).any():
        nan_positions = torch.where(torch.isnan(tensor))
        error_message = f"NaN detected in {message} at positions {nan_positions}."
        if detailed:
            error_message += f" Tensor details: {tensor}"
        raise ValueError(error_message)


def check_determinant(det_value, message="Matrix"):
    if (torch.abs(det_value) < 1e-8).all():
        raise ValueError(
            f"{message} has a determinant too close to zero. Determinant: {det_value}"
        )


def check_for_extreme_values(tensor, threshold=1e-8, message="Tensor"):
    if (tensor < threshold).any():
        positions = torch.where(tensor < threshold)
        raise ValueError(
            f"Extreme values detected in {message} at positions {positions}. Values: {tensor[positions]}"
        )


def kl_divergence_multivar(mu_q, cov_q, mu_p, cov_p): # q is predcited and p is target distr
    # Inverse and determinant of the target covariance matrix
    inv_cov_p = torch.linalg.inv(cov_p)
    det_cov_p = torch.det(cov_p)
    det_cov_q = torch.det(cov_q)

    # Trace term
    trace_term = torch.trace(torch.matmul(inv_cov_p, cov_q))

    # Difference in means
    diff_means = mu_p - mu_q
    quadratic_term = torch.matmul(
        torch.matmul(diff_means.unsqueeze(-2), inv_cov_p), diff_means.unsqueeze(-1)
    )

    # Log determinant term
    log_det_term = torch.log(det_cov_p) - torch.log(det_cov_q)

    # KL divergence
    kl = 0.5 * (trace_term + quadratic_term - mu_q.size(-1) + log_det_term)

    return kl.squeeze() # returns diff between the distributions


def gen_joint_structural_outlier(data, m, n, random_state=None):
    """
    We randomly select n nodes from the network which will be the anomalies
    and for each node we select m nodes from the network.
    We connect each of n nodes with the m other nodes.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(m, int):
        check_parameter(m, low=0, high=data.num_nodes, param_name="m")
    else:
        raise ValueError("m should be int, got %s" % m)

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name="n")
    else:
        raise ValueError("n should be int, got %s" % n)

    check_parameter(m * n, low=0, high=data.num_nodes, param_name="m*n")

    if random_state:
        np.random.seed(random_state)

    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    all_nodes = [i for i in range(data.num_nodes)]
    rem_nodes = []

    for node in all_nodes:
        if node is not outlier_idx:
            rem_nodes.append(node)

    new_edges = []
    # connect all m nodes in each clique
    for i in range(0, n):
        other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
        for j in other_idx:
            new_edges.append(torch.tensor([[i, j]], dtype=torch.long))

    new_edges = torch.cat(new_edges)

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier


def count_ones_and_zeros(tensor):
    ones_count = torch.sum(tensor).item()
    zeros_count = tensor.size(0) - ones_count
    ones_percentage = ones_count / tensor.size(0)
    zeros_percentage = zeros_count / tensor.size(0)
    return ones_count, zeros_count, ones_percentage, zeros_percentage


# ======================================================================
#   Graph normalization functions
# ======================================================================


def get_normalization(
    edge_index, num_nodes, improved=1.0, dtype=float, norm="sym", laplacian=True
):
    """
    Obtain the Laplacian/Normalized matrix generated from normalized adjacency matrix
    (For PyTorch Geometric)
    """

    fill_value = improved

    edge_weight = torch.ones(
        (edge_index.size(1),), dtype=dtype, device=edge_index.device
    )

    # 'add_remaining_self_loop' moves all self-link to the end of list
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )

    # Normalize
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if norm == "sym":
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif norm == "rw":
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = deg_inv[row] * edge_weight

    if laplacian:
        # Laplacian
        tmp_edge_weight = edge_weight.new_full((edge_weight.size(0),), 0)
        tmp_edge_weight[-num_nodes:] = 1.0
        edge_weight = tmp_edge_weight - edge_weight

    return edge_index, edge_weight


# ======================================================================
#   Model activation/normalization creation function
# ======================================================================


def obtain_act(name=None):
    """
    Return activation function module
    """
    if name == "relu":
        act = nn.ReLU()
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "elu":
        act = nn.ELU()
    elif name is None:
        act = nn.Identity()
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return act


def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return norm


def approx_kernel_order(kernel):
    """
    Return the order of remez approximation of different kernel
    """
    if kernel == "gcn":
        order = 9
    elif "heat" in kernel:
        order = 2
    elif "ppr" in kernel:
        order = 2

    return order


def wiener_kernel_func(x, kernel, penalty):
    """
    Construct wiener kernel function for approximation
    """
    if kernel not in ["gcn", "heat", "ppr"]:
        raise ValueError("Invalid convolution kernel!")

    if kernel == "gcn":
        conv = 1 - x
    elif kernel == "heat":
        conv = np.exp(-x)
    elif kernel == "ppr":
        conv = 1 / (4 * x + 1)

    return conv / (conv**2 + penalty)


def remez_init(n, left, right):
    """
    Chebyshev nodes for remez approximation
    """

    # Chebyshev nodes for initialization
    points = np.array([left] + [0] * n + [right])

    order = n + 2
    for i in range(order):
        p = ((2 * i + 1) / (2 * order)) * np.pi
        points[i] = (left + right) / 2 + ((right - left) / 2) * np.cos(p)

    points.sort()

    return points


def remez_step(n, points, func):
    """
    Coefficients calcuation
    """

    # Left-side of equation
    A = [np.repeat(1, n + 2)]
    for i in range(n):
        A.append(A[i] * points)
    A.append([(-1) ** (i % 2) for i in range(n + 2)])
    A = np.asarray(A).T

    # Right side of equation
    b = func(points)

    # Coefficients
    coefs = np.linalg.solve(A, b)

    return coefs[:-1]


def remez_enc_approx(n, left, right, func):
    """
    Compute the coefficients of nth-order remez polynomial approximation
    """

    # Initialization
    points = remez_init(n, left, right)

    # Compute coefficients
    coefs = remez_step(n, points, func)

    return coefs


def remez_approx(n, left, right, wiener_func, penalty):
    # diff to remez_enc_approx is that this one is for wiener kernel approximation, which requires a partial function with penalty as argument
    # penalty is a hyperparameter that controls the trade-off between approximation accuracy and stability in the wiener kernel approximation. 
    # A smaller penalty may lead to better approximation but can cause numerical instability, while a larger penalty can improve stability but may reduce approximation accuracy.
    """
    Compute the coefficients of nth-order remez polynomial approximation
    """

    # Initialization
    points = remez_init(n, left, right)

    # Construct partial function
    func = partial(wiener_func, penalty=penalty) 

    # Compute coefficients
    coefs = remez_step(n, points, func)

    return coefs


def KL_neighbor_loss(predictions, targets, mask_len, device):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]
    h_dim = x1.shape[1]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max((nn - 1), 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max((nn - 1), 1)

    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye

    KL_loss = 0.5 * (
        math.log(torch.det(cov_x1) / torch.det(cov_x2))
        - h_dim
        + torch.trace(torch.inverse(cov_x2).matmul(cov_x1))
        + (mean_x2 - mean_x1)
        .reshape(1, -1)
        .matmul(torch.inverse(cov_x2))
        .matmul(mean_x2 - mean_x1)
    )
    KL_loss = KL_loss.to(device)
    return KL_loss # diff to kl_divergende_multivar as this one is for neighbor loss and the other 
#one is for distribution diff loss, so this one adds an identity matrix to the covariance matrices 
# to ensure they are positive definite and to prevent numerical instability during inversion and 
# determinant calculation, while the other one does not add this regularization term.


def W2_neighbor_loss(predictions, targets, mask_len, device):

    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / (nn - 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / (nn - 1)

    W2_loss = torch.square(mean_x1 - mean_x2).sum() + torch.trace(
        cov_x1 + cov_x2 + 2 * sqrtm(sqrtm(cov_x1) @ (cov_x2.numpy()) @ (sqrtm(cov_x1)))
    )

    return W2_loss # diff to KL_neighbour_loss as this one calculates the Wasserstein-2 distance #
#between the predicted and target distributions, which consists of two terms: the squared Euclidean
#  distance between the means of the distributions and a term involving the covariance matrices 
# that captures the difference in their shapes. The KL_neighbor_loss, on the other hand, c
# alculates the Kullback-Leibler divergence, which measures how one probability distribution 
# diverges from a second, expected probability distribution.
