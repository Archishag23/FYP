import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import spspmm, spmm, matmul
from torch_geometric.nn.conv import MessagePassing
from functools import partial
import sympy as sym

from utils import obtain_act, approx_kernel_order, remez_approx, wiener_kernel_func


def gen_wave_series(mother_wave, scale, x=sym.symbols("x"), overlap=False):
    """
    mother_wave: sym.sybol; the original function which need to be modulated
    scale: int; the interval resolution scale, which decide the number of wave bases
    x: sym.symbol; dedicate the character of mother_wave
    overlap: bool; decide if construct overlaped base
    """
    expr = mother_wave
    wave_set = []
    num_split_interval = 0
    if overlap:
        for i in range(scale - 1, -scale, -1):
            wave_set.append(mother_wave.subs(x, scale * x + i))
            num_split_interval += 1
    else:
        for i in range(scale - 1, -scale, -2):
            wave_set.append(mother_wave.subs(x, scale * x + i))
            num_split_interval += 1

    return wave_set, num_split_interval


class Wave_prop(MessagePassing):
    def __init__(self, K, N, bias=True, **kwargs):
        super(Wave_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.len_eigen = N
        self.translation_max = self.len_eigen // self.K
        self.temp = Parameter(torch.Tensor(self.translation_max + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def Haar(self, x, scale):
        # x are eigenvalues
        x[x < 0] = 0  # Set negetive values to 0.
        start = 1 / self.K * scale
        end = start + 1 / self.K

        if end == 2:
            mask = torch.logical_and(x >= start, x <= end)
        mask = torch.logical_and(x >= start, x < end)
        A = torch.where(mask, torch.ones_like(x), torch.zeros_like(x))

        return A

    def Db2(self, x, translation):
        x[x < 0] = 0
        h = torch.tensor(
            [
                (1 + torch.sqrt(torch.tensor(3.0))) / 4 / torch.sqrt(torch.tensor(2.0)),
                (3 + torch.sqrt(torch.tensor(3.0))) / 4 / torch.sqrt(torch.tensor(2.0)),
                (3 - torch.sqrt(torch.tensor(3.0))) / 4 / torch.sqrt(torch.tensor(2.0)),
                (1 - torch.sqrt(torch.tensor(3.0))) / 4 / torch.sqrt(torch.tensor(2.0)),
            ]
        )

        A = torch.zeros_like(x)
        for k in range(len(h)):
            start = (1 / self.K * translation) + k / self.K
            end = start + 1 / self.K
            if end == 2:
                mask = torch.logical_and(x >= start, x <= end)
            mask = torch.logical_and(x >= start, x < end)
            A += torch.where(mask, h[k] * torch.ones_like(x), torch.zeros_like(x))

        return A

    def Db4(self, x, translation):
        x[x < 0] = 0

        h = torch.tensor(
            [
                (
                    1
                    + torch.sqrt(torch.tensor(10.0))
                    + torch.sqrt(torch.tensor(5.0) + torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    5
                    + torch.sqrt(torch.tensor(10.0))
                    + torch.sqrt(torch.tensor(5.0) - torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    5
                    - torch.sqrt(torch.tensor(10.0))
                    + torch.sqrt(torch.tensor(5.0) + torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    1
                    - torch.sqrt(torch.tensor(10.0))
                    + torch.sqrt(torch.tensor(5.0) - torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    1
                    - torch.sqrt(torch.tensor(10.0))
                    - torch.sqrt(torch.tensor(5.0) + torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    5
                    - torch.sqrt(torch.tensor(10.0))
                    - torch.sqrt(torch.tensor(5.0) - torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    5
                    + torch.sqrt(torch.tensor(10.0))
                    - torch.sqrt(torch.tensor(5.0) + torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
                (
                    1
                    + torch.sqrt(torch.tensor(10.0))
                    - torch.sqrt(torch.tensor(5.0) - torch.sqrt(torch.tensor(40.0)))
                )
                / 16,
            ]
        )

        A = torch.zeros_like(x)
        for k in range(len(h)):
            start = (1 / self.K * translation) + k / self.K
            end = start + 1 / self.K
            if end == 2:
                mask = torch.logical_and(x >= start, x <= end)
            mask = torch.logical_and(x >= start, x < end)
            A += torch.where(mask, h[k] * torch.ones_like(x), torch.zeros_like(x))

        return A

    def forward(self, x, eigen, eigen_vector, kernel="haar"):

        TEMP = (
            self.temp
        )  # (self.len_eigen // self.K + 1) of theta_k, learnable parameter
        sum_matrix = []

        print("Length of Eigen Values is: ", self.len_eigen)
        print(
            f"Range of Translation is: [0, {self.translation_max}]",
        )

        if kernel == "haar":
            Pro_Matrix = TEMP[0] * self.Haar(eigen, scale=0)
            for i in range(1, self.translation_max):
                Pro_Matrix += TEMP[i] * self.Haar(eigen, scale=i)
        elif kernel == "db2":
            Pro_Matrix = TEMP[0] * self.Db2(eigen, translation=0)
            for i in range(1, self.translation_max):
                Pro_Matrix += TEMP[i] * self.Db2(eigen, translation=i)
        elif kernel == "db4":
            Pro_Matrix = TEMP[0] * self.Db4(eigen, translation=0)
            for i in range(1, self.translation_max):
                Pro_Matrix += TEMP[i] * self.Db4(eigen, translation=i)
        else:
            raise ValueError(
                "The specified wavelet kernel is not supported in the current model."
            )

        Pro_Matrix = eigen_vector @ torch.diag(Pro_Matrix) @ eigen_vector.T
        x = Pro_Matrix @ x

        return x, Pro_Matrix

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)


class WaveletConvolution(nn.Module):
    def __init__(self, nnode, in_features, out_features, residual=False, variant=False):
        super(WaveletConvolution, self).__init__()
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.f = Parameter(torch.ones(self.nnode))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, support0, support1, h0, lamda, alpha, l=2):
        # l stands for the l-th layer of the GWCN
        beta = math.log(
            lamda / l + 1
        )  # beta is the ratio of the original feature transformation matrix
        bala1 = torch.spmm(
            support0, torch.diag(self.f)
        )  # f is an identical matrix with dimension of number of nodes
        bala2 = torch.mm(bala1, support1)
        hi = torch.mm(bala2, input)  # hi represents P'

        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (
                1 - alpha
            ) * hi + alpha * h0  # alpha represents the ratio of the initial residual term
        else:
            support = (1 - alpha) * hi + alpha * h0  # support represents H^{l'}
            r = support
        output = beta * torch.mm(support, self.weight) + (1 - beta) * r

        if self.residual:
            output = output + input

        return output


class GraphWaveletLayer(nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """

    def __init__(self, in_channels, out_channels, ncount, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.diagonal_weight_indices = torch.LongTensor(
            [
                [node for node in range(self.ncount)],
                [node for node in range(self.ncount)],
            ]
        )

        self.diagonal_weight_indices = self.diagonal_weight_indices.to(self.device)
        self.diagonal_weight_filter = nn.Parameter(torch.Tensor(self.ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight_matrix)


class SparseGraphWaveletLayer(GraphWaveletLayer):
    """
    Sparse Graph Wavelet Layer Class.
    """

    def forward(
        self,
        phi_indices,
        phi_values,
        phi_inverse_indices,
        phi_inverse_values,
        feature_indices,
        feature_values,
        dropout,
    ):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param dropout: Dropout rate.
        :return dropout_features: Filtered feature matrix extracted.
        """
        rescaled_phi_indices, rescaled_phi_values = spspmm(
            phi_indices,
            phi_values,
            self.diagonal_weight_indices,
            self.diagonal_weight_filter.view(-1),
            self.ncount,
            self.ncount,
            self.ncount,
        )

        phi_product_indices, phi_product_values = spspmm(
            rescaled_phi_indices,
            rescaled_phi_values,
            phi_inverse_indices,
            phi_inverse_values,
            self.ncount,
            self.ncount,
            self.ncount,
        )

        filtered_features = spmm(
            feature_indices,
            feature_values,
            self.ncount,
            self.in_channels,
            self.weight_matrix,
        )

        localized_features = spmm(
            phi_product_indices,
            phi_product_values,
            self.ncount,
            self.ncount,
            filtered_features,
        )

        dropout_features = torch.nn.functional.dropout(
            nn.functional.relu(localized_features), training=self.training, p=dropout
        )
        return dropout_features


class DenseGraphWaveletLayer(GraphWaveletLayer):
    """
    Dense Graph Wavelet Layer Class.
    """

    def forward(
        self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, features
    ):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param features: Feature matrix.
        :return localized_features: Filtered feature matrix extracted.
        """
        rescaled_phi_indices, rescaled_phi_values = spspmm(
            phi_indices,
            phi_values,
            self.diagonal_weight_indices,
            self.diagonal_weight_filter.view(-1),
            self.ncount,
            self.ncount,
            self.ncount,
        )

        phi_product_indices, phi_product_values = spspmm(
            rescaled_phi_indices,
            rescaled_phi_values,
            phi_inverse_indices,
            phi_inverse_values,
            self.ncount,
            self.ncount,
            self.ncount,
        )

        filtered_features = torch.mm(features, self.weight_matrix)

        localized_features = spmm(
            phi_product_indices,
            phi_product_values,
            self.ncount,
            self.ncount,
            filtered_features,
        )

        return localized_features


class DeconvWiener(MessagePassing):

    def __init__(
        self,
        in_dim,
        emb_dim,
        gamma,
        act="prelu",
        kernel="gcn",
        aggr="add",
        transform="sum",
        large=False,
        bias=True,
    ):
        super(DeconvWiener, self).__init__(aggr=aggr)
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.num_channel = len(gamma)
        self.transform = transform
        self.act = obtain_act(act)
        self.large = large

        self.order = approx_kernel_order(kernel)
        self.func = partial(wiener_kernel_func, kernel=kernel)

        self.lin = nn.ModuleList()
        for _ in range(self.num_channel):
            if self.large and self.num_channel > 1:
                lin = nn.Linear(in_dim, in_dim, bias=bias)
            else:
                if self.num_channel > 1 and self.transform == "concat":
                    lin = nn.Linear(in_dim, emb_dim // self.num_channel, bias=bias)
                else:
                    lin = nn.Linear(in_dim, emb_dim, bias=bias)
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                lin.bias.data.fill_(0.0)
            self.lin.append(lin)

        if self.large and self.num_channel > 1:
            self.fuse1 = nn.Linear(in_dim, emb_dim, bias=bias)
            nn.init.xavier_uniform_(self.fuse1.weight)
            if self.fuse1.bias is not None:
                self.fuse1.bias.data.fill_(0.0)

        if self.num_channel > 1:
            if self.transform == "sum":
                self.fuse = partial(torch.sum, dim=0)
            elif self.transform == "avg":
                self.fuse = partial(torch.mean, dim=0)
            elif self.transform == "max":
                self.fuse = partial(torch.max, dim=0)
            else:
                raise ValueError("Invalid aggregation method!")

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        avg_edge_index,
        avg_edge_weight,
        signal_mat=None,
        noise_src=None,
    ):

        # Signal estimation
        if signal_mat is not None:
            signal = (
                signal_mat.var(0).mean().detach()
                + signal_mat.pow(0).mean(1).mean().detach()
            )
        else:
            signal = x.var(0).sum().detach() + x.pow(2).mean(0).sum().detach()

        # Noise estimation
        if noise_src is not None:
            noise = noise_src
        else:
            diff = x - self.propagate(avg_edge_index, x=x, edge_weight=avg_edge_weight)
            noise = diff.pow(2).mean(0).sum().detach()

        message = x
        coefs = [None] * self.num_channel
        outs = [None] * self.num_channel
        for i in range(self.num_channel):
            penalty = (noise / (self.gamma[i] * signal)).item()
            coefs[i] = remez_approx(self.order, 0.0, 2.0, self.func, penalty)
            outs[i] = coefs[i][0] * message

        for k in range(self.order):
            message = self.propagate(edge_index, x=message, edge_weight=edge_weight)
            for i in range(self.num_channel):
                outs[i] += coefs[i][k + 1] * message

        for i in range(self.num_channel):
            outs[i] = self.lin[i](outs[i])
            if self.num_channel > 1:
                outs[i] = self.act(outs[i])

        if self.num_channel == 1:
            out = outs[0]
        else:
            if self.transform == "concat":
                out = torch.concat(outs, dim=1)
            else:
                out = torch.stack(outs)
                if self.transform == "max":
                    out = self.fuse(out)[0]
                else:
                    out = self.fuse(out)

            if self.large:
                out = self.fuse1(out)

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)

                if len(h.shape) > 2:
                    h = torch.transpose(h, 0, 1)
                    h = torch.transpose(h, 1, 2)

                h = self.batch_norms[layer](h)

                if len(h.shape) > 2:
                    h = torch.transpose(h, 1, 2)
                    h = torch.transpose(h, 0, 1)

                h = F.relu(h)

            return self.linears[self.num_layers - 1](h)


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding):
        neighbor_embedding = F.relu(self.linear(embedding))
        neighbor_embedding = F.relu(self.linear2(neighbor_embedding))
        neighbor_embedding = F.relu(self.linear3(neighbor_embedding))
        neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding


class PairNorm(nn.Module):
    def __init__(self, mode="PN", scale=10):

        assert mode in ["None", "PN", "PN-SI", "PN-SCS"]
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale
        # Scale can be set based on original data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == "None":
            return x
        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == "PN-SCS":
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


# FNN
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        x = F.relu(x)
        return x
