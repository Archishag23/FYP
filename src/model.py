import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, PNAConv
import torch_geometric.utils as pyg_utils
import time
import networkx as nx
import numpy as np
from scipy import sparse

from layer import (
    DeconvWiener,
    PairNorm,
    MLP,
    SparseGraphWaveletLayer,
    DenseGraphWaveletLayer,
    FNN,
    MLP_generator,
)
from WaveConv import WaveConv
from utils import obtain_act, obtain_norm, check_for_nan
from wavelet_spar import WaveletSparsifier


class WDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layer,
        gamma,
        kernel="gcn",
        drop_ratio=0,
        skip=False,
        act="prelu",
        norm=None,
        dec_aggr="sum",
        large=False,
        beta=1.0,
    ):
        super(WDecoder, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = [in_dim] + [hid_dim] * (num_layer - 1) + [out_dim]
        self.drop_ratio = drop_ratio

        self.skip = skip
        self.beta = beta
        self.norm = norm

        self.decs = torch.nn.ModuleList()
        self.dec_acts = torch.nn.ModuleList()
        if self.norm:
            self.dec_norms = torch.nn.ModuleList()

        for i in range(self.num_layer):
            if i == self.num_layer - 1:
                large = large and len(gamma) > 1
                self.decs.append(
                    DeconvWiener(
                        self.emb_dim[i],
                        self.emb_dim[i + 1],
                        act=act,
                        gamma=gamma,
                        kernel=kernel,
                        transform=dec_aggr,
                        large=large,
                    )
                )
                self.dec_acts.append(obtain_act())
            else:
                self.decs.append(
                    DeconvWiener(
                        self.emb_dim[i],
                        self.emb_dim[i + 1],
                        act=act,
                        gamma=gamma,
                        kernel=kernel,
                        transform=dec_aggr,
                    )
                )
                self.dec_acts.append(obtain_act(act))

                if self.norm:
                    self.dec_norms.append(obtain_norm(self.norm)(self.emb_dim[i + 1]))

    def forward(self, enc, edge_index, edge_weight, avg_edge_index, avg_edge_weight):

        coef = enc[-1].std().item() * self.beta
        dec_list = [
            enc[-1]
            + coef * torch.normal(0, 1, size=enc[-1].size(), device=enc[-1].device)
        ]

        for layer in range(self.num_layer):
            if layer == 0:
                d = self.decs[layer](
                    dec_list[layer],
                    edge_index,
                    edge_weight,
                    avg_edge_index,
                    avg_edge_weight,
                )
            else:
                if self.skip:
                    adv_enc = enc[-(layer + 1)]
                    coef = adv_enc.std().item() * self.beta
                    adv_enc = adv_enc + coef * torch.normal(
                        0, 1, size=adv_enc.size(), device=enc[-(layer + 1)].device
                    )
                    d1 = self.decs[layer](
                        adv_enc,
                        edge_index,
                        edge_weight,
                        avg_edge_index,
                        avg_edge_weight,
                    )

                    adv_dec = dec_list[layer]
                    d2 = self.decs[layer](
                        adv_dec,
                        edge_index,
                        edge_weight,
                        avg_edge_index,
                        avg_edge_weight,
                    )

                    d = d1 + d2
                else:
                    adv_dec = dec_list[layer]
                    d = self.decs[layer](
                        adv_dec,
                        edge_index,
                        edge_weight,
                        avg_edge_index,
                        avg_edge_weight,
                    )

            d = (
                self.dec_norms[layer](d)
                if self.norm and layer < self.num_layer - 1
                else d
            )
            d = F.dropout(
                self.dec_acts[layer](d), self.drop_ratio, training=self.training
            )
            dec_list.append(d)

        return dec_list[-1]

    def fast(self, enc, adj, avg_adj):

        coef = enc[-1].std().item() * self.beta
        dec_list = [
            enc[-1]
            + coef * torch.normal(0, 1, size=enc[-1].size(), device=enc[-1].device)
        ]

        for layer in range(self.num_layer):
            if layer == 0:
                d = self.decs[layer](dec_list[layer], adj, None, avg_adj, None)
            else:
                if self.skip:
                    adv_enc = enc[-(layer + 1)]
                    coef = adv_enc.std().item() * self.beta
                    adv_enc = adv_enc + coef * torch.normal(
                        0, 1, size=adv_enc.size(), device=enc[-(layer + 1)].device
                    )
                    d1 = self.decs[layer](adv_enc, adj, None, avg_adj, None)

                    adv_dec = dec_list[layer]
                    d2 = self.decs[layer](adv_dec, adj, None, avg_adj, None)

                    d = d1 + d2
                else:
                    adv_dec = dec_list[layer]
                    d = self.decs[layer](adv_dec, adj, None, avg_adj, None)

            d = (
                self.dec_norms[layer](d)
                if self.norm and layer < self.num_layer - 1
                else d
            )
            d = F.dropout(
                self.dec_acts[layer](d), self.drop_ratio, training=self.training
            )
            dec_list.append(d)

        return dec_list[-1]


# Main Autoencoder structure here
class GNNStructEncoder(nn.Module):
    def __init__(
        self,
        args,
        dataset,
        in_dim,
        hidden_dim,
        layer_num,
        sample_size,
        device,
        neighbor_num_list,
        norm_mode="PN-SCS",
        norm_scale=20,
        lambda_loss1=0.01,
        lambda_loss2=0.001,
        lambda_loss3=0.0001,
    ):
        super(GNNStructEncoder, self).__init__()
        self.args = args
        self.device = device
        self.in_dim = in_dim
        self.hid_dim = hidden_dim
        self.out_dim = hidden_dim
        self.dataset = dataset
        self.mlp0 = nn.Linear(self.in_dim, self.hid_dim)
        self.norm = PairNorm(norm_mode, norm_scale)

        self.sample_size = sample_size
        self.lambda_loss1 = lambda_loss1  # neighbor reconstruction loss weight
        self.lambda_loss2 = lambda_loss2  # feature loss weight
        self.lambda_loss3 = lambda_loss3  # degree loss weight

        self.neighbor_num_list = neighbor_num_list  # neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        self.GNN_name = self.args.encoder
        # GNN Encoder
        if self.GNN_name == "GIN":
            self.linear1 = MLP(layer_num, self.in_dim, self.hid_dim, self.hid_dim)
            self.graphconv1 = GINConv(self.linear1)
            self.linear2 = MLP(layer_num, self.hid_dim, self.hid_dim, self.hid_dim)
            self.graphconv2 = GINConv(self.linear2)
        elif self.GNN_name == "GCN":
            self.graphconv1 = GCNConv(self.in_dim, self.hid_dim)
            self.graphconv2 = GCNConv(self.hid_dim, self.hid_dim)
        elif self.GNN_name == "GAT":
            self.graphconv1 = GATConv(self.in_dim, self.hid_dim)
            self.graphconv2 = GATConv(self.hid_dim, self.hid_dim)
        elif self.GNN_name == "WaveNet":
            self.graphconv1 = WaveConv(
                self.dataset, self.in_dim, self.hid_dim, self.args
            )
            self.graphconv2 = WaveConv(
                self.dataset, self.hid_dim, self.hid_dim, self.args
            )
        elif self.GNN_name == "GWNN":
            self.graphconv1 = SparseGraphWaveletLayer(
                self.in_dim, self.hid_dim, self.tot_node, self.device
            )
            # (in_channels, out_channels, ncount, self.device)
            self.graphconv2 = DenseGraphWaveletLayer(
                self.hid_dim, self.hid_dim, self.tot_node, self.device
            )
        else:
            self.graphconv1 = SAGEConv(self.in_dim, self.hid_dim, aggr=args.aggregator)
            self.graphconv2 = SAGEConv(self.hid_dim, self.hid_dim, aggr=args.aggregator)

        # For reconstruction_neighbors
        self.m = torch.distributions.Normal(
            torch.zeros(sample_size, self.hid_dim),
            torch.ones(sample_size, self.hid_dim),
        )

        # For reconstruction_neighbors2
        self.m_batched = torch.distributions.Normal(
            torch.zeros(sample_size, self.tot_node, self.hid_dim),
            torch.ones(sample_size, self.tot_node, self.hid_dim),
        )

        # For reconstruction_neighbors2
        self.mean_agg = SAGEConv(
            self.in_dim, self.hid_dim, aggr=self.args.aggregator, normalize=False
        )
        self.std_agg = PNAConv(
            self.in_dim,
            self.hid_dim,
            aggregators=["std"],
            scalers=["identity"],
            deg=neighbor_num_list,
        )

        # For reconstruction_neighbors and reconstruction_neighbors2
        self.mean_mlp = FNN(self.hid_dim, self.hid_dim, self.hid_dim, 3)
        self.sigma_mlp = FNN(self.hid_dim, self.hid_dim, self.hid_dim, 3)

        self.layer1_generator = MLP_generator(self.hid_dim, self.hid_dim)

        # Decoders
        self.degree_decoder = FNN(self.hid_dim, self.hid_dim, 1, 4)
        if self.args.simple_dec:
            self.dec_layer = 1
        else:
            self.dec_layer = self.args.gnn_layer_num

        if self.args.feat_decoder == "Deconv":
            self.feature_decoder = WDecoder(
                self.hid_dim,
                self.hid_dim,
                self.in_dim,
                self.dec_layer,
                self.args.gamma,
                self.args.dec_kernel,
                self.args.drop_ratio,
                self.args.skip,
                self.args.act,
                self.args.norm,
                self.args.dec_aggr,
                self.args.large,
                self.args.beta,
            )

        elif self.args.feat_decoder == "MLP":
            self.feature_decoder = FNN(self.hid_dim, self.hid_dim, self.in_dim, 3)
        else:
            print(
                "Error: Not available feature decoder. You could only choose: 'Deconv' or 'MLP'."
            )
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()

    def setup_wavelet(self, sparse_fusion_feat_coo):
        indices = np.array([sparse_fusion_feat_coo.row, sparse_fusion_feat_coo.col])
        self.feature_indices = torch.tensor(indices, dtype=torch.long)

        self.feature_indices = self.feature_indices.to(self.device)
        self.feature_values = (
            torch.FloatTensor(sparse_fusion_feat_coo.data).view(-1).to(self.device)
        )

        nonzero_indices = np.vstack(self.sparsifier.phi_matrices[0].nonzero())
        self.phi_indices = torch.tensor(nonzero_indices, dtype=torch.long).to(
            self.device
        )

        self.phi_values = torch.FloatTensor(
            self.sparsifier.phi_matrices[0][self.sparsifier.phi_matrices[0].nonzero()]
        )
        self.phi_values = self.phi_values.view(-1).to(self.device)

        nonzero_indices = np.vstack(self.sparsifier.phi_matrices[1].nonzero())
        self.phi_inverse_indices = torch.tensor(nonzero_indices, dtype=torch.long).to(
            self.device
        )

        self.phi_inverse_values = torch.FloatTensor(
            self.sparsifier.phi_matrices[1][self.sparsifier.phi_matrices[1].nonzero()]
        )
        self.phi_inverse_values = self.phi_inverse_values.view(-1).to(self.device)

    def compute_phi(self):
        try:
            data = self.dataset.data
        except:
            data = self.dataset

        print("Computing wavelet bases for GWNN.")
        print()

        compute_wavelet_start = time.time()
        adj = pyg_utils.to_dense_adj(data.edge_index)[0]
        adj_np = adj.cpu().numpy()
        np.fill_diagonal(adj_np, 0)
        self.G = nx.from_numpy_array(adj_np)
        print("Nexworkx graph is established.")
        print()

        self.sparsifier = WaveletSparsifier(
            self.G, self.args.scale, self.args.approximation_order, self.args.tolerance
        )
        self.sparsifier.calculate_all_wavelets()
        sparse_feat_coo = sparse.coo_matrix(data.x.detach().cpu().numpy())
        self.setup_wavelet(sparse_feat_coo)
        compute_wavelet_end = time.time()

        print("Wavelet bases for GWNN computed!")
        print(
            f"Total time taken: {compute_wavelet_end - compute_wavelet_start:.2f} seconds"
        )  # 打印总耗时
        print()

    def forward_encoder(self, x, edge_index):
        # Apply graph convolution and activation, pair-norm to avoid trivial solution
        if self.GNN_name == "WaveNet":
            enc_list = [x]

            l1 = self.graphconv1(x)
            enc_list.append(l1)
            l2 = self.graphconv2(l1)
            enc_list.append(l2)

        elif self.GNN_name == "GWNN":
            enc_list = [x]

            self.compute_phi()
            l1 = self.graphconv1(
                self.phi_indices,
                self.phi_values,
                self.phi_inverse_indices,
                self.phi_inverse_values,
                self.feature_indices,
                self.feature_values,
                self.args.dropout,
            )
            enc_list.append(l1)
            l2 = self.graphconv2(
                self.phi_indices,
                self.phi_values,
                self.phi_inverse_indices,
                self.phi_inverse_values,
                l1,
            )
            enc_list.append(l2)

        else:
            enc_list = [x]

            l1 = self.graphconv1(x, edge_index)
            enc_list.append(l1)
            l2 = self.graphconv2(l1, edge_index)
            enc_list.append(l2)
        return enc_list

    def neighbor_decoder(
        self,
        encs,
        ground_truth_degree_matrix,
        x,
        edge_index,
        edge_weight,
        avg_edge_index,
        avg_edge_weight,
    ):
        # gij is latent embeddings
        # Degree decoder below:
        gij = encs[-1]
        tot_nodes = gij.shape[0]

        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(
            degree_logits, ground_truth_degree_matrix.float()
        )  # MSELoss
        degree_loss_per_node = (degree_logits - ground_truth_degree_matrix).pow(2)
        try:
            check_for_nan(degree_loss, "Degree Logits")
        except ValueError as e:
            print(e)
        try:
            check_for_nan(degree_loss_per_node, "Degree Logits")
        except ValueError as e:
            print(e)

        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        feature_loss = 0

        # layer 1
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []

        # Sample multiple times to remove noise
        for _ in range(3):
            if self.args.feat_decoder == "Deconv":
                h0_prime = self.feature_decoder(
                    encs, edge_index, edge_weight, avg_edge_index, avg_edge_weight
                )
            elif self.args.feat_decoder == "MLP":
                h0_prime = self.feature_decoder(gij)

            feature_losses_per_node = (x - h0_prime).pow(2).mean(1)  # MSELoss
            feature_loss_list.append(feature_losses_per_node)

            local_index_loss, local_index_loss_per_node = (
                self.reconstruction_neighbors2(gij, x, edge_index)
            )
            local_index_loss, local_index_loss_per_node = (
                self.reconstruction_neighbors2(gij, x, edge_index)
            )
            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)

        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)

        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node, dim=0)

        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list), dim=0)
        print("$" * 30)
        print("feature_loss_per_node: ")
        print(feature_loss_per_node)
        print("$" * 30)
        feature_loss += torch.mean(torch.stack(feature_loss_list))
        print("$" * 30)
        print("feature_loss: ")
        print(feature_loss)
        print("$" * 30)

        h_loss_per_node = h_loss_per_node.reshape(tot_nodes, 1)
        degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes, 1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes, 1)

        loss = (
            self.lambda_loss1 * h_loss
            + self.lambda_loss2 * feature_loss
            + self.lambda_loss3 * degree_loss
        )
        loss_per_node = (
            self.lambda_loss1 * h_loss_per_node
            + self.lambda_loss2 * feature_loss_per_node
            + self.lambda_loss3 * degree_loss_per_node
        )

        return (
            loss,
            h_loss,
            feature_loss,
            degree_loss,
            loss_per_node,
            h_loss_per_node,
            degree_loss_per_node,
            feature_loss_per_node,
        )

    def reconstruction_neighbors2(self, l2, x, edge_index):
        recon_loss = 0
        recon_loss_per_node = []
        epsilon = 1e-8

        target_mean = self.mean_agg(x, edge_index).detach()
        target_std = self.std_agg(x, edge_index).detach()

        target_cov = torch.bmm(
            target_std.unsqueeze(dim=-1), target_std.unsqueeze(dim=1)
        )
        check_for_nan(target_cov, "Target Covariance")

        self_embedding = l2
        self_embedding = self_embedding.unsqueeze(0)
        self_embedding = self_embedding.repeat(self.sample_size, 1, 1)

        generated_mean = self.mean_mlp(self_embedding)
        generated_logstd = self.sigma_mlp(self_embedding)

        std_z = self.m_batched.sample().to(self.device)
        var = generated_mean + generated_logstd.exp() * std_z

        nhij = self.layer1_generator(var)

        generated_mean = torch.mean(nhij, dim=0)
        generated_std = torch.std(nhij, dim=0)
        generated_cov = (
            torch.bmm(generated_std.unsqueeze(dim=-1), generated_std.unsqueeze(dim=1))
            / self.sample_size
        )  # [total nodes, h_dim, h_dim]

        h_dim = l2.shape[1]

        single_eye = torch.eye(h_dim).to(self.device).unsqueeze(dim=0)
        batch_eye = single_eye.repeat(
            self.tot_node, 1, 1
        )  # [total nodes, h_dim, h_dim]

        target_cov = target_cov + batch_eye
        generated_cov = generated_cov + batch_eye

        # det + log approach (kept for reference; prone to NaN on near-singular matrices)
        # det_target_cov = torch.linalg.det(target_cov)
        # det_generated_cov = torch.linalg.det(generated_cov)
        # if torch.any(det_generated_cov == 0):
        #     raise ValueError(
        #         "Generated Covariance contains zero values, which could lead to division by zero."
        #     )
        # log_det_ratio = torch.log(
        #     (det_target_cov + epsilon) / (det_generated_cov + epsilon)
        # )
        # check_for_nan(log_det_ratio, "Log Determinant Ratio")

        # slogdet is numerically stable: returns (sign, log|det|) without
        # computing det explicitly, avoiding log(negative) NaN on near-singular matrices.
        _, logdet_target = torch.linalg.slogdet(target_cov)
        _, logdet_gen = torch.linalg.slogdet(generated_cov)
        log_det_ratio = logdet_target - logdet_gen

        trace_term = torch.einsum(
            "bij,bji->b", torch.linalg.inv(generated_cov), target_cov
        )

        mean_diff = generated_mean - target_mean
        quad_term = torch.einsum(
            "bi,bij,bj->b", mean_diff, torch.linalg.inv(generated_cov), mean_diff
        )

        KL_loss = 0.5 * (log_det_ratio - h_dim + trace_term + quad_term)
        check_for_nan(KL_loss, "KL Loss")

        recon_loss = torch.mean(KL_loss)
        recon_loss_per_node = KL_loss

        return recon_loss, recon_loss_per_node

    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def forward(
        self,
        edge_index,
        edge_weight,
        avg_edge_index,
        avg_edge_weight,
        x,
        ground_truth_degree_matrix,
        neighbor_dict,
    ):
        # Generate GNN encodings
        encs = self.forward_encoder(x, edge_index)
        (
            loss,
            h_loss,
            feature_loss,
            degree_loss,
            loss_per_node,
            h_loss_per_node,
            degree_loss_per_node,
            feature_loss_per_node,
        ) = self.neighbor_decoder(
            encs,
            ground_truth_degree_matrix,
            x,
            edge_index,
            edge_weight,
            avg_edge_index,
            avg_edge_weight,
        )
        return (
            loss,
            h_loss,
            feature_loss,
            degree_loss,
            loss_per_node,
            h_loss_per_node,
            degree_loss_per_node,
            feature_loss_per_node,
        )
