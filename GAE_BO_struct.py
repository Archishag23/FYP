"""
GAE_BO_struct.py
================
Bayesian Optimisation for BubbleDetectionModel (GAE_working_13Apr_struct)
Objective: maximise VAL AUC (forward_window=22, crash_threshold=-0.2)

HYPERPARAMETERS TUNED
---------------------
 1. lr                  – learning rate              log-uniform [1e-4, 1e-2]
 2. weight_decay        – L2 regularisation          log-uniform [1e-6, 1e-3]
 3. hidden_dim          – GRU hidden dimension       categorical {16, 32, 64, 128}
 4. encoder_channels    – SpatialEncoder layer dims  categorical (see options below)
 5. num_diffusion_steps – diffusion steps (all layers) int [1, 4]
 6. dropout             – dropout probability        float [0.0, 0.4]
 7. gat_out_features    – GAT output dim per head    categorical {4, 8, 16, 32}
 8. gat_num_heads       – GAT attention heads        categorical {1, 2, 4}
 9. use_structure_recon – include structure recon loss categorical {True, False}
10. patience            – early-stopping patience    int [3, 10]
11. K                   – correlation lookback (days) int [15, 30]

NOT TUNED (require internal model/training code changes)
---------------------------------------------------------
 * self_loop_weight  – hardcoded default 0.5 inside SpatialEncoder;
                       BubbleDetectionModel does not forward this kwarg
 * k_neighbors       – hardcoded 10 inside train_model -> create_edge_index
 * embedding_dim     – fixed at 10 by compute_initial_node_embeddings SVD
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (verbatim from GAE_working_13Apr_struct.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

# ── Correlation matrix helper ─────────────────────────────────────────────────
def correlation_matrix(returns, t, K, eps=0.0, active=None):
    t = pd.to_datetime(t)
    window = returns.loc[t - pd.Timedelta(days=K * 1.5): t]
    window = window.dropna(axis=1, how='all')
    if active is not None:
        window = window[[c for c in active if c in window.columns]]
    if window.shape[0] < 2 or window.shape[1] == 0:
        return np.zeros((0, 0)), []
    if eps == 0.0:
        window = window.loc[:, ~(window.fillna(0.0) == 0.0).all(axis=0)]
    else:
        window = window.loc[:, ~(window.fillna(0.0).abs() <= eps).all(axis=0)]
    active_cols = window.columns.tolist()
    if not active_cols:
        return np.zeros((0, 0)), []
    data = window.fillna(0.0).values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr, active_cols


def compute_initial_node_embeddings(returns, t, K, eps=0.0, active=None):
    node_embeddings = {}
    t = pd.to_datetime(t)
    windowed_returns = returns.loc[t - pd.Timedelta(days=K * 1.5): t].dropna(how='all')
    if windowed_returns.empty:
        return {}, []
    window = windowed_returns.dropna(axis=1, how='all')
    if active is not None:
        valid_active = [c for c in active if c in window.columns]
        if not valid_active:
            return {}, []
        window = window[valid_active]
    if window.shape[0] < 2 or window.shape[1] == 0:
        return {}, []
    if eps == 0.0:
        window = window.loc[:, ~(window.fillna(0.0) == 0.0).all(axis=0)]
    else:
        window = window.loc[:, ~(window.fillna(0.0).abs() <= eps).all(axis=0)]
    active_cols = window.columns.tolist()
    if not active_cols:
        return {}, []
    data_to_svd = window.fillna(0.0).values
    data_to_svd = np.nan_to_num(data_to_svd, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(data_to_svd, full_matrices=False)
    V = Vt.T
    H = V @ np.diag(S)
    H = H[:, :10]
    if H.shape[1] < 10:
        padding = np.zeros((H.shape[0], 10 - H.shape[1]))
        H = np.hstack([H, padding])
    scaler = StandardScaler()
    H = scaler.fit_transform(H)
    H = np.nan_to_num(H, nan=0.0)
    for i, col in enumerate(active_cols):
        node_embeddings[col] = H[i]
    return node_embeddings, active_cols


# ── GATLayer ──────────────────────────────────────────────────────────────────
class GATLayer(torch.nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 2

    def __init__(self, num_in_features, num_out_features, num_of_heads,
                 concat=True, activation=nn.ELU(), dropout_prob=0.1,
                 add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None
        self.init_params()

    def forward(self, data):
        in_nodes_features, edge_index, corr = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(
            -1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = \
            self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        src = edge_index[self.src_nodes_dim]
        trg = edge_index[self.trg_nodes_dim]

        corr_e = corr[src, trg]
        mask_pos = (corr_e >= 0).float().unsqueeze(-1)
        mask_neg = (corr_e < 0).float().unsqueeze(-1)

        att_pos = self.neighborhood_aware_softmax(scores_per_edge, trg, num_of_nodes, mask_pos)
        att_neg = self.neighborhood_aware_softmax(scores_per_edge, trg, num_of_nodes, mask_neg)

        attentions_per_edge = torch.cat([att_pos, att_neg], dim=1)
        attentions_per_edge = self.dropout(attentions_per_edge)
        return attentions_per_edge

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes, mask):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp() * mask
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(
            exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype,
                                        device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index,
                             in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype,
                                         device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(
            edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted,
                                        nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(
            self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


# ── GAT ───────────────────────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.1, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        num_heads_per_layer = [1] + num_heads_per_layer
        gat_layers = []
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True if i < num_of_layers - 1 else False,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)
        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, data):
        return self.gat_net(data)


# ── DiffusionConvLayer ────────────────────────────────────────────────────────
class DiffusionConvLayer(nn.Module):
    def __init__(self, in_features, out_channels, num_diffusion_steps=1,
                 bias=True, self_loop_weight=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.num_diffusion_steps = num_diffusion_steps
        self.self_loop_weight = self_loop_weight

        self.theta_pos = nn.Parameter(torch.Tensor(num_diffusion_steps, in_features, out_channels))
        self.theta_neg = nn.Parameter(torch.Tensor(num_diffusion_steps, in_features, out_channels))

        if bias:
            self.bias_pos = nn.Parameter(torch.Tensor(out_channels))
            self.bias_neg = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta_pos)
        nn.init.xavier_uniform_(self.theta_neg)
        if self.bias_pos is not None:
            nn.init.zeros_(self.bias_pos)
        if self.bias_neg is not None:
            nn.init.zeros_(self.bias_neg)

    def forward(self, X_pos, X_neg, A_pos, A_neg):
        N = X_pos.shape[0]
        alpha = self.self_loop_weight
        D_pos = A_pos.sum(dim=1, keepdim=True) + 1e-8
        D_neg = A_neg.sum(dim=1, keepdim=True) + 1e-8
        A_pos_norm = A_pos / D_pos
        A_neg_norm = A_neg / D_neg
        I = torch.eye(N, device=X_pos.device)
        A_pos_norm = alpha * I + (1 - alpha) * A_pos_norm
        A_neg_norm = alpha * I + (1 - alpha) * A_neg_norm

        Z_pos = torch.zeros(N, self.out_channels, device=X_pos.device)
        Z_neg = torch.zeros(N, self.out_channels, device=X_neg.device)

        A_power = torch.eye(N, device=X_pos.device)
        for s in range(self.num_diffusion_steps):
            diffused = A_power @ X_pos
            Z_pos += diffused @ self.theta_pos[s]
            A_power = A_power @ A_pos_norm

        A_power = torch.eye(N, device=X_neg.device)
        for s in range(self.num_diffusion_steps):
            diffused = A_power @ X_neg
            Z_neg += diffused @ self.theta_neg[s]
            A_power = A_power @ A_neg_norm

        if self.bias_pos is not None:
            Z_pos = Z_pos + self.bias_pos
        if self.bias_neg is not None:
            Z_neg = Z_neg + self.bias_neg

        return Z_pos, Z_neg


# ── SpatialEncoder ────────────────────────────────────────────────────────────
class SpatialEncoder(nn.Module):
    def __init__(self, in_features_dim, hidden_channels, num_diffusion_steps=1,
                 dropout=0.1, self_loop_weight=0.5):
        super().__init__()
        layers = []
        channels = [in_features_dim] + hidden_channels
        for i in range(len(channels) - 1):
            layers.append(DiffusionConvLayer(
                in_features=channels[i],
                out_channels=channels[i + 1],
                num_diffusion_steps=num_diffusion_steps,
                self_loop_weight=self_loop_weight
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, Z_pos, Z_neg, A_pos, A_neg):
        for layer in self.layers:
            if isinstance(layer, DiffusionConvLayer):
                Z_pos, Z_neg = layer(Z_pos, Z_neg, A_pos, A_neg)
            else:
                Z_pos = layer(Z_pos)
                Z_neg = layer(Z_neg)
        return Z_pos, Z_neg


# ── GraphConvGRUCell ──────────────────────────────────────────────────────────
class GraphConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_diffusion_steps=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_r = DiffusionConvLayer(input_dim + hidden_dim, hidden_dim, num_diffusion_steps)
        self.conv_u = DiffusionConvLayer(input_dim + hidden_dim, hidden_dim, num_diffusion_steps)
        self.conv_c = DiffusionConvLayer(input_dim + hidden_dim, hidden_dim, num_diffusion_steps)

    def forward(self, x_t_pos, x_t_neg, h_prev_pos, h_prev_neg, A_pos, A_neg):
        combined_pos = torch.cat([x_t_pos, h_prev_pos], dim=1)
        combined_neg = torch.cat([x_t_neg, h_prev_neg], dim=1)

        r_pos, r_neg = self.conv_r(combined_pos, combined_neg, A_pos, A_neg)
        r_t_pos, r_t_neg = torch.sigmoid(r_pos), torch.sigmoid(r_neg)

        u_pos, u_neg = self.conv_u(combined_pos, combined_neg, A_pos, A_neg)
        u_t_pos, u_t_neg = torch.sigmoid(u_pos), torch.sigmoid(u_neg)

        h_tilde_pos = r_t_pos * h_prev_pos
        h_tilde_neg = r_t_neg * h_prev_neg
        combined_c_pos = torch.cat([x_t_pos, h_tilde_pos], dim=1)
        combined_c_neg = torch.cat([x_t_neg, h_tilde_neg], dim=1)
        c_pos, c_neg = self.conv_c(combined_c_pos, combined_c_neg, A_pos, A_neg)
        c_t_pos, c_t_neg = torch.tanh(c_pos), torch.tanh(c_neg)

        h_t_pos = u_t_pos * h_prev_pos + (1 - u_t_pos) * c_t_pos
        h_t_neg = u_t_neg * h_prev_neg + (1 - u_t_neg) * c_t_neg
        return h_t_pos, h_t_neg


# ── ReconstructionDecoder ─────────────────────────────────────────────────────
class ReconstructionDecoder(nn.Module):
    def __init__(self, hidden_dim, feature_dim, use_structure_recon=True):
        super().__init__()
        self.use_structure_recon = use_structure_recon
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, h_pos, h_neg, X_t, A_t_pos, A_t_neg):
        h_t = h_pos + h_neg
        X_hat = self.feature_decoder(h_t)
        feature_error = torch.norm(X_t - X_hat, dim=1)

        if self.use_structure_recon and A_t_pos is not None and A_t_neg is not None:
            A_hat_pos = torch.sigmoid(h_pos @ h_pos.T)
            A_hat_neg = torch.sigmoid(h_neg @ h_neg.T)
            struct_err_pos = torch.norm(A_t_pos - A_hat_pos, dim=1)
            struct_err_neg = torch.norm(A_t_neg - A_hat_neg, dim=1)
            structure_error = 0.5 * (struct_err_pos + struct_err_neg)
            loss_struct_pos = torch.mean((A_t_pos - A_hat_pos) ** 2)
            loss_struct_neg = torch.mean((A_t_neg - A_hat_neg) ** 2)
            loss_struct = 0.5 * (loss_struct_pos + loss_struct_neg)
            alpha = 1
            combined_error = alpha * feature_error + (1 - alpha) * structure_error
            loss_feat = torch.mean(feature_error ** 2)
            total_loss = alpha * loss_feat + (1 - alpha) * loss_struct
        else:
            A_hat_pos = None
            A_hat_neg = None
            combined_error = feature_error
            total_loss = torch.mean(feature_error ** 2)

        signals = combined_error
        return X_hat, A_hat_pos, A_hat_neg, total_loss, signals


# ── BubbleDetectionModel ──────────────────────────────────────────────────────
class BubbleDetectionModel(nn.Module):
    def __init__(self, gat_config, feature_dim, embedding_dim, encoder_channels,
                 hidden_dim, num_diffusion_steps=1, use_structure_recon=True, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.gat = GAT(
            num_of_layers=gat_config['num_layers'],
            num_heads_per_layer=gat_config['heads'],
            num_features_per_layer=gat_config['features'],
            dropout=dropout
        )
        self.spatial_encoder = SpatialEncoder(
            in_features_dim=feature_dim,
            hidden_channels=encoder_channels,
            num_diffusion_steps=num_diffusion_steps,
            dropout=dropout
        )
        spatial_out_dim = encoder_channels[-1]
        self.gru = GraphConvGRUCell(
            input_dim=spatial_out_dim + embedding_dim,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps
        )
        self.decoder = ReconstructionDecoder(
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            use_structure_recon=use_structure_recon
        )

    def forward(self, X_t, H_t, edge_index, corr_matrix,
                h_prev_pos=None, h_prev_neg=None, A_t_pos=None, A_t_neg=None):
        N = X_t.shape[0]
        if h_prev_pos is None:
            h_prev_pos = torch.zeros(N, self.hidden_dim, device=X_t.device)
        if h_prev_neg is None:
            h_prev_neg = torch.zeros(N, self.hidden_dim, device=X_t.device)

        attention_weights = self.gat((H_t, edge_index, corr_matrix))
        A_pos, A_neg = self.attention_to_adjacency(attention_weights, edge_index, N)

        Z_t_pos, Z_t_neg = self.spatial_encoder(X_t, X_t, A_pos, A_neg)

        Z_t_full_pos = torch.cat([Z_t_pos, H_t], dim=1)
        Z_t_full_neg = torch.cat([Z_t_neg, H_t], dim=1)

        h_t_pos, h_t_neg = self.gru(Z_t_full_pos, Z_t_full_neg, h_prev_pos, h_prev_neg, A_pos, A_neg)
        X_hat, A_hat_pos, A_hat_neg, loss, bubble_signals = self.decoder(
            h_t_pos, h_t_neg, X_t, A_t_pos, A_t_neg)

        return h_t_pos, h_t_neg, bubble_signals, loss, A_pos, A_neg

    def attention_to_adjacency(self, attention_weights, edge_index, num_nodes):
        NH = attention_weights.shape[1] // 2
        att_pos = attention_weights[:, :NH, 0].mean(dim=1)
        att_neg = attention_weights[:, NH:, 0].mean(dim=1)
        src = edge_index[0]
        trg = edge_index[1]
        A_pos = torch.zeros(num_nodes, num_nodes, device=attention_weights.device)
        A_neg = torch.zeros(num_nodes, num_nodes, device=attention_weights.device)
        A_pos[src, trg] = att_pos
        A_neg[src, trg] = att_neg
        A_pos = (A_pos + A_pos.T) / 2
        A_neg = (A_neg + A_neg.T) / 2
        return A_pos, A_neg


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (verbatim from GAE_working_13Apr_struct.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def get_active_stocks(returns, t, lookback_days, feature_dfs=None, min_obs=21, eps=0.0):
    t = pd.to_datetime(t)
    window = returns.loc[t - pd.Timedelta(days=lookback_days): t]
    counts = window.notna().sum(axis=0)
    ok_obs = counts >= min_obs
    if eps == 0.0:
        ok_nonzero = ~(window.fillna(0.0) == 0.0).all(axis=0)
    else:
        ok_nonzero = ~(window.fillna(0.0).abs() <= eps).all(axis=0)
    active = window.columns[ok_nonzero].tolist()
    if feature_dfs is not None:
        active_set = set(active)
        for df in feature_dfs:
            active_set &= set(df.columns)
        active = [s for s in active if s in active_set]
    return active


def create_adjacency_from_correlation(corr_matrix, threshold=0.0):
    N = corr_matrix.shape[0]
    C = np.array(corr_matrix, dtype=np.float32)
    k_neighbors = 10
    np.fill_diagonal(C, 0.0)
    abs_C = np.abs(C)
    k_actual = min(k_neighbors, N - 1)
    A_pos = np.zeros((N, N), dtype=np.float32)
    A_neg = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_k_idx = np.argsort(abs_C[i])[-k_actual:]
        for j in top_k_idx:
            if C[i, j] >= threshold:
                A_pos[i, j] = C[i, j]
            else:
                A_neg[i, j] = abs(C[i, j])
    A_pos_t = torch.tensor(A_pos, dtype=torch.float32)
    A_neg_t = torch.tensor(A_neg, dtype=torch.float32)
    return A_pos_t, A_neg_t


def create_edge_index(corr_matrix, k_neighbors=10):
    N = corr_matrix.shape[0]
    if isinstance(corr_matrix, np.ndarray):
        corr = torch.tensor(corr_matrix, dtype=torch.float32)
    else:
        corr = corr_matrix.clone()
    mask_diag = torch.eye(N, dtype=torch.bool, device=corr.device)
    corr.masked_fill_(mask_diag, float('-inf'))
    vals, indices = torch.topk(corr.abs(), k=min(k_neighbors, N - 1), dim=1)
    src_list = torch.arange(N, device=corr.device).repeat_interleave(k_neighbors)
    trg_list = indices.flatten()
    edge_index = torch.stack([src_list, trg_list], dim=0)
    return edge_index


def prepare_node_features(stocks, Z_DATA, t, norm_window=63):
    rows = []
    t = pd.to_datetime(t)
    for stock in stocks:
        feats = []
        for key in Z_DATA:
            df = Z_DATA[key]
            val = df.loc[t, stock] if stock in df.columns and t in df.index else 0.0
            feats.append(float(val))
        rows.append(feats)
    features = np.array(rows, dtype=np.float32)
    return torch.tensor(np.nan_to_num(features), dtype=torch.float32)


# ── EarlyStopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING / EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_model_bo(model, optimizer, returns, volatility, Market_caps, PE_ratios,
                   Implied_vol, Beta, RSI_momentum, Turnover, Z_DATA,
                   train_dates, stock2idx, patience=5, val_dates=None,
                   embedding_cache=None, K=21, save_path='checkpoints/bo_trial.pt',
                   n_epochs=20):
    """
    Wrapper around the original train_model logic with a configurable n_epochs
    cap for faster BO evaluation. All model logic is unchanged.
    """
    device = next(model.parameters()).device
    N_full = len(stock2idx)
    early_stopping = EarlyStopping(patience=patience, save_path=save_path)
    feature_dfs_list = [volatility, Market_caps, PE_ratios, Implied_vol, Beta, RSI_momentum, Turnover]

    train_loss_history = []
    val_loss_history = []

    for epoch in range(n_epochs):
        epoch_losses = []
        h_prev_pos_full = torch.zeros(N_full, model.hidden_dim, dtype=torch.float32).to(device)
        h_prev_neg_full = torch.zeros(N_full, model.hidden_dim, dtype=torch.float32).to(device)

        for t in train_dates:
            active = get_active_stocks(returns, t, lookback_days=int(K * 1.5),
                                       feature_dfs=feature_dfs_list, min_obs=K)
            if not active:
                continue
            returns_t = returns[active]
            stocks_t = active

            node_embeddings = embedding_cache.get(t)
            H_list = []
            for stock in stocks_t:
                if stock in node_embeddings:
                    emb = node_embeddings[stock]
                    if len(emb.shape) > 1 and emb.shape[0] == 1:
                        H_list.append(emb[0])
                    else:
                        H_list.append(emb)
                else:
                    H_list.append(np.zeros(10))
            H_t = torch.tensor(np.array(H_list), dtype=torch.float32)

            corr_matrix, _ = correlation_matrix(returns_t, t, K)
            if corr_matrix.shape[0] == 0:
                continue
            corr_t = torch.tensor(corr_matrix, dtype=torch.float32)
            edge_index = create_edge_index(corr_matrix, k_neighbors=10)
            X_t = prepare_node_features(stocks_t, Z_DATA, t)
            A_t_pos, A_t_neg = create_adjacency_from_correlation(corr_matrix, threshold=0.0)
            A_t_pos = A_t_pos.to(device)
            A_t_neg = A_t_neg.to(device)

            active_idx = torch.tensor([stock2idx[s] for s in stocks_t], dtype=torch.long).to(device)
            h_prev_pos = h_prev_pos_full.index_select(0, active_idx)
            h_prev_neg = h_prev_neg_full.index_select(0, active_idx)

            model.train()
            optimizer.zero_grad()
            h_t_pos, h_t_neg, signals, loss, A_pos, A_neg = model(
                X_t, H_t, edge_index, corr_t,
                h_prev_pos=h_prev_pos, h_prev_neg=h_prev_neg,
                A_t_pos=A_t_pos, A_t_neg=A_t_neg
            )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            h_prev_pos_full[active_idx] = h_t_pos.detach()
            h_prev_neg_full[active_idx] = h_t_neg.detach()
            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        if val_dates is not None:
            with torch.no_grad():
                h_val_pos_full = torch.zeros(N_full, model.hidden_dim).to(device)
                h_val_neg_full = torch.zeros(N_full, model.hidden_dim).to(device)
                for t in val_dates:
                    active = get_active_stocks(returns, t, lookback_days=int(K * 1.5),
                                               feature_dfs=feature_dfs_list, min_obs=K)
                    if not active:
                        continue
                    returns_t = returns[active]
                    stocks_t = active

                    node_embeddings = embedding_cache.get(t)
                    H_list = []
                    for stock in stocks_t:
                        if stock in node_embeddings:
                            emb = node_embeddings[stock]
                            if len(emb.shape) > 1 and emb.shape[0] == 1:
                                H_list.append(emb[0])
                            else:
                                H_list.append(emb)
                        else:
                            H_list.append(np.zeros(10))
                    H_t = torch.tensor(np.array(H_list), dtype=torch.float32).to(device)

                    corr_matrix, _ = correlation_matrix(returns_t, t, K)
                    if corr_matrix.shape[0] == 0:
                        continue
                    corr_t = torch.tensor(corr_matrix, dtype=torch.float32).to(device)
                    edge_index = create_edge_index(corr_matrix, k_neighbors=10)
                    X_t = prepare_node_features(stocks_t, Z_DATA, t).to(device)
                    A_t_pos, A_t_neg = create_adjacency_from_correlation(corr_matrix, threshold=0.0)
                    A_t_pos = A_t_pos.to(device)
                    A_t_neg = A_t_neg.to(device)

                    active_idx = torch.tensor([stock2idx[s] for s in stocks_t], dtype=torch.long).to(device)
                    h_val_prev_pos = h_val_pos_full.index_select(0, active_idx)
                    h_val_prev_neg = h_val_neg_full.index_select(0, active_idx)

                    h_t_pos, h_t_neg, signals, loss, A_pos, A_neg = model(
                        X_t, H_t, edge_index, corr_t,
                        h_prev_pos=h_val_prev_pos, h_prev_neg=h_val_prev_neg,
                        A_t_pos=A_t_pos, A_t_neg=A_t_neg
                    )
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_losses.append(loss.item())
                    h_val_pos_full[active_idx] = h_t_pos.detach()
                    h_val_neg_full[active_idx] = h_t_neg.detach()

        avg_train = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_val = np.mean(val_losses) if val_losses else float('inf')
        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)
        print(f"  Epoch {epoch+1}/{n_epochs}  train={avg_train:.5f}  val={avg_val:.5f}")

        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location='cpu'))
    return model, train_loss_history, val_loss_history


def run_val_inference(model, returns, volatility, Market_caps, PE_ratios, Implied_vol,
                      Beta, RSI_momentum, Turnover, Z_DATA, embedding_cache,
                      val_dates, stock2idx, K=21):
    """Run inference on val_dates to produce signals for AUC computation."""
    device = next(model.parameters()).device
    N_full = len(stock2idx)
    feature_dfs_list = [volatility, Market_caps, PE_ratios, Implied_vol, Beta, RSI_momentum, Turnover]

    h_prev_pos_full = torch.zeros(N_full, model.hidden_dim, device=device)
    h_prev_neg_full = torch.zeros(N_full, model.hidden_dim, device=device)

    all_signals, valid_dates, active_lists = [], [], []

    model.eval()
    with torch.no_grad():
        for t in val_dates:
            try:
                active = get_active_stocks(returns, t, lookback_days=int(K * 1.5),
                                           feature_dfs=feature_dfs_list, min_obs=K)
                if not active:
                    continue
                returns_t = returns[active]
                stocks_t = active

                node_embeddings = embedding_cache.get(t)
                H_list = []
                for stock in stocks_t:
                    if stock in node_embeddings:
                        H_list.append(node_embeddings[stock])
                    else:
                        H_list.append(np.zeros(10))
                H_t = torch.tensor(np.array(H_list), dtype=torch.float32).to(device)

                corr_matrix, _ = correlation_matrix(returns_t, t, K)
                if corr_matrix.shape[0] == 0:
                    continue
                corr_t = torch.tensor(corr_matrix, dtype=torch.float32).to(device)
                edge_index = create_edge_index(corr_matrix, k_neighbors=10)
                X_t = prepare_node_features(stocks_t, Z_DATA, t).to(device)

                active_idx = torch.tensor([stock2idx[s] for s in stocks_t], dtype=torch.long).to(device)
                h_prev_pos = h_prev_pos_full.index_select(0, active_idx)
                h_prev_neg = h_prev_neg_full.index_select(0, active_idx)

                h_t_pos, h_t_neg, signals, _, A_pos, A_neg = model(
                    X_t, H_t, edge_index, corr_t,
                    h_prev_pos=h_prev_pos, h_prev_neg=h_prev_neg,
                    A_t_pos=None, A_t_neg=None
                )
                h_prev_pos_full[active_idx] = h_t_pos.detach()
                h_prev_neg_full[active_idx] = h_t_neg.detach()

                all_signals.append(signals.cpu().numpy())
                valid_dates.append(t)
                active_lists.append(stocks_t)
            except Exception as e:
                continue

    return {d: (active_lists[i], all_signals[i]) for i, d in enumerate(valid_dates)}


def evaluate_once(test_results, prices, forward_window, crash_threshold):
    """Compute AUC for a given forward_window / crash_threshold pair."""
    y_true, y_scores = [], []
    sorted_dates = sorted(test_results.keys())
    valid_dates = [
        d for d in sorted_dates
        if d <= prices.index[-1] - pd.Timedelta(days=forward_window)
    ]
    for t in valid_dates:
        stocks, signals = test_results[t]
        if isinstance(signals, torch.Tensor):
            signals = signals.cpu().numpy()
        signals = signals.flatten()
        available = [s for s in stocks if s in prices.columns]
        if not available:
            continue
        mask = [i for i, s in enumerate(stocks) if s in available]
        signals = signals[mask]
        p_t = prices.loc[t, available]
        future_idx = prices.index.searchsorted(t + pd.Timedelta(days=forward_window))
        if future_idx >= len(prices):
            continue
        future_date = prices.index[future_idx]
        p_future = prices.loc[future_date, available]
        fwd_returns = (p_future - p_t) / p_t
        crash = (fwd_returns < crash_threshold).astype(int)
        y_true.extend(crash.values)
        y_scores.extend(signals)
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return None
    auc = roc_auc_score(np.array(y_true), np.array(y_scores))
    baseline = np.mean(y_true)
    precision = np.mean(np.array(y_true)[np.array(y_scores) > np.percentile(y_scores, 90)])
    lift = precision / baseline if baseline > 0 else float('nan')
    return auc, lift, baseline


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (run once — shared across all trials via closure)
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading prices...")
    prices = pd.read_excel('SPX_sectors_data.xlsx', header=[0, 1], index_col=0)
    prices.dropna(how='all', inplace=True)
    prices = prices.ffill().bfill()
    prices.columns = prices.columns.droplevel(1)

    all_stocks = prices.columns.tolist()
    stock2idx = {s: i for i, s in enumerate(all_stocks)}

    returns = pd.read_excel('SPX_sectors_data.xlsx', header=[0, 1], index_col=0)
    returns.columns = returns.columns.get_level_values(0)
    returns.dropna(how='all', inplace=True)
    returns = returns.pct_change().dropna(how='all')

    train_returns = returns.loc['2012-01-01':'2016-12-31'].ffill().bfill()
    val_returns   = returns.loc['2017-01-01':'2019-06-30'].ffill().bfill()

    train_volatility = train_returns.rolling(window=21).std().dropna(how='all') * np.sqrt(252)
    volatility = returns.rolling(window=21).std() * np.sqrt(252)

    def load_and_fix_index(filename):
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index, format='%m/%d/%Y')
        df.dropna(how='all', inplace=True)
        return df.ffill().bfill()

    Market_caps  = load_and_fix_index('Data/SPX_Constituents_market_cap_2006_2025(in).csv')
    PE_ratios    = load_and_fix_index('Data/SPX_Constituents_Calculated_PE_2006_2025(in).csv')
    Implied_vol  = load_and_fix_index('Data/SPX_Constituents_Implied_vol_2006_2025(in).csv')
    Beta         = load_and_fix_index('Data/SPX_Constituents_Beta_2006_2025(in).csv')
    RSI_momentum = load_and_fix_index('Data/SPX_Constituents_RSI_momentum_2006_2025(in).csv')
    Turnover     = load_and_fix_index('Data/SPX_Constituents_Turnover_30D_2006_2025(in).csv')

    def precompute_zscores(df, window=63):
        rolling_mean = df.rolling(window=window, min_periods=1).mean()
        rolling_std  = df.rolling(window=window, min_periods=1).std().fillna(1.0)
        z_scores = (df - rolling_mean) / (rolling_std + 1e-8)
        return z_scores.clip(-5.0, 5.0).fillna(0.0)

    Z_DATA = {
        'volatility':  precompute_zscores(volatility),
        'market_caps': precompute_zscores(Market_caps),
        'pe_ratios':   precompute_zscores(PE_ratios),
        'implied_vol': precompute_zscores(Implied_vol),
        'beta':        precompute_zscores(Beta),
        'rsi':         precompute_zscores(RSI_momentum),
        'turnover':    precompute_zscores(Turnover),
    }

    K_default = 21
    all_dates = sorted(
        set(train_returns.index) | set(val_returns.index)
    )
    print(f"Pre-computing embeddings for {len(all_dates)} dates...")
    embedding_cache = {}
    for t in tqdm(all_dates):
        node_embeddings, _ = compute_initial_node_embeddings(returns, t, K_default)
        embedding_cache[t] = node_embeddings

    # Use full prices for forward-return computation so val tail has enough runway
    val_prices_full = prices  # evaluate_once filters valid_dates internally

    val_dates = val_returns.index

    return dict(
        returns=returns,
        train_returns=train_returns,
        val_returns=val_returns,
        train_volatility=train_volatility,
        volatility=volatility,
        Market_caps=Market_caps,
        PE_ratios=PE_ratios,
        Implied_vol=Implied_vol,
        Beta=Beta,
        RSI_momentum=RSI_momentum,
        Turnover=Turnover,
        Z_DATA=Z_DATA,
        embedding_cache=embedding_cache,
        stock2idx=stock2idx,
        val_dates=val_dates,
        val_prices_full=val_prices_full,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

# BO evaluation settings
FORWARD_WINDOW    = 22
CRASH_THRESHOLD   = -0.20
BO_EPOCHS         = 20    # cap per trial to keep BO tractable (increase if time permits)

# Search-space options for encoder_channels (listed as strings to keep Optuna happy)
ENCODER_OPTIONS = {
    '[8,4]':       [8, 4],
    '[16,8]':      [16, 8],
    '[32,16]':     [32, 16],
    '[64,32]':     [64, 32],
    '[16,8,4]':    [16, 8, 4],
    '[32,16,8]':   [32, 16, 8],
}


def make_objective(data):
    """Returns a closure over the pre-loaded data."""

    def objective(trial):
        # Fix seed per trial so differences in AUC reflect hyperparameters, not init randomness
        torch.manual_seed(0)
        np.random.seed(0)

        # ── Sample hyperparameters ─────────────────────────────────────────
        lr                  = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay        = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hidden_dim          = trial.suggest_categorical('hidden_dim', [16, 32, 64])
        enc_key             = trial.suggest_categorical('encoder_channels', list(ENCODER_OPTIONS.keys()))
        encoder_channels    = ENCODER_OPTIONS[enc_key]
        num_diffusion_steps = trial.suggest_int('num_diffusion_steps', 1, 4)
        dropout             = trial.suggest_float('dropout', 0.0, 0.4)
        gat_out_features    = trial.suggest_categorical('gat_out_features', [4, 8, 16, 32])
        gat_num_heads       = trial.suggest_categorical('gat_num_heads', [1, 2, 4])
        #use_structure_recon = trial.suggest_categorical('use_structure_recon', [True, False])
        use_structure_recon = True
        patience            = trial.suggest_int('patience', 3, 10)
        K                   = trial.suggest_int('K', 15, 30)

        embedding_dim = 10  # fixed – set by compute_initial_node_embeddings SVD
        feature_dim   = 7   # fixed – [vol, mktcap, pe, ivol, beta, rsi, turnover]

        gat_config = {
            'num_layers': 1,
            'heads':    [gat_num_heads],
            'features': [embedding_dim, gat_out_features],
        }

        print(f"\n[Trial {trial.number}] lr={lr:.2e}  wd={weight_decay:.2e}  "
              f"hidden={hidden_dim}  enc={enc_key}  S={num_diffusion_steps}  "
              f"do={dropout:.2f}  gat_h={gat_num_heads}  gat_f={gat_out_features}  "
              f"struct={use_structure_recon}  pat={patience}  K={K}")

        # ── Build training dates with this trial's K ───────────────────────
        train_returns = data['train_returns']
        min_date      = train_returns.index[0] + pd.Timedelta(days=K * 2)
        train_dates   = train_returns.loc[min_date:].index

        val_returns  = data['val_returns']
        val_min_date = val_returns.index[0] + pd.Timedelta(days=K * 2)
        val_dates    = val_returns.loc[val_min_date:].index

        # ── Rebuild embedding_cache entries for val_dates with new K ───────
        # (train embeddings were pre-computed at K=21; for other K values we
        #  recompute on the fly only when K differs from 21)
        embedding_cache = data['embedding_cache']
        if K != 21:
            extra_dates = [t for t in val_dates if t not in embedding_cache]
            for t in extra_dates:
                node_embeddings, _ = compute_initial_node_embeddings(
                    data['returns'], t, K)
                embedding_cache[t] = node_embeddings

        # ── Initialise model ───────────────────────────────────────────────
        model = BubbleDetectionModel(
            gat_config=gat_config,
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            encoder_channels=encoder_channels,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps,
            use_structure_recon=use_structure_recon,
            dropout=dropout,
        )
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        os.makedirs('outputs/bo_checkpoints', exist_ok=True)
        save_path = f'outputs/bo_checkpoints/trial_{trial.number}.pt'

        # ── Train ──────────────────────────────────────────────────────────
        try:
            model, _, _ = train_model_bo(
                model, optimizer,
                data['returns'],
                data['train_volatility'],
                data['Market_caps'],
                data['PE_ratios'],
                data['Implied_vol'],
                data['Beta'],
                data['RSI_momentum'],
                data['Turnover'],
                data['Z_DATA'],
                train_dates,
                data['stock2idx'],
                patience=patience,
                val_dates=val_dates,
                embedding_cache=embedding_cache,
                K=K,
                save_path=save_path,
                n_epochs=BO_EPOCHS,
            )
        except Exception as e:
            print(f"  Training failed: {e}")
            traceback.print_exc()
            return 0.0

        # ── Validate – run inference on val_dates ──────────────────────────
        try:
            val_results = run_val_inference(
                model,
                data['returns'],
                data['volatility'],
                data['Market_caps'],
                data['PE_ratios'],
                data['Implied_vol'],
                data['Beta'],
                data['RSI_momentum'],
                data['Turnover'],
                data['Z_DATA'],
                embedding_cache,
                val_dates,
                data['stock2idx'],
                K=K,
            )
        except Exception as e:
            print(f"  Val inference failed: {e}")
            return 0.0

        # ── Compute AUC ────────────────────────────────────────────────────
        result = evaluate_once(val_results, data['val_prices_full'],
                               FORWARD_WINDOW, CRASH_THRESHOLD)
        if result is None:
            print("  AUC: N/A (no valid crash events in val window)")
            return 0.0

        auc, lift, baseline = result
        print(f"  VAL AUC={auc:.4f}  Lift={lift:.2f}  Baseline={baseline:.4f}")
        return auc

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("GAE Bayesian Optimisation")
    print(f"Objective: VAL AUC  (FW={FORWARD_WINDOW}, CT={CRASH_THRESHOLD})")
    print(f"Max epochs per trial: {BO_EPOCHS}")
    print("=" * 60)

    data = load_data()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='gae_struct_bo',
        storage='sqlite:///outputs/gae_bo.db',  # persists results for resume
        load_if_exists=True,
    )

    N_TRIALS = 50
    study.optimize(make_objective(data), n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  VAL AUC: {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save full results table
    df = study.trials_dataframe()
    os.makedirs('outputs', exist_ok=True)
    df.to_csv('outputs/gae_bo_results.csv', index=False)
    print("\nAll trial results saved to outputs/gae_bo_results.csv")
    print("Optuna DB saved to outputs/gae_bo.db  (resume with load_if_exists=True)")
