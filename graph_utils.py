"""
Shared graph-building utilities extracted from
GAE_working_BEST18Apr.ipynb / GAE_working_BEST23Apr_spectral.ipynb.
Imported by GRASPED_04May.ipynb so the logic lives in one place.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


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
            if not df.empty:
                active_set = active_set.intersection(df.columns)
        active = list(active_set)

    return active


def correlation_matrix(returns, t, K, eps=0.0, active=None):
    t = pd.to_datetime(t)
    window = returns.loc[t - pd.Timedelta(days=K * 1.5): t]

    if active is not None:
        window = window[active]
    if eps == 0.0:
        window = window.loc[:, (window.fillna(0.0) != 0.0).any(axis=0)]
    else:
        window = window.loc[:, (window.fillna(0.0).abs() > eps).any(axis=0)]

    active_cols = window.columns.tolist()

    X = window.values
    X = np.nan_to_num(X)

    centered    = X - np.mean(X, axis=0)
    cov         = (centered.T @ centered) / (X.shape[0] - 1)
    std         = np.sqrt(np.diag(cov))
    denominator = np.outer(std, std) + 1e-9
    corr        = cov / denominator

    return np.clip(corr, -1.0, 1.0), active_cols


def create_adjacency_from_correlation(corr_matrix, threshold=0.0):
    N          = corr_matrix.shape[0]
    C          = np.array(corr_matrix, dtype=np.float32)
    k_neighbors = 10
    np.fill_diagonal(C, 0.0)

    A_pos = np.zeros((N, N), dtype=np.float32)
    A_neg = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        abs_corr = np.abs(C[i])
        k        = min(k_neighbors, N - 1)
        top_k    = np.argpartition(abs_corr, -k)[-k:]
        for j in top_k:
            if C[i, j] > 0:
                A_pos[i, j] = C[i, j]
            elif C[i, j] < 0:
                A_neg[i, j] = abs(C[i, j])

    A_pos = (A_pos + A_pos.T) / 2
    A_neg = (A_neg + A_neg.T) / 2

    return torch.tensor(A_pos, dtype=torch.float32), torch.tensor(A_neg, dtype=torch.float32)


def prepare_node_features(stocks, Z_DATA, t, norm_window=63):
    rows = []
    t    = pd.to_datetime(t)
    for stock in stocks:
        feats = []
        for key in Z_DATA:
            df  = Z_DATA[key]
            val = df.loc[t, stock] if (stock in df.columns and t in df.index) else 0.0
            feats.append(float(val))
        rows.append(feats)
    features = np.array(rows, dtype=np.float32)
    return torch.tensor(np.nan_to_num(features), dtype=torch.float32)


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
    U, S, Vt   = np.linalg.svd(data_to_svd, full_matrices=False)
    V          = Vt.T
    H          = V @ np.diag(S)
    H          = H[:, :10]

    if H.shape[1] < 10:
        H = np.hstack([H, np.zeros((H.shape[0], 10 - H.shape[1]))])

    scaler = StandardScaler()
    H      = scaler.fit_transform(H)
    H      = np.nan_to_num(H, nan=0.0)

    for i, stock in enumerate(active_cols):
        node_embeddings[stock] = np.array(H[i, :])

    return node_embeddings, active_cols
