"""
Train GRASPED (04May config) over seeds [0, 2, 4, 6], average the per-date
anomaly signals, and save the ensemble result for evaluation.

Outputs:
  outputs/test_results_grasped_seeds_avg.pkl   – {date: (active_cols, avg_scores)}
  outputs/test_prices_grasped_seeds_avg.pkl    – {date: price_series}
  outputs/grasped_model_seed{s}.pt             – best checkpoint per seed
"""

import sys, os, io, pickle, warnings, random, argparse
from contextlib import redirect_stdout
warnings.filterwarnings('ignore')

SRC = os.path.join(os.getcwd(), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from scipy.linalg import eigh as scipy_eigh
from tqdm import tqdm

from graph_utils import get_active_stocks, correlation_matrix, prepare_node_features

# ── Patches ───────────────────────────────────────────────────────────────────
import WaveConv as _wc_module
import layer as _layer_module
from model import GNNStructEncoder
from utils import get_normalization

_Wave_prop = _layer_module.Wave_prop
_WaveConv  = _wc_module.WaveConv

def _wavconv_init_patched(self, dataset, in_channels, out_channels, args):
    nn.Module.__init__(self)
    self.args = args
    self.lin1 = Linear(in_channels, args.hidden_channels_wave)
    self.lin2 = Linear(args.hidden_channels_wave, out_channels)
    self.toy  = False
    eigen, eigen_vector = dataset.eigenvalues, dataset.vectors
    _dev = torch.device('cpu')
    self.eigen        = eigen.to(_dev)
    self.eigen_vector = eigen_vector.to(_dev)
    N = len(self.eigen)
    self.prop     = _Wave_prop(args.K, N)
    self.dprate   = args.dprate
    self.dropout  = args.dropout
    self.Pro_Matrix = None

_WaveConv.__init__ = _wavconv_init_patched

def _quiet_wave_forward(self, x, eigen, eigen_vector, kernel='haar'):
    TEMP = self.temp
    if kernel == 'haar':
        Pro_Matrix = TEMP[0] * self.Haar(eigen, scale=0)
        for i in range(1, self.translation_max):
            Pro_Matrix += TEMP[i] * self.Haar(eigen, scale=i)
    elif kernel == 'db2':
        Pro_Matrix = TEMP[0] * self.Db2(eigen, translation=0)
        for i in range(1, self.translation_max):
            Pro_Matrix += TEMP[i] * self.Db2(eigen, translation=i)
    elif kernel == 'db4':
        Pro_Matrix = TEMP[0] * self.Db4(eigen, translation=0)
        for i in range(1, self.translation_max):
            Pro_Matrix += TEMP[i] * self.Db4(eigen, translation=i)
    else:
        raise ValueError('Wavelet kernel not supported.')
    Pro_Matrix = eigen_vector @ torch.diag(Pro_Matrix) @ eigen_vector.T
    x = Pro_Matrix @ x
    return x, Pro_Matrix

_Wave_prop.forward = _quiet_wave_forward
print('Patches applied.')

# ── Args ──────────────────────────────────────────────────────────────────────
def make_grasped_args():
    args = argparse.Namespace()
    args.encoder              = 'WaveNet'
    args.wave_kernel          = 'haar'
    args.K                    = 9
    args.hidden_channels_wave = 64
    args.dprate               = 0.0
    args.dropout              = 0.2
    args.feat_decoder         = 'Deconv'
    args.gamma                = [0.1, 1.0, 10.0]
    args.dec_kernel           = 'heat'
    args.dec_aggr             = 'sum'
    args.simple_dec           = True
    args.gnn_layer_num        = 2
    args.skip                 = False
    args.large                = False
    args.beta                 = 0.23383278064157817
    args.act                  = 'prelu'
    args.norm                 = ''
    args.drop_ratio           = 0.0
    args.lambda_loss1         = 0.0
    args.lambda_loss2         = 0.5350470693780868
    args.lambda_loss3         = 0.0
    args.h_loss_weight        = 0.0
    args.feature_loss_weight  = 1.0
    args.degree_loss_weight   = 0.0
    args.aggregator           = 'mean'
    args.sample_size          = 10
    args.hidden_dimension     = 32
    args.dataset              = 'finance'
    args.norm_type            = 'sym'
    return args

ARGS   = make_grasped_args()
DEVICE = torch.device('cpu')
LOOKBACK_K = 21

# ── Data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices = pd.read_excel('SPX_sectors_data.xlsx', header=[0, 1], index_col=0)
prices.dropna(how='all', inplace=True)
prices = prices.ffill().bfill()
prices.columns = prices.columns.droplevel(1)

all_stocks = prices.columns.tolist()
stock2idx  = {s: i for i, s in enumerate(all_stocks)}
N_FULL     = len(all_stocks)
print(f'Universe: {N_FULL} stocks')

returns = prices.pct_change().dropna(how='all')
train_returns = returns.loc['2012-01-01':'2016-12-31'].ffill().bfill()
val_returns   = returns.loc['2017-01-01':'2019-06-30']
test_returns  = returns.loc['2019-07-01':'2024-12-31'].ffill().bfill()

K = LOOKBACK_K
train_dates = train_returns.loc[train_returns.index[0] + pd.Timedelta(days=K * 2):].index
val_dates   = val_returns.index
test_dates  = test_returns.loc[test_returns.index[0] + pd.Timedelta(days=K * 2):].index

print(f'Train: {len(train_dates)} | Val: {len(val_dates)} | Test: {len(test_dates)}')

def load_and_fix_index(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

Market_caps  = load_and_fix_index('Data/SPX_Constituents_market_cap_2006_2025(in).csv')
PE_ratios    = load_and_fix_index('Data/SPX_Constituents_Calculated_PE_2006_2025(in).csv')
Implied_vol  = load_and_fix_index('Data/SPX_Constituents_Implied_vol_2006_2025(in).csv')
Beta         = load_and_fix_index('Data/SPX_Constituents_Beta_2006_2025(in).csv')
RSI_momentum = load_and_fix_index('Data/SPX_Constituents_RSI_momentum_2006_2025(in).csv')
Turnover     = load_and_fix_index('Data/SPX_Constituents_Turnover_30D_2006_2025(in).csv')

train_volatility = train_returns.rolling(window=21).std().dropna(how='all') * np.sqrt(252)
test_volatility  = test_returns.rolling(window=21).std().dropna(how='all') * np.sqrt(252)
volatility = pd.concat([train_volatility, test_volatility]).sort_index()

feature_dfs_list = [volatility, Market_caps, PE_ratios, Implied_vol, Beta, RSI_momentum, Turnover]

def precompute_zscores(df, window=63):
    mu  = df.rolling(window=window, min_periods=1).mean()
    std = df.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return ((df - mu) / std).fillna(0.0)

Z_DATA = {
    'volatility':  precompute_zscores(volatility),
    'market_caps': precompute_zscores(Market_caps),
    'pe_ratios':   precompute_zscores(PE_ratios),
    'implied_vol': precompute_zscores(Implied_vol),
    'beta':        precompute_zscores(Beta),
    'rsi':         precompute_zscores(RSI_momentum),
    'turnover':    precompute_zscores(Turnover),
}
FEATURE_DIM = len(Z_DATA)
print(f'Feature dim: {FEATURE_DIM}')

# ── Graph / Eigen cache (built once, seed-independent) ────────────────────────
class _MockDataset:
    def __init__(self, eigenvalues, vectors):
        self.eigenvalues = eigenvalues
        self.vectors     = vectors


def compute_eigenbasis(A_full_np):
    A = np.array(A_full_np, dtype=np.float64)
    A = (A + A.T) / 2
    A[A > 0] = 1.0
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(deg ** -0.5)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    L = np.eye(A.shape[0]) - A_norm
    evals, evecs = scipy_eigh(L)
    return (torch.tensor(evals, dtype=torch.float32),
            torch.tensor(evecs, dtype=torch.float32))


class DailyGraphBuilder:
    def __init__(self, returns, all_stocks, stock2idx, lookback_k=21, k_neighbors=10):
        self.returns    = returns
        self.all_stocks = all_stocks
        self.stock2idx  = stock2idx
        self.N          = len(all_stocks)
        self.K          = lookback_k
        self.kn         = k_neighbors

    def build(self, t, feature_dfs=None):
        active = get_active_stocks(
            self.returns, t,
            lookback_days=int(self.K * 1.5),
            feature_dfs=feature_dfs,
            min_obs=self.K
        )
        if len(active) < 2:
            return None
        corr, active_cols = correlation_matrix(self.returns, t, self.K, active=active)
        n_local = len(active_cols)
        C = corr.copy()
        np.fill_diagonal(C, 0.0)
        A_full = np.zeros((self.N, self.N), dtype=np.float32)
        for i_loc in range(n_local):
            gi = self.stock2idx[active_cols[i_loc]]
            abs_c = np.abs(C[i_loc])
            k = min(self.kn, n_local - 1)
            top_k = np.argpartition(abs_c, -k)[-k:]
            for j_loc in top_k:
                gj = self.stock2idx[active_cols[j_loc]]
                A_full[gi, gj] = float(np.abs(corr[i_loc, j_loc]))
        A_full = (A_full + A_full.T) / 2
        rows, cols_arr = np.nonzero(A_full)
        src_list = rows.tolist() + list(range(self.N))
        dst_list = cols_arr.tolist() + list(range(self.N))
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        neighbor_dict = {i: [] for i in range(self.N)}
        for u, v in zip(src_list, dst_list):
            neighbor_dict[u].append(v)
        neighbor_num_list = torch.tensor(
            [len(neighbor_dict[i]) for i in range(self.N)], dtype=torch.long
        )
        global_idx = torch.tensor(
            [self.stock2idx[s] for s in active_cols], dtype=torch.long
        )
        return {
            'active_cols':       active_cols,
            'global_idx':        global_idx,
            'edge_index':        edge_index,
            'A_full':            A_full,
            'neighbor_dict':     neighbor_dict,
            'neighbor_num_list': neighbor_num_list,
        }


BUILDER = DailyGraphBuilder(returns, all_stocks, stock2idx, lookback_k=K)

all_dates = sorted(set(list(train_dates) + list(val_dates) + list(test_dates)))

print('Pre-computing graphs and eigenbases (once)...')
GRAPH_CACHE = {}
EIGEN_CACHE = {}
for t in tqdm(all_dates):
    r = BUILDER.build(t, feature_dfs=feature_dfs_list)
    if r is None:
        continue
    GRAPH_CACHE[t] = r
    evals, evecs = compute_eigenbasis(r['A_full'])
    EIGEN_CACHE[t] = (evals, evecs)
print(f'Cache built: {len(GRAPH_CACHE)} / {len(all_dates)} dates')

# ── Helpers ───────────────────────────────────────────────────────────────────
def inject_eigenbasis(model, eigenvalues, eigenvectors, device):
    for conv in (model.graphconv1, model.graphconv2):
        conv.eigen        = eigenvalues.to(device)
        conv.eigen_vector = eigenvectors.to(device)


def build_norm_edges(edge_index, N, device):
    ei_sym, ew_sym = get_normalization(
        edge_index.cpu(), N, improved=1.0, dtype=torch.float32, norm='sym', laplacian=False
    )
    ei_rw, ew_rw = get_normalization(
        edge_index.cpu(), N, improved=1.0, dtype=torch.float32, norm='rw', laplacian=False
    )
    return (ei_sym.to(device), ew_sym.to(device),
            ei_rw.to(device),  ew_rw.to(device))


def prepare_full_X(active_cols, global_idx, N, Z_DATA, t, device):
    x_active = prepare_node_features(active_cols, Z_DATA, t)
    F_dim = x_active.shape[1]
    X = torch.zeros(N, F_dim, dtype=torch.float32)
    X[global_idx] = x_active
    return X.to(device)


def _safe_norm(v):
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-12)


class EarlyStopping:
    def __init__(self, patience=5, save_path='outputs/grasped_model.pt', init_best_loss=float('inf')):
        self.patience  = patience
        self.save_path = save_path
        self.best_loss = init_best_loss
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss, model, optimizer=None):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            ckpt = {'model': model.state_dict()}
            if optimizer is not None:
                ckpt['optimizer'] = optimizer.state_dict()
            torch.save(ckpt, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def build_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _rep_t = list(GRAPH_CACHE.keys())[0]
    _rep_evals, _rep_evecs = EIGEN_CACHE[_rep_t]
    _mock_dataset = _MockDataset(_rep_evals, _rep_evecs)
    _rep_nn_list  = torch.full((N_FULL,), 11, dtype=torch.long)

    model = GNNStructEncoder(
        args              = ARGS,
        dataset           = _mock_dataset,
        in_dim            = FEATURE_DIM,
        hidden_dim        = ARGS.hidden_dimension,
        layer_num         = ARGS.gnn_layer_num,
        sample_size       = ARGS.sample_size,
        device            = DEVICE,
        neighbor_num_list = _rep_nn_list,
        lambda_loss1      = ARGS.lambda_loss1,
        lambda_loss2      = ARGS.lambda_loss2,
        lambda_loss3      = ARGS.lambda_loss3,
    ).to(DEVICE)

    deg_params  = list(map(id, model.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in deg_params, model.parameters())
    optimizer = torch.optim.Adam(
        [{'params': base_params},
         {'params': model.degree_decoder.parameters(), 'lr': 0.001946}],
        lr=0.001946, weight_decay=3e-4
    )
    return model, optimizer


def train_one_seed(model, optimizer, save_path, n_epochs=50, patience=7):
    early_stop = EarlyStopping(patience=patience, save_path=save_path)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_train = 0.0, 0
        for t in tqdm(train_dates, desc=f'  Epoch {epoch+1}/{n_epochs} train', leave=False):
            r = GRAPH_CACHE.get(t)
            if r is None:
                continue
            evals, evecs = EIGEN_CACHE[t]
            inject_eigenbasis(model, evals, evecs, DEVICE)
            ei_sym, ew_sym, ei_rw, ew_rw = build_norm_edges(r['edge_index'], N_FULL, DEVICE)
            X_t    = prepare_full_X(r['active_cols'], r['global_idx'], N_FULL, Z_DATA, t, DEVICE)
            gt_deg = r['neighbor_num_list'].float().to(DEVICE)
            optimizer.zero_grad()
            try:
                with redirect_stdout(io.StringIO()):
                    loss, *_ = model(ei_sym, ew_sym, ei_rw, ew_rw, X_t, gt_deg, r['neighbor_dict'])
            except (ValueError, RuntimeError):
                continue
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_train    += 1

        avg_train = epoch_loss / max(n_train, 1)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for t in tqdm(val_dates, desc=f'  Epoch {epoch+1}/{n_epochs} val', leave=False):
                r = GRAPH_CACHE.get(t)
                if r is None:
                    continue
                evals, evecs = EIGEN_CACHE[t]
                inject_eigenbasis(model, evals, evecs, DEVICE)
                ei_sym, ew_sym, ei_rw, ew_rw = build_norm_edges(r['edge_index'], N_FULL, DEVICE)
                X_t    = prepare_full_X(r['active_cols'], r['global_idx'], N_FULL, Z_DATA, t, DEVICE)
                gt_deg = r['neighbor_num_list'].float().to(DEVICE)
                try:
                    with redirect_stdout(io.StringIO()):
                        loss_v, *_ = model(ei_sym, ew_sym, ei_rw, ew_rw, X_t, gt_deg, r['neighbor_dict'])
                except (ValueError, RuntimeError):
                    continue
                if not torch.isnan(loss_v):
                    val_loss += loss_v.item()
                    n_val    += 1

        avg_val = val_loss / max(n_val, 1)
        print(f'  Epoch {epoch+1:03d} | train={avg_train:.4f} | val={avg_val:.4f}')

        early_stop(avg_val, model, optimizer)
        if early_stop.stop:
            print('  Early stopping.')
            break

    ckpt = torch.load(save_path)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    return model


def test_one_seed(model):
    model.eval()
    results     = {}
    test_prices = {}

    with torch.no_grad():
        for t in tqdm(test_dates, desc='  Testing'):
            r = GRAPH_CACHE.get(t)
            if r is None:
                continue
            evals, evecs = EIGEN_CACHE[t]
            inject_eigenbasis(model, evals, evecs, DEVICE)
            ei_sym, ew_sym, ei_rw, ew_rw = build_norm_edges(r['edge_index'], N_FULL, DEVICE)
            X_t    = prepare_full_X(r['active_cols'], r['global_idx'], N_FULL, Z_DATA, t, DEVICE)
            gt_deg = r['neighbor_num_list'].float().to(DEVICE)
            try:
                with redirect_stdout(io.StringIO()):
                    (_, _, _, _, _, h_lp, deg_lp, feat_lp) = model(
                        ei_sym, ew_sym, ei_rw, ew_rw, X_t, gt_deg, r['neighbor_dict']
                    )
            except (ValueError, RuntimeError):
                continue

            h_lp    = h_lp.squeeze().cpu()
            deg_lp  = deg_lp.squeeze().cpu()
            feat_lp = feat_lp.squeeze().cpu()

            score_full = (
                ARGS.h_loss_weight       * _safe_norm(h_lp)   +
                ARGS.degree_loss_weight  * _safe_norm(deg_lp)  +
                ARGS.feature_loss_weight * _safe_norm(feat_lp)
            )

            active_scores = score_full[r['global_idx']].numpy()
            results[t] = (r['active_cols'], active_scores)

            t_pd = pd.Timestamp(t)
            if t_pd in prices.index:
                test_prices[t] = prices.loc[t_pd, r['active_cols']]

    return results, test_prices


# ── Main loop over seeds ──────────────────────────────────────────────────────
SEEDS = [0, 2, 4, 6]
os.makedirs('outputs', exist_ok=True)

all_seed_results = {}   # seed -> {date: (active_cols, scores)}
test_prices_ref  = None  # same across seeds (graph is deterministic)

for seed in SEEDS:
    print(f'\n{"="*60}')
    print(f'SEED {seed}')
    print(f'{"="*60}')

    save_path = f'outputs/grasped_model_seed{seed}.pt'
    model, optimizer = build_model(seed)
    model = train_one_seed(model, optimizer, save_path)
    seed_results, seed_prices = test_one_seed(model)
    all_seed_results[seed] = seed_results

    if test_prices_ref is None:
        test_prices_ref = seed_prices

    print(f'Seed {seed}: {len(seed_results)} test days collected.')

# ── Average signals across seeds ──────────────────────────────────────────────
print('\nAveraging signals across seeds...')
all_test_dates = sorted(
    set.intersection(*[set(all_seed_results[s].keys()) for s in SEEDS])
)

avg_results = {}
for date in all_test_dates:
    active_cols = all_seed_results[SEEDS[0]][date][0]  # same across seeds
    scores_stack = np.stack(
        [all_seed_results[s][date][1] for s in SEEDS], axis=0
    )  # (4, n_active)
    avg_scores = scores_stack.mean(axis=0)
    avg_results[date] = (active_cols, avg_scores)

print(f'Averaged results for {len(avg_results)} test dates.')

# ── Save ──────────────────────────────────────────────────────────────────────
RESULTS_PATH = 'outputs/test_results_grasped_seeds_avg.pkl'
PRICES_PATH  = 'outputs/test_prices_grasped_seeds_avg.pkl'

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump(avg_results, f)
with open(PRICES_PATH, 'wb') as f:
    pickle.dump(test_prices_ref, f)

print(f'\nSaved {len(avg_results)} averaged test days.')
print(f'Results -> {RESULTS_PATH}')
print(f'Prices  -> {PRICES_PATH}')
