"""
test_wavelet_date.py
--------------------
Tests WaveletConvLayer on a single date using your real pipeline.
Loads functions directly from the .ipynb — no conversion needed.

Usage
-----
  python test_wavelet_date.py                          # defaults to 2020-03-12
  python test_wavelet_date.py --date 2008-09-15        # Lehman
  python test_wavelet_date.py --date 2020-03-16 --K 16
  python test_wavelet_date.py --channel neg --top 20

Requirements
------------
  - SPX_sectors_data.xlsx in the same directory
  - GAE_working_BEST23Apr_spectral.ipynb in the same directory
  - gae_grasped.py in the same directory
"""

import argparse
import sys
import types
import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------
# 1.  Load functions from the notebook WITHOUT converting it to .py
# --------------------------------------------------------------------------
def load_notebook_functions(notebook_path: str) -> types.ModuleType:
    """
    Executes all code cells in the notebook in order and returns a module
    object whose namespace contains every function/class defined there.
    """
    try:
        import nbformat
    except ImportError:
        print("[ERROR] nbformat not installed. Run:  pip install nbformat")
        sys.exit(1)

    nb = nbformat.read(notebook_path, as_version=4)

    # Build a fresh module namespace so nothing pollutes the global scope
    mod = types.ModuleType("notebook_functions")
    mod.__dict__.update({
        "__name__": "notebook_functions",
        "__file__": notebook_path,
    })

    # Execute each code cell in sequence, accumulating definitions
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source.strip()
        if not src:
            continue
        try:
            exec(compile(src, f"<cell {i}>", "exec"), mod.__dict__)
        except Exception as e:
            # Skip cells that call main() or load files at module level
            if any(kw in src for kw in ["main()", "train_model(", "plt.show", "pickle.load"]):
                pass  # expected — silently skip
            else:
                print(f"  [WARN] Cell {i} skipped — {type(e).__name__}: {e}")

    return mod


# --------------------------------------------------------------------------
# 2.  Main test routine
# --------------------------------------------------------------------------
def run_one_date(nb, returns, date, K_wavelet, lookback, channel, out_features, device, top_n):

    t = pd.to_datetime(date)
    print(f"\n{'='*60}")
    print(f"  Date     : {t.date()}")
    print(f"  K bins   : {K_wavelet}   lookback: {lookback}d   channel: {channel}")
    print(f"{'='*60}")

    # -- Active stocks -------------------------------------------------------
    active = nb.get_active_stocks(returns, t, lookback_days=int(lookback * 1.5), min_obs=lookback)
    if not active:
        print("[WARN] No active stocks on this date. Try a different date.")
        return
    N = len(active)
    print(f"  Active stocks : {N}")

    # -- Correlation + dual adjacency ----------------------------------------
    corr, active = nb.correlation_matrix(returns, t, K=lookback, active=active)
    N = len(active)
    A_pos, A_neg = nb.create_adjacency_from_correlation(corr)   # (N,N) float32 tensors

    # -- Node features: raw returns window (N, lookback) ---------------------
    window = (
        returns[active]
        .loc[t - pd.Timedelta(days=int(lookback * 1.5)): t]
        .iloc[-lookback:]
    )
    X = torch.tensor(window.values.T, dtype=torch.float32)
    X = torch.nan_to_num(X, nan=0.0)
    in_features = X.shape[1]
    print(f"  Feature shape : {X.shape}  (N={N}, d={in_features})")

    # -- Spectral basis via your SpectralBasisGenerator ----------------------
    def make_basis(A_tensor):
        gen = nb.SpectralBasisGenerator(A_tensor.numpy())
        return gen.get_eigen_pair()   # (eigvals (N,), eigvecs (N,N)) CPU tensors

    print("  Computing eigendecompositions ...")
    eigvals_pos, eigvecs_pos = make_basis(A_pos)
    eigvals_neg, eigvecs_neg = make_basis(A_neg)

    # -- Instantiate WaveletConvLayer ----------------------------------------
    from gae_grasped import WaveletConvLayer

    layer_pos = WaveletConvLayer(in_features, out_features, K=K_wavelet).to(device)
    layer_neg = WaveletConvLayer(in_features, out_features, K=K_wavelet).to(device)

    X_dev = X.to(device)

    # -- Forward pass --------------------------------------------------------
    layer_pos.eval()
    layer_neg.eval()
    with torch.no_grad():
        h_pos = layer_pos(X_dev, eigvals_pos, eigvecs_pos)
        h_neg = layer_neg(X_dev, eigvals_neg, eigvecs_neg)
        if channel == "pos":
            h = h_pos
        elif channel == "neg":
            h = h_neg
        else:
            h = 0.5 * h_pos + 0.5 * h_neg    # equal mix; gamma is learned in full encoder

    print(f"  Embedding shape : {tuple(h.shape)}")

    # -- Reconstruction error (random projection back to input dim) ----------
    torch.manual_seed(42)
    proj    = torch.randn(out_features, in_features, device=device) / (out_features ** 0.5)
    x_hat   = h @ proj                           # (N, in_features)
    errors  = torch.norm(X_dev - x_hat, dim=1)  # (N,)
    errors_np = errors.cpu().numpy()

    # -- Ranked results table ------------------------------------------------
    ranked = np.argsort(errors_np)[::-1]
    print(f"\n  Top {top_n} stocks by reconstruction error  (layer randomly initialised):")
    print(f"  {'Rank':<5} {'Ticker':<12} {'Error':>10}  {'Top-%':>8}")
    print(f"  {'-'*40}")
    for rank, idx in enumerate(ranked[:top_n], 1):
        ticker = active[idx]
        err    = errors_np[idx]
        pct    = 100.0 * (1 - np.searchsorted(np.sort(errors_np), err) / N)
        print(f"  {rank:<5} {ticker:<12} {err:>10.4f}  {pct:>7.1f}%")

    # -- Theta at init -------------------------------------------------------
    print(f"\n  theta (A+ channel at init — expect all 1.0):")
    theta = layer_pos.theta.detach().cpu().numpy()
    for k, v in enumerate(theta):
        bar = "█" * max(1, int(abs(v) * 12))
        print(f"    bin {k:>3d} | {v:+.4f}  {bar}")

    # -- Eigenvalue spectrum -------------------------------------------------
    ev = eigvals_pos.cpu().numpy()
    print(f"\n  Eigenvalue spectrum (A+ Laplacian):")
    print(f"    min={ev.min():.4f}  median={np.median(ev):.4f}  max={ev.max():.4f}")
    n_out = ((ev < -1e-6) | (ev > 2 + 1e-6)).sum()
    print(f"    Outside [0,2] before clamp: {n_out}  (clamped inside forward — safe)")

    print(f"\n  [OK] Forward pass complete on {t.date()},  N={N} stocks.\n")

    return {
        "date":          t,
        "active_stocks": active,
        "errors":        dict(zip(active, errors_np.tolist())),
        "h":             h.cpu(),
        "theta_pos":     theta,
        "eigvals_pos":   ev,
    }


# --------------------------------------------------------------------------
# 3.  Entry point
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date",     default="2020-03-12")
    p.add_argument("--K",        type=int, default=8,
                   help="Haar spectral bins (power of 2)")
    p.add_argument("--lookback", type=int, default=21,
                   help="Correlation window in trading days")
    p.add_argument("--channel",  default="both", choices=["pos","neg","both"])
    p.add_argument("--top",      type=int, default=10)
    p.add_argument("--out_dim",  type=int, default=32)
    p.add_argument("--gpu",      action="store_true")
    p.add_argument("--data",     default="SPX_sectors_data.xlsx")
    p.add_argument("--notebook", default="GAE_working_BEST23Apr_spectral.ipynb")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"\n[test_wavelet_date.py]  device={device}")

    print(f"\n[1/3] Loading notebook functions from {args.notebook} ...")
    nb = load_notebook_functions(args.notebook)
    required = ["get_active_stocks", "correlation_matrix",
                "create_adjacency_from_correlation", "SpectralBasisGenerator"]
    missing = [fn for fn in required if not hasattr(nb, fn)]
    if missing:
        print(f"[ERROR] Could not find these in the notebook: {missing}")
        sys.exit(1)
    print(f"  Loaded: {required}")

    print(f"\n[2/3] Loading returns from {args.data} ...")
    returns = pd.read_excel(args.data, header=[0,1], index_col=0)
    returns.columns = returns.columns.get_level_values(0)
    returns.dropna(how="all", inplace=True)
    returns = returns.pct_change().dropna(how="all").ffill().bfill()
    print(f"  Shape: {returns.shape}  "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")

    print(f"\n[3/3] Running forward pass ...")
    run_one_date(
        nb           = nb,
        returns      = returns,
        date         = args.date,
        K_wavelet    = args.K,
        lookback     = args.lookback,
        channel      = args.channel,
        out_features = args.out_dim,
        device       = device,
        top_n        = args.top,
    )

    print("Other crash dates to try:")
    for d, label in [
        ("2020-03-12", "COVID crash"),
        ("2020-03-16", "COVID circuit breaker"),
        ("2008-09-15", "Lehman collapse"),
        ("2020-02-24", "COVID first drop"),
    ]:
        print(f"  python test_wavelet_date.py --date {d}   # {label}")

