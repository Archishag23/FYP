"""
gae_grasped.py — GRASPED spectral encoder building block.

Implements the wavelet-based graph encoder from:
  Choong et al. (2025) "GRASPED: Graph Anomaly Detection using Autoencoder
  with Spectral Encoder and Decoder", arXiv:2508.15633.

References to paper equations used in WaveletConvLayer:
  Eq 5:  g_c(λ) = Σ_{k=0}^{K-1} θ_{J,k} φ_{Haar,J,k}(λ)
           The spectral filter is piecewise-constant on [0, 2]: the domain is
           partitioned into K equal-width bins and each bin has one learnable
           scalar weight θ_{J,k}.  Because each Haar basis function is 1 on
           exactly one bin and 0 elsewhere, g_c(λ_i) = θ_{J, bin(λ_i)}.
           We compute this with a vectorised bin-lookup:
             boundaries = linspace(0, 2, K+1)[1:-1]   # K-1 interior edges
             bin_idx    = bucketize(λ, boundaries)     # (N,) in {0,...,K-1}
             Phi        = one_hot(bin_idx, K)          # (N, K) indicator matrix
             g          = Phi @ theta                  # (N,) filter response
           No Python loop over nodes is required.

  Eq 7:  M = U G_c U^T,   G_c = diag(g_c(λ_1), ..., g_c(λ_N))
           M is never materialised as an (N, N) dense matrix.  Instead:
             M x = U (g ⊙ (U^T x))
           costs two (N,N)·(N,d) matmuls and one elementwise multiply,
           keeping the autograd graph O(N·d) rather than O(N²).

  Eq 8:  H^(i) = σ( M H^(i-1) W^(i-1) )
           After the spectral mixing step above, a single nn.Linear W maps
           in_features → out_features, followed by ReLU and dropout.
           W is per-layer (not shared across layers) and not replicated per
           filter bank — there is only one filter bank g_c in the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletConvLayer(nn.Module):
    """
    Single GRASPED Haar-wavelet convolution layer (Eqs 5, 7, 8).

    Parameters
    ----------
    in_features  : int    input feature dimension per node
    out_features : int    output feature dimension per node
    K            : int    number of Haar spectral bins (must be a power of 2)
    dropout      : float  dropout probability applied after activation
    """

    def __init__(self, in_features: int, out_features: int, K: int, dropout: float = 0.0):
        super().__init__()
        assert K > 0 and (K & (K - 1)) == 0, f"K must be a power of 2, got {K}"

        self.K = K

        # Eq 5: one learnable weight per Haar bin.
        # Initialised to 1 → g_c(λ) = 1 ∀λ → G_c = I → M = U U^T = I.
        # The filter therefore starts as a no-op (identity diffusion).
        self.theta = nn.Parameter(torch.ones(K))

        # Eq 8: linear projection applied after spectral mixing.
        self.W = nn.Linear(in_features, out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (N, in_features)   node feature matrix
        eigvals : (N,)               ascending eigenvalues of the normalised
                                     Laplacian, as returned by get_eigen_pair()
        eigvecs : (N, N)             corresponding eigenvector matrix (columns),
                                     as returned by get_eigen_pair()

        Returns
        -------
        (N, out_features)
        """
        device = x.device

        # --- Eq 5: build g_c(λ_i) for every eigenvalue ---

        # Clamp first: numerical noise from eigh can push values just outside [0,2].
        lam = eigvals.clamp(0.0, 2.0).to(device)

        # K-1 interior bin boundaries partition [0, 2] into K equal-width bins.
        # bucketize maps each λ_i to an index in {0, ..., K-1} — no node loop.
        boundaries = torch.linspace(0.0, 2.0, self.K + 1, device=device)[1:-1]
        bin_idx = torch.bucketize(lam, boundaries)            # (N,)

        # Phi[i, k] = 1 iff λ_i falls in bin k  (exactly one 1 per row).
        # g[i] = theta[bin_idx[i]]  — the filter's response at eigenvalue λ_i.
        Phi = F.one_hot(bin_idx, num_classes=self.K).float()  # (N, K)
        g = Phi @ self.theta                                   # (N,)

        # --- Eqs 7 & 8: spectral mixing without materialising M ---
        # M x = U diag(g) U^T x = eigvecs @ (g ⊙ (eigvecs.T @ x))
        eigvecs = eigvecs.to(device)
        x_spec = eigvecs.T @ x                                 # (N, in_features)
        x_spec = g.unsqueeze(1) * x_spec                      # (N, in_features)
        h = eigvecs @ x_spec                                   # (N, in_features)

        # Eq 8: linear projection → activation → dropout
        out = self.W(h)                                        # (N, out_features)
        out = F.relu(out)
        out = self.dropout(out)
        return out


def compare_spectral_ranges(
    A_pos,
    A_neg,
    label: str = "",
    thresholds: tuple = (1.5, 1.75),
) -> dict:
    """
    Print a spectral statistics table comparing the A+ and A- Laplacians
    for a single date, and return the raw numbers.

    The fractions above 1.5 and 1.75 directly report how many eigenvalues
    land in the two highest Haar bins ([1.5, 1.75) and [1.75, 2.0]).  A
    fraction of 0.0 means those bins receive no gradient — they are dead.

    Parameters
    ----------
    A_pos      : (N, N) array-like   positive-correlation adjacency
    A_neg      : (N, N) array-like   negative-correlation adjacency
    label      : str                 printed in the header (use the date)
    thresholds : tuple of float      eigenvalue thresholds to report fractions above

    Returns
    -------
    dict with keys "pos" and "neg", each a dict of the computed statistics.
    Collect the return values across dates to build a multi-date summary.

    Example
    -------
    results = {}
    for date, A_pos, A_neg in date_graph_triples:
        results[date] = compare_spectral_ranges(A_pos, A_neg, label=date)
    """
    import numpy as np

    def _eigvals(adj):
        adj = np.array(adj, dtype=np.float64)
        deg = adj.sum(axis=1)
        d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
        A_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
        L = np.eye(adj.shape[0]) - A_norm
        return np.linalg.eigh(L)[0]   # ascending, same path as SpectralBasisGenerator

    ev_pos = _eigvals(A_pos)
    ev_neg = _eigvals(A_neg)
    N = len(ev_pos)

    def _stats(ev):
        s = {
            "lambda_max":    float(ev[-1]),
            "lambda_median": float(np.median(ev)),
        }
        for t in thresholds:
            key = f"frac_above_{t:.2f}"
            s[key] = float((ev > t).sum() / N)
        return s

    stats_pos = _stats(ev_pos)
    stats_neg = _stats(ev_neg)

    # --- print table ---
    header = "Spectral range comparison"
    if label:
        header += f"  [{label}]"
    print(f"\n  {header}")
    print(f"  {'Statistic':<24} {'A+':>10} {'A-':>10}  {'Δ (A- − A+)':>14}")
    print("  " + "-" * 62)

    row_keys = [
        ("lambda_max",                      "lambda_max"),
        ("lambda_median",                   "lambda_median"),
    ] + [
        (f"frac_above_{t:.2f}", f"frac > {t:.2f}") for t in thresholds
    ]

    for key, display in row_keys:
        vp = stats_pos[key]
        vn = stats_neg[key]
        print(f"  {display:<24} {vp:>10.4f} {vn:>10.4f}  {vn - vp:>+14.4f}")

    print()
    return {"pos": stats_pos, "neg": stats_neg}


def plot_theta(
    layer: "WaveletConvLayer",
    eigvals: torch.Tensor,
    epoch: int = None,
    ax=None,
):
    """
    Diagnostic: bar chart of learned Haar filter weights with eigenvalue
    density overlaid.

    After training, theta should be biased toward high-frequency bins (right
    side of the plot) if the encoder is exploiting the right-shift phenomenon
    (Tang et al. 2022).  A flat or left-biased theta means the encoder has
    collapsed to a low-pass filter — no better than a GCN baseline.

    The eigenvalue density overlay shows which bins actually receive gradient
    signal.  Empty bins (no eigenvalues) have dead theta parameters and will
    never move regardless of training.

    Parameters
    ----------
    layer   : WaveletConvLayer  — the trained (or partially trained) layer
    eigvals : (N,) tensor       — today's eigenvalues from get_eigen_pair()
    epoch   : int or None       — appended to the title when provided
    ax      : matplotlib Axes or None  — drawn into existing axes if given,
                                         otherwise a new figure is created

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    K = layer.K
    theta = layer.theta.detach().cpu().numpy()          # (K,)
    lam   = eigvals.detach().cpu().numpy().clip(0., 2.) # (N,) clamped

    bin_edges   = np.linspace(0.0, 2.0, K + 1)         # K+1 edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width   = 2.0 / K

    # Count how many eigenvalues fall in each bin so we can flag empty ones.
    bin_counts, _ = np.histogram(lam, bins=bin_edges)
    empty_bins     = bin_counts == 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    # --- primary axis: theta bar chart ---
    bar_colours = ["steelblue" if not e else "#cccccc" for e in empty_bins]
    ax.bar(
        bin_centers, theta,
        width=bin_width * 0.75,
        color=bar_colours,
        alpha=0.85,
        label="θ (filter weight)",
        zorder=3,
    )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.9,
               label="identity (θ = 1)", zorder=2)
    ax.set_xlabel("Eigenvalue λ  [0, 2]")
    ax.set_ylabel("Filter weight θ", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f"{e:.2f}" for e in bin_edges], rotation=45, ha="right")

    # --- secondary axis: eigenvalue density ---
    ax2 = ax.twinx()
    ax2.hist(lam, bins=bin_edges, density=True, alpha=0.25,
             color="darkorange", label="Eigenvalue density", zorder=1)
    ax2.set_ylabel("Eigenvalue density", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(bottom=0)

    # --- title and legend ---
    title = "Haar filter weights θ"
    if epoch is not None:
        title += f"  (epoch {epoch})"
    if empty_bins.any():
        empty_labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}"
                        for i in np.where(empty_bins)[0]]
        title += f"\n⚠ grey bins empty (no eigenvalues): {', '.join(empty_labels)}"
    ax.set_title(title, fontsize=10)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig
