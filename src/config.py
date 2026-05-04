import argparse


# Configurations for experiments
seed_list = [200, 300, 400, 470, 600, 950, 970, 2856, 4335, 8000, 8500, 8611]
K_values = [2, 4, 8, 16, 32, 64, 128, 256]
beta_values = [0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0]
sample_size_values = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25]

# Optimal hyperparamters for each dataset
disney_key_param_list = [
    {
        "K": 8,
        "beta": 1.2,
        "sample_size": 20,
        "lambda_loss1": 0.6,
        "lambda_loss2": 3,
        "lambda_loss3": 0,
    }
]

books_key_param_list = [
    {
        "K": 128,
        "beta": 1.5,
        "sample_size": 20,
        "lambda_loss1": 3,
        "lambda_loss2": 6,
        "lambda_loss3": 0.05,
    }
]

reddit_key_param_list = [
    {
        "K": 16,
        "beta": 0.5,
        "sample_size": 35,
        "lambda_loss1": 0.2,
        "lambda_loss2": 6,
        "lambda_loss3": 0,
    }
]

enron_key_param_list = [
    {
        "K": 16,
        "beta": 0.5,
        "sample_size": 20,
        "lambda_loss1": 0.4,
        "lambda_loss2": 3,
        "lambda_loss3": 0,
    }
]

weibo_key_param_list = [
    {
        "K": 8,
        "beta": 0.5,
        "sample_size": 45,
        "lambda_loss1": 0,
        "lambda_loss2": 4,
        "lambda_loss3": 0,
    }
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument("-f")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["disney", "books", "enron", "reddit", "weibo", "cora"],
        default="cora",
    )
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epoch_num", type=int, default=200)  # 200 / 250 / 150
    parser.add_argument(
        "--lambda_loss1", type=float, default=0.3
    )  # neighbor reconstruction loss weight
    parser.add_argument(
        "--lambda_loss2", type=float, default=2.0
    )  # feature loss weight
    parser.add_argument("--lambda_loss3", type=float, default=0)  # degree loss weight
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--hidden_dimension", type=int, default=32)
    parser.add_argument(
        "--encoder", type=str, choices=["GCN", "GAT", "WaveNet"], default="WaveNet"
    )
    parser.add_argument(
        "--struc_decoder", type=str, choices=["MLP", "DOT"], default="MLP"
    )
    parser.add_argument(
        "--feat_decoder", type=str, choices=["MLP", "Deconv"], default="Deconv"
    )
    parser.add_argument("--loss_step", type=int, default=50)
    parser.add_argument(
        "--real_loss", type=bool, default=True
    )  # use real loss or adaptive loss
    parser.add_argument("--neigh_loss", type=str, default="KL")
    parser.add_argument(
        "--h_loss_weight", type=float, default=0.3
    )  # adaptive loss weight for h_loss contributing to AUC score
    parser.add_argument(
        "--feature_loss_weight", type=float, default=2.0
    )  # adaptive loss weight for feature loss contributing to AUC score
    parser.add_argument(
        "--degree_loss_weight", type=float, default=0
    )  # adaptive loss weight for degree loss contributing to AUC score
    # Hyperparameter for Injecting Synthetic Outliers
    parser.add_argument(
        "--inject_typ", type=str, choices=["str", "ctx", "jnt"], default="str"
    )
    parser.add_argument("--inject_percentage", type=float, default=5)
    parser.add_argument("--use_combine_outlier", type=bool, default=False)
    parser.add_argument("--plot_loss", type=bool, default=True)
    parser.add_argument("--normalize_feat", type=bool, default=True)
    parser.add_argument("--aggregator", type=str, default="mean")
    # For contextual outliers
    parser.add_argument("--calculate_contextual", type=bool, default=False)
    parser.add_argument("--contextual_n", type=int, default=140)
    parser.add_argument("--contextual_k", type=int, default=10)
    # For structural outliers
    parser.add_argument("--calculate_structural", type=bool, default=False)
    parser.add_argument("--structural_n", type=int, default=28)
    parser.add_argument("--structural_m", type=int, default=5)
    # For joint outliers
    parser.add_argument("--calculate_joint", type=bool, default=False)
    parser.add_argument("--joint_structural_n", type=int, default=140)
    parser.add_argument("--joint_structural_m", type=int, default=10)
    # Encoder
    ## For WaveNet
    parser.add_argument(
        "--wave_kernel", type=str, choices=["haar", "db2", "db4"], default="haar"
    )
    parser.add_argument("--hidden_channels_wave", type=int, default=128)
    parser.add_argument("--K", type=int, default=8, help="num of bases.")
    parser.add_argument(
        "--dprate", type=float, default=0.5, help="dropout for propagation layer."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout for neural networks."
    )
    parser.add_argument("--prefix", type=str, default="0", help="sample_prefix")
    parser.add_argument(
        "--suffix", type=str, default=None, help="PCA or not"
    )  # 'pca_200', 'pca_100', 'pca_50'
    ## For GWNN
    parser.add_argument(
        "--scale",
        type=float,
        default=6.0,
        help="Heat kernel scale length. Default is 1.0.",
    )
    parser.add_argument(
        "--approximation-order",
        type=int,
        default=3,
        help="Order of Chebyshev polynomial. Default is 3.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10**-4,
        help="Sparsification parameter. Default is 10^-4.",
    )
    # For Wiener Deconv
    parser.add_argument(
        "--dec_kernel",
        type=str,
        choices=["gcn", "heat", "ppr"],
        default="heat",
        help="Encoding kernel type",
    )
    parser.add_argument(
        "--gnn_layer_num",
        type=int,
        default=2,
        help="Number of hidden layer in GNN based model",
    )
    parser.add_argument(
        "--act", type=str, default="prelu", help="Activation function type"
    )
    parser.add_argument("--norm", type=str, default="", help="Normlaization layer type")
    # For decoder
    parser.add_argument(
        "--dec_aggr",
        type=str,
        default="sum",
        help="Aggregation for multi-channel decoder",
    )
    parser.add_argument(
        "--single",
        action="store_false",
        default=True,
        help="Indicator of single channel",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        default=False,
        help="Indicator of two layer deconvolution in decoder",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        default=False,
        help="Indicator of skip connection",
    )
    parser.add_argument(
        "--simple_dec",
        action="store_false",
        default=True,
        help="Indicator of 1 layer decoder",
    )

    parser.add_argument(
        "--norm_type",
        type=str,
        default="sym",
        help="Type of normalization of adjacency matrix",
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Hyperparameter for latent augmentation"
    )
    parser.add_argument("--gamma", type=float, default=1, help="Hyperparameter for AER")
    parser.add_argument(
        "--gamma_list",
        type=float,
        default=[0.1, 1, 10],
        nargs="+",
        help="Hyperparameter for AER list",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0, help="Dropout rate for node in training"
    )

    return parser.parse_args()
