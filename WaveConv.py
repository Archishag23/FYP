import torch
import torch.nn.functional as F
from torch.nn import Linear
from layer import Wave_prop


class WaveConv(torch.nn.Module):
    def __init__(self, dataset, in_channels, out_channels, args):
        super(WaveConv, self).__init__()
        self.args = args
        self.lin1 = Linear(in_channels, args.hidden_channels_wave)
        self.lin2 = Linear(args.hidden_channels_wave, out_channels)
        self.toy = False
        if args.dataset in ["Toygraph_sin", "Toygraph_esin"]:
            self.toy = True
            eigen, eigen_vector = dataset.eigenvalues, dataset.vectors
            self.eigen, self.eigen_vector = eigen.to("cuda"), eigen_vector.to("cuda")
        else:
            eigen, eigen_vector = dataset.eigenvalues, dataset.vectors
            self.eigen, self.eigen_vector = eigen.to("cuda"), eigen_vector.to("cuda")

        N = len(self.eigen)
        self.prop = Wave_prop(args.K, N)

        self.dprate = args.dprate
        self.dropout = args.dropout
        self.Pro_Matrix = None

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x, _ = self.prop(
                x, self.eigen, self.eigen_vector, kernel=self.args.wave_kernel
            )
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x, _ = self.prop(x, self.eigen, self.eigen_vector)
            return x
