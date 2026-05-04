import pygsp
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import normalize


# For Graph Wavelet
class WaveletSparsifier(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """

    def __init__(self, graph, scale, approximation_order, tolerance):
        """
        :param graph: NetworkX graph object.
        :param scale: Kernel scale length parameter.
        :param approximation_order: Chebyshev polynomial order.
        :param tolerance: Tolerance for sparsification.
        """
        self.graph = graph
        self.pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(self.graph))
        self.pygsp_graph.estimate_lmax()
        self.scales = [-scale, scale]
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []

    def calculate_wavelet(self):
        """
        Creating sparse wavelets.
        :return remaining_waves: Sparse matrix of attenuated wavelets.
        """
        impulse = np.eye(self.graph.number_of_nodes(), dtype=int)
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(
            self.pygsp_graph, self.chebyshev, impulse
        )
        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        n_count = self.graph.number_of_nodes()
        remaining_waves = sparse.csr_matrix(
            (wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),
            shape=(n_count, n_count),
            dtype=np.float32,
        )
        return remaining_waves

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm="l1", axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.phi_matrices[0].nonzero()[0]) / (
            self.graph.number_of_nodes() ** 2
        )
        wavelet_density = str(round(100 * wavelet_density, 2))
        inverse_wavelet_density = len(self.phi_matrices[1].nonzero()[0]) / (
            self.graph.number_of_nodes() ** 2
        )
        inverse_wavelet_density = str(round(100 * inverse_wavelet_density, 2))
        print("Density of wavelets: " + wavelet_density + "%.")
        print("Density of inverse wavelets: " + inverse_wavelet_density + "%.\n")

    def calculate_all_wavelets(self):
        """
        Graph wavelet coefficient calculation.
        """
        print("\nWavelet calculation and sparsification started.\n")
        for i, scale in enumerate(self.scales):
            self.heat_filter = pygsp.filters.Heat(self.pygsp_graph, tau=[scale])
            self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(
                self.heat_filter, m=self.approximation_order
            )
            sparsified_wavelets = self.calculate_wavelet()
            self.phi_matrices.append(sparsified_wavelets)
        self.normalize_matrices()
        self.calculate_density()
