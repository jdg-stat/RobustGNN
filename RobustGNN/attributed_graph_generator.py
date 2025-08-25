from RobustGNN.utils import *
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph as LFR
import matplotlib.pyplot as plt
import random

class synthetic_graph_generator:
    '''
    Generate Synthetic Graph Benchmarks including LFR and SBM.
    
    Parameters:
    -----------
    type : str, default="LFR"
        Type of synthetic graph to generate ("LFR" or "SBM")
    n : int, default=1000
        Number of nodes in the graph
    feat_len : int, default=32
        Length of synthetic feature vectors
    mu : float, default=0.1
        Mixing parameter for LFR benchmark
    seed : int, default=None
        Random seed for reproducibility
    '''
    def __init__(self, type="LFR", n=1000, mu=0.1, feat_len=32, seed=None):
        self.type = type
        self.feat_len = feat_len
        self.n = n
        
        # Set random seed if provided
        if seed is not None:
            self.set_seed(seed)
        
        if self.type == "LFR":
            self.graph = self.gen_LFR(n, mu)
        elif self.type == "SBM":
            self.graph = self.gen_SBM(n)
        else:
            raise ValueError("Type must be either 'LFR' or 'SBM'")

    @staticmethod
    def set_seed(seed):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def gen_LFR(self, n, mu):
        """Generate LFR benchmark graph"""
        LFR_graph = LFR(n=n, tau1=2, 
                        tau2=1.1, mu=mu, 
                        average_degree=25, 
                        max_degree=int(0.1*n),
                        min_community=int(0.1*n),
                        max_community=int(0.1*n))
        LFR_graph.remove_edges_from(nx.selfloop_edges(LFR_graph))
        labels_dict = nx.get_node_attributes(LFR_graph, "community")
        self.y = self.get_ground_truth(labels_dict)
        
        # Generate features based on community structure
        features = self.generate_features(self.y)
        LFR_pyg = self.nx_to_pyg(LFR_graph, features, self.y)
        return LFR_pyg

    def gen_SBM(self, n):
        """Generate Stochastic Block Model graph"""
        # sbm params
        num_communities = int(np.sqrt(n/10))  
        sizes = [n // num_communities] * num_communities 
        
        # Probability matrix for edges
        p_in = 0.3  
        p_out = 0.05  
        p_matrix = np.full((num_communities, num_communities), p_out)
        np.fill_diagonal(p_matrix, p_in)
        
      
        SBM_graph = nx.stochastic_block_model(sizes, p_matrix)
        
    
        y = []
        for i in range(num_communities):
            y.extend([i] * sizes[i])
        self.y = torch.tensor(y, dtype=torch.long)
        
        features = self.generate_features(self.y)
        SBM_pyg = self.nx_to_pyg(SBM_graph, features, self.y)
        return SBM_pyg

    def generate_features(self, labels):
        """Generate node features based on community structure"""
        num_communities = len(torch.unique(labels))
    
        community_features = np.random.randn(num_communities, self.feat_len)
        

        features = np.zeros((len(labels), self.feat_len))
        for i in range(num_communities):
            mask = (labels == i)
            features[mask] = community_features[i] + 0.1 * np.random.randn(mask.sum(), self.feat_len)
        
        return features

    def plot_graph(self, figsize=(15, 13), node_size=20, edge_width=0.1, 
                  with_labels=False, save_path=None):
        """
        Plot the generated graph with community colors.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 13)
            Figure size
        node_size : int, default=20
            Size of nodes in the plot
        edge_width : float, default=0.1
            Width of edges in the plot
        with_labels : bool, default=False
            Whether to show node labels
        save_path : str, default=None
            Path to save the plot. If None, plot is displayed
        """
        vis = to_networkx(self.graph)
        vis.remove_edges_from(nx.selfloop_edges(vis))
        
        node_labels = self.y.numpy()
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(vis)
        nx.draw(vis, pos,
                cmap=plt.get_cmap('Set3'),
                node_color=node_labels,
                node_size=node_size,
                width=edge_width,
                with_labels=with_labels)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        # plt.close()

    def get_ground_truth(self, labels):
        """Extract ground truth labels from community dictionary"""
        collect_values = []
        for keys, values in labels.items():
            if values not in collect_values:
                collect_values.append(values)

        dict_labels = {}
        for i in range(len(collect_values)):
            y_labels = {labels: i for labels in collect_values[i]}
            dict_labels.update(y_labels)
        y_labels = dict(sorted(dict_labels.items()))
        self.true_comms = y_labels
        y = [values for keys, values in y_labels.items()]
        return torch.tensor(y, dtype=torch.long)

    def nx_to_pyg(self, graph, features, labels):
        """Convert networkx graph to PyG Data object"""
        num_nodes = len(graph.nodes)
        x = torch.tensor(features, dtype=torch.float)
        edge_index = from_networkx(graph).edge_index
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        
        G = Data(x=x, 
                edge_index=edge_index,
                y=y)
        return G
