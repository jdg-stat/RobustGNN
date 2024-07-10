from RobustGNN.utils import *
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx ,from_networkx
from torch_geometric.data import Data
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph as LFR

class synthData():
    '''
    To Generate Synthetic LFR Benchmarks.
    params~
    type     = "LFR" (default) #Only LFR implemented, todo->SBM
    feat_len = 32, how long synthetic feature size should be
    noise    = 2, scale parameter for Gaussian noise
    loc      = 0, mean parameter for Gaussian noise
    n        = 1000, Number of nodes
    mu = 0.1, mu parameter for LFR benchmark
    '''
    def __init__(self, type = "LFR", n=1000, mu = 0.1, feat_len = 32, noise = 2, loc = 0):
        self.type = type
        self.feat_len = feat_len
        
        if self.type == "LFR":
            self.graph = self.gen_LFR(n, mu, noise,loc)
        else:
            pass

    #Main Function to generate LFR
    def gen_LFR(self, n, mu,noise, loc):
        LFR_graph = LFR(n= n, tau1= 2, 
                        tau2 = 1.1,mu = mu, 
                        average_degree = 25 , 
                        max_degree = int(0.1*n),
                        min_community = int(0.1*n),
                        max_community = int(0.1*n))
        LFR_graph.remove_edges_from(nx.selfloop_edges(LFR_graph))
        labels_dict = nx.get_node_attributes(LFR_graph, "community")
        y = self.y = self.get_ground_truth(labels_dict) 
        num_communities = len(set(y))
        clean, dirty, labels = line_gaussians(n_points = n, n_clusters=num_communities ,loc_scale=loc,
                                              noise_scale = noise, feat_len = self.feat_len)
        self.clean = self.get_node_feats(y,labels, clean)
        self.dirty = self.get_node_feats(y,labels, dirty)
        LFR_pyg = self.nx_to_pyg(LFR_graph, y)
        return LFR_pyg
        
    #Generates Noisy benchmark 
    def gen_LFR_bad(self):
        temp_graph = self.graph.clone()
        temp_graph.x = self.noisy_feats
        return temp_graph
        
    #Function to visualize Synthetic Data
    def plot_graph(self):
        vis = to_networkx(self.graph)
        vis.remove_edges_from(nx.selfloop_edges(vis))
        
        
        node_labels = self.y.numpy(force=True)
        
        import matplotlib.pyplot as plt
        plt.figure(1,figsize=(15,13)) 
        nx.draw(vis, cmap=plt.get_cmap('Set3'),node_color = node_labels,node_size=20,linewidths=6)
        plt.show()
    
    def get_ground_truth(self, labels):
        collect_values= []
        for keys, values in labels.items():
            if values not in collect_values:
                collect_values.append(values)

        dict_labels = {}
        for i in range(len(collect_values)):
            y_labels = {labels:i for labels in collect_values[i]}
            dict_labels.update(y_labels)
        y_labels = dict(sorted(dict_labels.items()))
        self.true_comms=y_labels
        y = [values for keys,values in y_labels.items()]
        return y

    def get_node_feats(self, response, label_set, feats):
        node_feats = np.zeros((feats.shape[0], feats.shape[1]))
        for k in range(len(label_set)):
            id1 = [i for i, j in enumerate(response) if j == k]
            id2 = [i for i, j in enumerate(label_set) if j == k]
            node_feats[id1, :] = feats[id2,:]
        return node_feats

    #Converts a networkx object to pyg object
    def nx_to_pyg(self, graph, response):
        feats = self.clean
        num_nodes = len(graph.nodes)
        x = torch.tensor(feats, dtype = torch.float )
        edge_index = from_networkx(graph).edge_index
        edge_index = torch.tensor(edge_index, dtype = torch.long)
        y = torch.tensor(response, dtype = torch.long)
        omega = np.zeros(num_nodes, dtype = bool)
        idx = np.random.choice(range(num_nodes),
                                            size = int(0.2*num_nodes),
                                            replace =False)
        omega[idx] =True
        train_mask = omega
        test_mask = ~train_mask
        temp_feats = self.clean
        temp_feats[test_mask] = self.dirty[test_mask]
        x_pert= torch.tensor(temp_feats, dtype = torch.float)
        self.noisy_feats = x_pert
        G = Data(x = x, edge_index = edge_index, y=y,
                 train_mask = torch.tensor(train_mask),
                 test_mask = torch.tensor(test_mask))
        return G