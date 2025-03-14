import numpy as np

from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim

import torch
from torch_geometric.utils import degree


def line_gaussians(n_points,  # pylint: disable=missing-function-docstring
                   n_clusters = 2,
                   cluster_distance = 2,
                   loc_scale = 0,
                   noise_scale = 2,
                   feat_len= 32):
  n_points = n_points // n_clusters * n_clusters
  points_per_cluster = n_points // n_clusters

  data_clean = np.vstack([
      np.random.normal(loc=cluster_distance * i, size=(points_per_cluster, feat_len))
      for i in range(n_clusters)
  ])

  data_clean -= data_clean.mean(axis=0)  # Make the data zero-mean.

#   data_dirty = data_clean + np.random.normal(loc = loc_scale,
#       scale=noise_scale, size=data_clean.shape)  # Add random noise to the data.

  labels = np.zeros(n_points, dtype=int)
  for i in range(n_clusters):
    labels[points_per_cluster * i:points_per_cluster * (i + 1)] = i

  return data_clean, labels


def compute_ecs(mask, ground_truth, pred):
    '''
    Computes the ECS metric
    params~
    mask = boolean mask for which you want to filter
    ground_truth = Ground truth communities
    pred = predicted communities
    '''
    mask = mask.numpy(force=True)
    mask = np.where(mask)
    gt = {key:ground_truth[key] for key in mask[0]}
    ground_truth=gt
    
    # pred = pred.numpy(force=True)
    pred_community = {i:[pred[i].numpy(force=True).item()] for i in mask[0]}
    ground_truth_community = {key:[value] for key, value in ground_truth.items()}
    clustering_true = Clustering(elm2clu_dict = ground_truth_community)
    # print(clustering_true)
    clustering_pred= Clustering(elm2clu_dict = pred_community)
    # print(clustering_pred)
    ecs = sim.element_sim(clustering_true, clustering_pred)
    return ecs


def del_edges_randomly(pyg_data, p = 0.1):
    
    '''
    takes a pyg data, select p% random nodes and deletes associated edges
    then we modify the edge set and return the data in pyg format

    '''

    data = pyg_data.clone()
    edge_set= data.edge_index.numpy(force=True)
    node_set_del= np.random.randint(len(data.x), size = int(p*len(data.x)))
    for i in node_set_del:
        idx = np.where((edge_set[0] == i) | (edge_set[1]== i))
        edge_set = np.delete(edge_set, idx, 1)
    print(edge_set.shape)
    data.edge_index= torch.tensor(edge_set)
    return data

def del_edges_targetted(pyg_data, p =0.1):

    '''
    takes a pyg data, select p% targetted by degree nodes and deletes associated edges
    then we modify the edge set and return the data in pyg format

    '''
    data = pyg_data.clone()
    del_size= int(p*len(data.x))
    edge_set= data.edge_index.numpy(force=True)
    deg_set= degree(data.edge_index[0]).numpy(force=True)
    node_set_del= np.argpartition(deg_set,-del_size )[-del_size:]
    # print(node_set_del)
    for i in node_set_del:
        idx = np.where((edge_set[0] == i) | (edge_set[1]== i))
        edge_set = np.delete(edge_set, idx, 1)
    data.edge_index= torch.tensor(edge_set)
    return data


import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from typing import Optional

def attribute_mean(graph: Data, mean: float = 1.0) -> Data:
    """
    Perturb node features by adding Gaussian noise with specified mean.
    
    Args:
        graph (Data): PyG graph
        mean (float): Mean value for the Gaussian noise
    Returns:
        Data: Graph with perturbed features
    """
    perturbed_graph = graph.clone()
    
    # Add noise with specified mean to all features
    noise = torch.normal(mean=mean * torch.ones_like(graph.x),
                        std=torch.ones_like(graph.x))
    perturbed_graph.x = graph.x + noise
        
    return perturbed_graph

def attribute_scale(graph: Data, scale: float = 0.1) -> Data:
    """
    Perturb node features by adding Gaussian noise with specified scale.
    
    Args:
        graph (Data): PyG graph
        scale (float): Standard deviation for the Gaussian noise
    Returns:
        Data: Graph with perturbed features
    """
    perturbed_graph = graph.clone()
    
    # Add noise with specified scale to all features
    noise = torch.normal(mean=torch.zeros_like(graph.x),
                        std=scale * torch.ones_like(graph.x))
    perturbed_graph.x = graph.x + noise
        
    return perturbed_graph

def remove_edges_random(graph: Data, remove_ratio: float = 0.1) -> Data:
    """
    Randomly remove edges from the graph.
    
    Args:
        graph (Data): PyG graph
        remove_ratio (float): Ratio of edges to remove
    Returns:
        Data: Graph with removed edges
    """
    perturbed_graph = graph.clone()
    
    # Get number of unique edges (divide by 2 since edges are bidirectional)
    num_unique_edges = graph.edge_index.shape[1] // 2
    num_remove = int(remove_ratio * num_unique_edges)
    
    # Convert to networkx for easier manipulation
    G = to_networkx(graph, to_undirected=True)
    
    # Get list of all edges and randomly sample edges to remove
    edges = list(G.edges())
    edges_to_remove = np.random.choice(len(edges), 
                                     size=num_remove, 
                                     replace=False)
    
    # Remove selected edges
    for idx in edges_to_remove:
        G.remove_edge(*edges[idx])
    
    # Convert back to PyG
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Make it bidirectional again
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    perturbed_graph.edge_index = edge_index
    
    return perturbed_graph

def remove_edges_betweenness(graph: Data, remove_ratio: float = 0.1) -> Data:
    """
    Remove edges based on betweenness centrality.
    
    Args:
        graph (Data): PyG graph
        remove_ratio (float): Ratio of edges to remove
    Returns:
        Data: Graph with removed edges
    """
    perturbed_graph = graph.clone()
    
    # Get number of unique edges (divide by 2 since edges are bidirectional)
    num_unique_edges = graph.edge_index.shape[1] // 2
    num_remove = int(remove_ratio * num_unique_edges)
    
    # Convert to networkx for centrality calculation
    G = to_networkx(graph, to_undirected=True)
    
    # Calculate edge betweenness centrality
    edge_centrality = nx.edge_betweenness_centrality(G)
    
    # Sort edges by centrality and select top edges to remove
    sorted_edges = sorted(edge_centrality.items(), 
                         key=lambda x: x[1],
                         reverse=True)
    
    # Remove selected edges
    for edge, _ in sorted_edges[:num_remove]:
        G.remove_edge(*edge)
    
    # Convert back to PyG
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Make it bidirectional again
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    perturbed_graph.edge_index = edge_index
    
    return perturbed_graph


def compute_ecs(mask, ground_truth, pred):
    '''
    Computes the ECS metric
    params~
    mask = boolean mask for which you want to filter
    ground_truth = Ground truth communities
    pred = predicted communities
    '''
    mask = mask.numpy(force=True)
    mask = np.where(mask)
    gt = {key:ground_truth[key] for key in mask[0]}
    ground_truth = gt
    
    pred_community = {i:[pred[i].numpy(force=True).item()] for i in mask[0]}
    ground_truth_community = {key:[value] for key, value in ground_truth.items()}
    
    clustering_true = Clustering(elm2clu_dict=ground_truth_community)
    clustering_pred = Clustering(elm2clu_dict=pred_community)
    
    ecs = sim.element_sim(clustering_true, clustering_pred)
    return ecs

def train_step(model: torch.nn.Module, 
               data: Data, 
               optimizer: torch.optim.Optimizer) -> float:
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Ensure data is on same device as model
    device = next(model.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    
    # Get number of unique classes
    num_classes = model.convs[-1].out_channels
    
    # Ensure labels are in correct range
    if y.max() >= num_classes:
        raise ValueError(f"Labels must be in range [0, {num_classes-1}], but found max label {y.max()}")
    
    out = model(x, edge_index)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test_step(model: torch.nn.Module, 
              data: Data) -> Tuple[float, float]:
    """
    Evaluate model on data using ECS and accuracy
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        
        # Move tensors to CPU for metric computation
        y_cpu = y.cpu()
        pred_cpu = pred.cpu()
        
        # Calculate metrics
        acc = accuracy_score(y_cpu.numpy(), pred_cpu.numpy())
        ecs = compute_ecs(torch.ones_like(y_cpu, dtype=torch.bool), 
                         {i: y_cpu[i].item() for i in range(len(y_cpu))},
                         pred_cpu)
        
    return ecs, acc
