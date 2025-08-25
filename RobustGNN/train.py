from RobustGNN.attributed_graph_generator import *
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DMoNPooling, DenseGCNConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, dropout_adj, degree
from torch_geometric.datasets import Planetoid
from torch.nn import Linear
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from clusim.clustering import Clustering
import clusim.sim as sim

n_nodes = 1000
mu = 0.1
feat_len = 36
generator = synthetic_graph_generator(type="LFR", n=n_nodes, mu=mu, feat_len=feat_len)

experimental_parameters = {"mu" : [0.1,0.2,0.3,0.4,0.5],
                           "num_nodes" : [1000,10000],
                           "experiments" : ["MEAN_SHIFT", "VAR_SHIFT",
                                            "RAND_EDGE" , "TARG_EDGE",
                                            "NETTACK"   , "METTACK"
                                           ],
                           "Means" : [1,2,3,4,5],
                           "Vars"  :[10, 20, 30, 40, 50],
                           "Rands" :[10, 20, 30, 40, 50],
                           "Targs" :[10, 20, 30, 40, 50]
                          }

runs = [f"Run_{i}" for i in range(1, 11)]
mus = [f"mu{i}" for i in range(1, 6)]
df = pd.DataFrame(index=runs, columns=mus)

class RGNN_experiments():
    def __init__(self, model_name= "GCN", graph_type = "LFR", 
                 experiment_name = "MS",num_nodes= 1000, mu = 0.1):
        self.model = self.load_model(model_name)
        self.graph = self.load_graph()

class GCN(torch.nn.Module):
   def __init__(self, num_features, num_classes):
       super().__init__()
       self.conv1 = GCNConv(num_features, 16)
       self.conv2 = GCNConv(16, num_classes)

   def forward(self, x, edge_index):
       x = self.conv1(x, edge_index)
       x = F.relu(x)
       x = F.dropout(x, training=self.training)
       x = self.conv2(x, edge_index)
       return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=feat_len, num_classes=generator.graph.y.max().item() + 1).to(device)
data = generator.graph.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
   model.train()
   optimizer.zero_grad()
   out = model(data.x, data.edge_index)
   loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
   loss.backward()
   optimizer.step()
   return loss

for epoch in range(200):
   loss = train()
   if (epoch + 1) % 10 == 0:
       print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data_cora = dataset[0].to(device)

class GCN_Cora(torch.nn.Module):
   def __init__(self, num_features, num_classes):
       super().__init__()
       self.conv1 = GCNConv(num_features, 32)
       self.conv2 = GCNConv(32, num_classes)

   def forward(self, x, edge_index):
       x = self.conv1(x, edge_index)
       x = F.relu(x)
       x = F.dropout(x, training=self.training)
       x = self.conv2(x, edge_index)
       return F.log_softmax(x, dim=1)

model_cora = GCN_Cora(num_features=1433, num_classes=data_cora.y.max().item() + 1).to(device)
data_cora = data_cora.to(device)
optimizer_cora = torch.optim.Adam(model_cora.parameters(), lr=0.01)

def train_cora():
   model_cora.train()
   optimizer_cora.zero_grad()
   out = model_cora(data_cora.x, data_cora.edge_index)
   loss = F.nll_loss(out[data_cora.train_mask], data_cora.y[data_cora.train_mask])
   loss.backward()
   optimizer_cora.step()
   return loss

def test_cora():
   model_cora.eval()
   with torch.no_grad():
       out = model_cora(data_cora.x, data_cora.edge_index)
       pred = out.argmax(dim=1)
       train_acc = (pred[data_cora.train_mask] == data_cora.y[data_cora.train_mask]).float().mean()
       test_acc = (pred[data_cora.test_mask] == data_cora.y[data_cora.test_mask]).float().mean()
   return train_acc, test_acc

for epoch in range(200):
   loss = train_cora()
   if (epoch + 1) % 10 == 0:
       train_acc, test_acc = test_cora()
       print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

def compute_ecs(mask, ground_truth, pred):
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

class DMoN(torch.nn.Module):
    def __init__(self, in_channels, num_clusters, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], 1000)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_clusters)
        
    def forward(self, x, edge_index):
        x = F.selu(self.conv1(x, edge_index))
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        
        ca, x, adj, sl1, ol1, cl1 = self.pool1(x, adj, mask)
        
        x = F.selu(self.conv2(x, adj))
        ca, x, adj, sl2, ol2, cl2 = self.pool2(x, adj)
        
        return ca.squeeze(0), sl1 + sl2 + ol1 + ol2 + cl1 + cl2

def perturb_mean(x, factor):
    means = x.mean(dim=0)
    perturbed_means = means * factor
    return x + (perturbed_means - means)

def perturb_variance(x, factor):
    means = x.mean(dim=0)
    centered_x = x - means
    perturbed_x = centered_x * factor
    return perturbed_x + means

def targeted_edge_deletion(edge_index, deletion_rate, degrees):
    if deletion_rate == 0:
        return edge_index
        
    num_edges = edge_index.size(1)
    num_to_delete = int(deletion_rate * num_edges)
    
    edge_importance = degrees[edge_index[0]] + degrees[edge_index[1]]
    _, indices = torch.sort(edge_importance, descending=True)
    
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[indices[:num_to_delete]] = False
    
    return edge_index[:, mask]

def run_experiments():
    mu_values = np.linspace(0.1, 0.5, 5)
    mean_factors = np.linspace(1, 5, 5)
    var_factors = np.linspace(1, 5, 5)
    edge_deletion_rates = np.linspace(0, 0.7, 8) 
    feat_len = 32
    
    results = {
        'variance_perturbation': {mu: [] for mu in mu_values},
        'mean_perturbation': {mu: [] for mu in mu_values},
        'random_edge': {mu: [] for mu in mu_values},
        'targeted_edge': {mu: [] for mu in mu_values}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for mu in mu_values:
        print(f"Processing μ = {mu:.2f}")
        generator = synthetic_graph_generator(type="LFR", n=n_nodes, mu=mu, feat_len=feat_len)
        data = generator.graph
        true_communities = generator.y
        num_clusters = len(torch.unique(true_communities))
        
        full_mask = torch.ones(data.x.size(0), dtype=torch.bool)
        
        model = DMoN(feat_len, num_clusters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            cluster_assignments, total_loss = model(
                data.x.to(device), 
                data.edge_index.to(device)
            )

            nll_loss = F.nll_loss(F.log_softmax(cluster_assignments, dim=-1), true_communities.to(device))
            loss = nll_loss + total_loss
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for factor in var_factors:
                print(f"  Variance factor: {factor:.1f}")
                x_var_perturbed = perturb_variance(data.x, factor)
                cluster_assignments, _ = model(x_var_perturbed.to(device), data.edge_index.to(device))
                cluster_assignments = cluster_assignments.argmax(dim=-1).cpu()
                sim = compute_ecs(full_mask, true_communities.cpu().numpy(), cluster_assignments)
                results['variance_perturbation'][mu].append(sim)
                
            for factor in mean_factors:
                print(f"  Mean factor: {factor:.1f}")
                x_mean_perturbed = perturb_mean(data.x, factor)
                cluster_assignments, _ = model(x_mean_perturbed.to(device), data.edge_index.to(device))
                cluster_assignments = cluster_assignments.argmax(dim=-1).cpu()
                sim = compute_ecs(full_mask, true_communities.cpu().numpy(), cluster_assignments)
                results['mean_perturbation'][mu].append(sim)
            
            degrees = degree(data.edge_index[0])
            for rate in edge_deletion_rates:
                print(f"  Edge deletion rate: {rate*100:.0f}%")
                if rate == 0:
                    edge_index_random = data.edge_index
                else:
                    edge_index_random, _ = dropout_adj(data.edge_index, p=float(rate))
                cluster_assignments, _ = model(data.x.to(device), edge_index_random.to(device))
                cluster_assignments = cluster_assignments.argmax(dim=-1).cpu()
                sim = compute_ecs(full_mask, true_communities.cpu().numpy(), cluster_assignments)
                results['random_edge'][mu].append(sim)
                
                edge_index_targeted = targeted_edge_deletion(data.edge_index, float(rate), degrees)
                cluster_assignments, _ = model(data.x.to(device), edge_index_targeted.to(device))
                cluster_assignments = cluster_assignments.argmax(dim=-1).cpu()
                sim = compute_ecs(full_mask, true_communities.cpu().numpy(), cluster_assignments)
                results['targeted_edge'][mu].append(sim)
    
    return results

def plot_results(results):
    palette = sns.color_palette("colorblind")
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.2)
    
    experiment_types = ['variance_perturbation', 'mean_perturbation', 'random_edge', 'targeted_edge']
    x_labels = ['Variance Factor', 'Mean Factor', 
                'Edge Deletion Rate (%)', 'Edge Deletion Rate (%)']
    x_values = {
        'variance_perturbation': np.linspace(1, 5, 5),
        'mean_perturbation': np.linspace(1, 5, 5),
        'random_edge': np.linspace(0, 70, 8),
        'targeted_edge': np.linspace(0, 70, 8)
    }
    
    mu_values = sorted(results['mean_perturbation'].keys())
    
    legend_ax = fig.add_axes([0.1, 0.95, 0.8, 0.05])
    legend_ax.axis('off')
    
    legend_lines = []
    legend_labels = []
    for mu_idx, mu in enumerate(mu_values):
        line, = legend_ax.plot([0], [0], color=palette[mu_idx], label=f'μ={mu:.1f}')
        legend_lines.append(line)
        legend_labels.append(f'μ={mu:.1f}')
    
    legend_ax.legend(legend_lines, legend_labels, loc='center', ncol=len(mu_values), 
                    frameon=False, bbox_to_anchor=(0.5, 0.5))
    
    for idx, (exp_type, x_label) in enumerate(zip(experiment_types, x_labels)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        for mu_idx, mu in enumerate(mu_values):
            similarities = results[exp_type][mu]
            ax.plot(x_values[exp_type], similarities, 
                   marker='o', 
                   color=palette[mu_idx])
        
        ax.set_xlabel(x_label)
        if idx in [0, 2]:
            ax.set_ylabel('Element-centric Similarity')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig

if __name__ == "__main__":
    torch.manual_seed(31)
    results = run_experiments()
    fig = plot_results(results)
    
    save_path = "dmon_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
