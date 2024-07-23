'''
Has classes of all the models investigated. 
They include, GCN, GAT, MinCut, GraphSAGE, DiffPool, DMon

'''


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,dense_mincut_pool, dense_diff_pool
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear




class GCN(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.x = data.x 
        self.num_feats = data.x.shape[1]
        self.num_labels= len(set(data.y))
        self.conv1 = GCNConv(self.num_feats, 16)
        self.conv2 = GCNConv(16, self.num_labels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self,data):
        super(GAT, self).__init__()
        
        self.num_classes = len(set(data.y))
        self.num_feats = data.x.shape[1]
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.conv1 = GATConv(self.num_feats, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, self.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1) 

class SAGE(torch.nn.Module):
    def __init__(self, data):
        super(SAGE, self).__init__()
        self.num_classes = len(set(data.y))
        self.num_feats = data.x.shape[1]
        self.conv1 = SAGEConv(self.num_feats, 16)
        self.conv2 = SAGEConv(16, self.num_classes)

    def forward(self, data):
        x,edge_index = data.x,data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class MinCut(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        in_channels = data.x.shape[1]
        hidden_channels = 32
        out_channels = len(set(data.y))
        n_clusters= 10
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.pool = Linear(hidden_channels, n_clusters)
        
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Cluster assignments
        s = self.pool(x)
        
        # Obtain MinCutPool losses
        adj = to_dense_adj(edge_index)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)
        
        # Final classification
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=-1), mc_loss, o_loss
    

class DiffPool(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        in_channels = data.x.shape[1]
        hidden_channels = 32
        out_channels = len(set(data.y))
        n_clusters= 10
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.pool = Linear(hidden_channels, n_clusters)
        
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Cluster assignments
        s = self.pool(x)
        
        # Obtain MinCutPool losses
        adj = to_dense_adj(edge_index)
        _, _, mc_loss, o_loss = dense_diff_pool(x, adj, s)
        
        # Final classification
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=-1), mc_loss, o_loss

    

