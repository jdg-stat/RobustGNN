'''
Has classes of all the models investigated. 
They include, GCN, GAT, MinCut, GraphSAGE, DiffPool, DMon

'''


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv




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
        x = F.elu(x)
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

    

