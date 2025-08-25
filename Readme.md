# GNN Community Detection Robustness

Code for the paper "Community detection robustness of graph neural networks"

## What this does

Tests how well different Graph Neural Networks can find communities in networks when the data is noisy or under attack.


## Packages required

```bash
pip install torch torch-geometric networkx scikit-learn numpy matplotlib networkx
```

## Quick Start


## Methods Tested

**Supervised:**
- GCN - Graph Convolutional Network  
- GAT - Graph Attention Network
- GraphSAGE - Sample and Aggregate

**Unsupervised:**
- DiffPool - Hierarchical pooling
- MinCUT - Spectral clustering  
- DMoN - Modularity optimization ⭐ **Most Robust**

## Datasets

**Synthetic:**
- LFR benchmark graphs
- Stochastic Block Models

**Real:**
- Cora (papers: 2,708 nodes)
- CiteSeer (papers: 3,327 nodes)  
- PubMed (papers: 19,717 nodes)

## Attacks Tested

1. **Node attribute noise** - Add random noise to features
2. **Edge removal** - Delete random or important edges  
3. **Adversarial attacks** - Nettack and Metattack

## Run Experiments



## Contact

- Jaidev Goel - jaidev@vt.edu
- Code: https://github.com/jdg-stat/robustgnn
