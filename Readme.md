# GNN Community Detection Robustness
Code for the paper "Community detection robustness of graph neural networks"

## What this does
Tests how well different Graph Neural Networks can find communities in networks when the data is noisy or under attack.

## Dependencies

### Core Packages
| Package | Version | Description |
|---------|---------|-------------|
| `torch` | 2.2.0 | Deep learning framework |
| `torch-geometric` | 2.7.0 | Graph neural network library |
| `numpy` | 1.26.0 | Numerical computing |
| `pandas` | 2.1.4 | Data manipulation |
| `networkx` | 3.1 | Graph generation and analysis |
| `matplotlib` | 3.8 | Plotting and visualization |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `clusim` | 0.4 | Clustering similarity metrics |

### PyTorch Geometric Sub-dependencies
These are required by `torch-geometric` and must match your PyTorch and CUDA versions:

| Package | Version |
|---------|---------|
| `torch_scatter` | 2.1.2+pt22cu118 |
| `torch_sparse` | 0.6.18+pt22cu118 |
| `torch_cluster` | 1.6.3+pt22cu118 |
| `torch_spline_conv` | 1.2.2+pt22cu118 |

### Local Package
- `RobustGNN` — Custom package included in this repository (provides `RobustGNN.utils` and `RobustGNN.attributed_graph_generator`)

### Installation

1. **Install PyTorch** (match your CUDA version):
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

2. **Install PyTorch Geometric and its dependencies**:
```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-geometric==2.7.0
```

3. **Install remaining packages**:
```bash
pip install numpy==1.26.0 pandas==2.1.4 networkx matplotlib seaborn==0.13.2 clusim==0.4
```

4. **Install the local RobustGNN package**:
```bash
pip install -e .
```

## Repository Structure

```
RobustGNN/
├── RobustGNN/
│   ├── Experiment Notebooks/   # Jupyter notebooks for running experiments
│   ├── Images/                 # Generated plots and figures
│   ├── __init__.py
│   ├── attributed_graph_generator.py  # Synthetic graph generation (LFR, DCSBM)
│   ├── models.py               # GNN model definitions (GCN, GAT, SAGE, DiffPool, MinCUT, DMoN)
│   ├── plotter.py              # Visualization and plotting functions
│   ├── train.py                # Training and evaluation loops
│   └── utils.py                # Utility functions and helpers
├── README.md
└── setup.py
```

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
- LFR benchmark graphs With Attributes
- Degree Corrected Stochastic Block Models with Attributes

**Real:**
- Cora (papers: 2,708 nodes)
- CiteSeer (papers: 3,327 nodes)  
- PubMed (papers: 19,717 nodes)

## Attacks Tested

1. **Node attribute noise** - Add random noise to features
2. **Edge removal** - Delete random or important edges  
3. **Adversarial attacks** - Nettack and Metattack

## Contact
- Jaidev Goel - jaidev@vt.edu
- Code: https://github.com/jdg-stat/robustgnn
