import numpy as np

from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim


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

  data_dirty = data_clean + np.random.normal(loc = loc_scale,
      scale=noise_scale, size=data_clean.shape)  # Add random noise to the data.

  labels = np.zeros(n_points, dtype=int)
  for i in range(n_clusters):
    labels[points_per_cluster * i:points_per_cluster * (i + 1)] = i

  return data_clean, data_dirty, labels


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
