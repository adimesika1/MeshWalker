import os

from easydict import EasyDict
import numpy as np

import utils

def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps

def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True

  return seq, jumps

def old_get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.int32)
  visited[-1] = 1
  visited[f0] = 1
  jump_count = 0
  for i in range(1, seq_len + 1):
    to_consider = [n for n in kdtr[seq[i - 1]] if visited[n] <= 10]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] += 1

  return seq, jumps


def get_model():
  from dataset_prepare import prepare_kdtree
  from dataset import load_model_from_npz

  model_fn = os.path.expanduser('~') + '/MeshWalker/datasets_processed/modelnet40_normal_resampled/train__car__car_0072.npz'
  model = load_model_from_npz(model_fn)
  model_dict = EasyDict({'vertices': np.asarray(model['vertices']), 'faces': np.asarray(model['faces']),
                   'edges': np.asarray(model['edges']), 'n_vertices': model['vertices'].shape[0]})
  prepare_kdtree(model_dict)
  return model_dict

def show_walk_on_model():
  walks = []
  for i in range(1):
    f0 = np.random.randint(model['vertices'].shape[0])
    walk, jumps = get_seq_random_walk_local_jumps(model, f0, 1000)
    walks.append(walk)
  vertices = model['vertices']
  if 0:
    dxdydz = np.diff(vertices[walk], axis=0)
    for i, title in enumerate(['dx', 'dy', 'dz']):
      plt.subplot(3, 1, i + 1)
      plt.plot(dxdydz[:, i])
      plt.ylabel(title)
    plt.suptitle('Walk features on Human Body')
  utils.visualize_model(vertices, model['faces'],
                               line_width=1, show_edges=1, edge_color_a='gray',
                               show_vertices=True, opacity=0.8,
                               point_size=4, all_colors='white',
                               walk=walks)


if __name__ == '__main__':
  utils.config_gpu(False)
  model = get_model()
  np.random.seed(999)
  show_walk_on_model()