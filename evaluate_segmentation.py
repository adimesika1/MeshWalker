import os, shutil, time, copy, json, glob, pickle, glob, sys
from easydict import EasyDict
from tqdm import tqdm

import scipy
import numpy as np
import pylab as plt
import trimesh
import open3d as o3d

import tensorflow as tf

import rnn_model
import dataset
import dataset_prepare
import utils


def fill_edges(model):
  # To compare accuracies to MeshCNN, this function build edges & edges length in the same way they do
  edge2key = dict()
  edges_length = []
  edges = []
  edges_count = 0
  for face_id, face in enumerate(model['faces']):
    faces_edges = []
    for i in range(3):
      cur_edge = (face[i], face[(i + 1) % 3])
      faces_edges.append(cur_edge)
    for idx, edge in enumerate(faces_edges):
      edge = tuple(sorted(list(edge)))
      faces_edges[idx] = edge
      if edge not in edge2key:
        edge2key[edge] = edges_count
        edges.append(list(edge))
        e_l = np.linalg.norm(model['vertices'][edge[0]] - model['vertices'][edge[1]])
        edges_length.append(e_l)
        edges_count += 1
  model['edges_meshcnn'] = np.array(edges)
  model['edges_length'] = edges_length


def get_model_by_name(name):
  fn = name[name.find(':')+1:]
  mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
  model = {'vertices': mesh_data['vertices'], 'faces': mesh_data['faces'], 'labels': mesh_data['labels'],
           'edges': mesh_data['edges']}

  if 'face_labels' in mesh_data.keys():
     model['face_labels'] = mesh_data['face_labels']

  if 'category' in mesh_data.keys():
     model['category'] = mesh_data['category']

  if 'vertices_fpfh' in mesh_data.keys():
    model['vertices_fpfh'] = mesh_data['vertices_fpfh']

  if 'vertices_pointnet' in mesh_data.keys():
    model['vertices_pointnet'] = mesh_data['vertices_pointnet']

  if 'kdtree_query' in mesh_data.keys():
    model['kdtree_query'] = mesh_data['kdtree_query']

  if 'model_features' in mesh_data.keys():
    model['model_features'] = mesh_data['model_features']

  if 'model_scales' in mesh_data.keys():
    model['model_scales'] = mesh_data['model_scales']

  if 'vertex_normals' in mesh_data.keys():
    model['vertex_normals'] = mesh_data['vertex_normals']

  if len(model['faces']) == 0 and len(model['edges']) == 0:
    model['point_cloud'] = {'p_vertices': mesh_data['vertices'], 'p_labels': mesh_data['labels']}

  return model


def calc_final_accuracy(models, print_details):
  if print_details:
    print(utils.color.BOLD + utils.color.BLUE + '\n\nAccuracy report : ' + utils.color.END)

  naive_accuracy = []
  n_total_vertices = 0
  n_vertices_no_prediction = 0
  iou_all_models = dataset_prepare.partnet_categ.copy()
  iou_all_models = {x: sys.float_info.epsilon for x in iou_all_models}
  count_models_per_cat = iou_all_models.copy()
  count_models = 0

  for model_name, model in models.items():
    if model['labels'].size == 0:
      continue
    n_vertices_no_prediction += np.sum((model['pred'].sum(axis=1) == 0))
    n_total_vertices += model['pred'].shape[0]
    best_pred = np.argmax(model['pred'], axis=-1)
    model['v_pred'] = best_pred

    # Calc vertices accuracy IOU
    if len(model['faces']) == 0 and len(model['edges']) == 0:
      #Add prediction for non predicted vertices
      repredict_list = []
      all_predictions = best_pred == model['labels']
      for index, value in enumerate(model['vertices']):
        if all_predictions[index] == False:
          repredict_list.append([index, value])
      for p in repredict_list:
        neigs = model['kdtree_query'][p[0]]
        neigs_prediction = best_pred[neigs]
        elements = list(neigs_prediction[list(np.nonzero(neigs_prediction)[0])])
        if len(elements) != 0:
          best_pred[p[0]] = Most_Common(elements) #most frequent category
      model['v_pred_with_fixing'] = best_pred

      #naive calculation
      accuracy = np.sum(best_pred == model['labels']) / model['labels'].shape[0]
      naive_accuracy.append(accuracy)

      #IOU calculation
      n_parts = max(model['labels'])
      io_per_part = []
      for p in range(1, n_parts+1):
        io_per_part.append(np.round((sum((best_pred == p) & (model['labels'] == p))+sys.float_info.epsilon) /
                                    (sum((best_pred == p) | (model['labels'] == p))+sys.float_info.epsilon), 3))
      iou_all_models[model['category'].item()] += sum(io_per_part)/n_parts
      count_models_per_cat[model['category'].item()] += 1
      count_models += 1


  if len(model['faces']) == 0 and len(model['edges']) == 0:
    iou_all = [x / y for x, y in zip(list(iou_all_models.values()), list(count_models_per_cat.values()))]
    iou_instance_average = sum([w * val for w, val in zip(list(count_models_per_cat.values()), iou_all)])/sum(list(count_models_per_cat.values()))
    iou_classes_average = np.mean(iou_all)
  if print_details:
    from dataset_prepare import partnet_categ
    for c in partnet_categ:
      print('category: ', partnet_categ[c], 'mean IOU accuracy: ', iou_all_models[c] / count_models_per_cat[c])
    print('cat. mean IOU accuracy: ', np.round(iou_classes_average * 100, 2), '%')
    print('inst. mean IOU accuracy: ', np.round(iou_instance_average * 100, 2), '%')

  return 0, iou_classes_average, iou_instance_average, np.mean(naive_accuracy)

from collections import Counter
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def postprocess_vertex_predictions(models):
  # Averaging vertices with thir neighbors, to get best prediction (eg.5 in the paper)
  for model_name, model in models.items():
    pred_orig = model['pred'].copy()
    av_pred = np.zeros_like(pred_orig)
    for v in range(model['vertices'].shape[0]):
      this_pred = pred_orig[v]
      nbrs_ids = model['edges'][v]
      nbrs_ids = np.array([n for n in nbrs_ids if n != -1])
      if nbrs_ids.size:
        first_ring_pred = (pred_orig[nbrs_ids].T / model['pred_count'][nbrs_ids]).T
        nbrs_pred = np.mean(first_ring_pred, axis=0) * 0.5
        av_pred[v] = this_pred + nbrs_pred
      else:
        av_pred[v] = this_pred
    model['pred'] = av_pred


def calc_accuracy_test(logdir=None, dataset_expansion=None, dnn_model=None, params=None,
                       n_iters=32, model_fn=None, n_walks_per_model=32, data_augmentation={}):
  # Prepare parameters for the evaluation
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    params.model_fn = logdir + '/learned_model.keras'
    params.new_run = 0
  else:
    params = copy.deepcopy(params)
  if logdir is not None:
    params.logdir = logdir
  params.mix_models_in_minibatch = False
  params.batch_size = 1
  params.net_input.append('vertex_indices')
  params.n_walks_per_model = n_walks_per_model

  # Prepare the dataset
  test_dataset, n_items = dataset.tf_mesh_dataset(params, dataset_expansion, mode=params.network_task,
                                                  shuffle_size=0, size_limit=np.inf, permute_file_names=False,
                                                  must_run_on_all=True, data_augmentation=data_augmentation)

  # If dnn_model is not provided, load it
  if dnn_model is None:
    #dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - 1, model_fn, model_must_be_load=True,
      #                                 dump_model_visualization=False)
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn, model_must_be_load=True,
                                       dump_model_visualization=False)

  # Skip the 1st half of the walk to get the vertices predictions that are more reliable
  skip = int(params.seq_len * 0.5)
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  models = {}

  # Go through the dataset n_iters times
  for _ in tqdm(range(n_iters)):
    for name_, model_ftrs_, labels_ in test_dataset:
        name = name_.numpy()[0].decode()
        assert name_.shape[0] == 1
        model_ftrs = model_ftrs_[:, :, :, :-1]
        all_seq = model_ftrs_[:, :, :, -1].numpy()
        if name not in models.keys():
          models[name] = get_model_by_name(name)
          models[name]['pred'] = np.zeros((models[name]['vertices'].shape[0], params.n_classes))
          models[name]['pred_count'] = 1e-6 * np.ones((models[name]['vertices'].shape[0], )) # Initiated to a very small number to avoid devision by 0

        sp = model_ftrs.shape
        ftrs = tf.reshape(model_ftrs, (-1, sp[-2], sp[-1]))
        predictions = dnn_model(ftrs, training=False).numpy()[:, skip:]
        all_seq = all_seq[0, :, skip + 1:].reshape(-1).astype(np.int32)
        predictions4vertex = predictions.reshape((-1, predictions.shape[-1]))
        for w_step in range(all_seq.size):
          models[name]['pred'][all_seq[w_step]] += predictions4vertex[w_step]
          models[name]['pred_count'][all_seq[w_step]] += 1

  _, v_acc_iou_per_class, v_acc_iou_per_instance, v_naive_acc = calc_final_accuracy(models, print_details=True)

  # For visualization (using meshlab), dump the results to ply files
  if 0:
    for colorize_only_bad_pred in [0, 1]:
      for name, model in tqdm(models.items()):
        if 0:
          utils.visualize_model(model['vertices'], model['faces'], v_size=1, show_vertices=0, face_colors=model['face_labels'], show_edges=0)
          utils.visualize_model(model['vertices'], model['faces'], v_size=1, show_vertices=0, face_colors=model['f_pred'], show_edges=0)
          utils.visualize_model(model['vertices'], model['faces'], vertex_colors_idx=model['labels'].astype(np.int), point_size=5, show_edges=0)
          utils.visualize_model(model['vertices'], model['faces'], vertex_colors_idx=model['v_pred'].astype(np.int), point_size=5, show_edges=0)
        pred_per_node = model['pred']
        no_pred_idxs = np.where(np.abs(pred_per_node).sum(axis=1) == 0)[0]
        best_per_node = np.argmax(pred_per_node, axis=1)
        colors_parts = np.zeros((model['vertices'].shape[0], 3))
        if 0:
          wrong_pred = np.ones_like(best_per_node)
          for v in range(best_per_node.shape[0]):
            wrong_pred[v] = (model['labels4test'][v][best_per_node[v] - 1] == 0)
        else:
          wrong_pred = best_per_node != model['labels']
        for part_id in range(np.max(model['labels'])+1):#16
          if colorize_only_bad_pred:
            idxs = np.where((best_per_node == part_id) * wrong_pred)
            colors_parts[idxs] = np.array((255, 255, 51), dtype=np.float32) / 255.0
          else:
            idxs = np.where(best_per_node == part_id)
            colors_parts[idxs] = utils.index2color(part_id)
        colors_parts[no_pred_idxs] = [0, 0, 0]
        v_clrs = colors_parts

        model = EasyDict({'vertices': models[name]['vertices'], 'faces': models[name]['faces']})
        model_fn = name.split('.')[-2].split('//')[-1]
        model_fn += '_only_bad_pred' if colorize_only_bad_pred else '_all'
        db_name = name[:name.find(':')]
        utils.colorize_and_dump_model(model, [], 'debug_models/eval_' + db_name + '_' + model_fn + '.ply',
                                        vertex_colors=v_clrs, norm_clrs=0, show=False, verbose=False)
        if 'point_cloud' in models[name].keys() and (not colorize_only_bad_pred):
          p_model = EasyDict({'vertices': models[name]['vertices'], 'faces': models[name]['faces']})
          p_labels = models[name]['point_cloud']['p_labels']
          colors_parts = np.zeros((models[name]['point_cloud']['p_vertices'].shape[0], 3))
          for part_id in range(np.max(p_labels) + 1):  # 16
            idxs = np.where(p_labels == part_id)
            colors_parts[idxs] = utils.index2color(part_id)
          utils.colorize_and_dump_model(p_model, [], 'debug_models/eval_' + db_name + '_' + model_fn + '_orig.ply',
                                        vertex_colors=colors_parts, norm_clrs=0, show=False, verbose=False)
        if 'original' in models[name].keys():
          o_model = EasyDict({'vertices': models[name]['original']['o_vertices'], 'faces': models[name]['original']['o_faces']})
          o_f_colors = np.zeros((o_model['faces'].shape[0], 3))
          for i, label in enumerate(models[name]['original']['o_face_labels']):
            pred = models[name]['original']['o_face_preds'][i]
            if not colorize_only_bad_pred or pred != label:
              o_f_colors[i] = utils.index2color(pred)
          utils.colorize_and_dump_model(o_model, [], 'debug_models/eval_' + db_name + '_' + model_fn + '_orig.ply',
                                        clrs=o_f_colors, norm_clrs=0, show=False, verbose=False)
  
  return [0, v_acc_iou_per_class, v_acc_iou_per_instance, v_naive_acc], dnn_model



if __name__ == '__main__':
  from train_val import get_params
  utils.config_gpu(1)
  np.random.seed(0)
  tf.random.set_seed(0)

  if len(sys.argv) != 3:
    print('<>'.join(sys.argv))
    print('Use: python evaluate_segmentation.py <job> <part> <trained model directory>')
    print('For example: python evaluate_segmentation.py coseg chairs pretrained/0009-14.11.2020..07.08__coseg_chairs')
  else:
    logdir = sys.argv[2]
    job = sys.argv[1]
    #job_part = sys.argv[2]
    params = get_params(job, None)
    dataset_expansion = params.datasets2use['test'][0]
    accs, _ = calc_accuracy_test(logdir, dataset_expansion)
    #print('Edge accuracy:', accs[0])

