import glob, os, shutil, sys, json
from pathlib import Path

import pandas
import pylab as plt
import trimesh
import open3d
from easydict import EasyDict
import numpy as np
from pyntcloud import PyntCloud
from tqdm import tqdm
import open3d as o3d
from scipy import spatial
from sklearn import neighbors
from plyfile import PlyData, PlyElement

import utils


# Labels for all datasets
# -----------------------
sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}

model_net_labels = [
  'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone'
  , 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard'
  , 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio'
  , 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase'
  , 'wardrobe', 'xbox'

]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}

cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}

coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}

partnet_categ = {"02691156": "Airplane",
                 "02773838": "Bag",
                 "02954340": "Cap",
                 "02958343": "Car",
                 "03001627": "Chair",
                 "03261776": "Earphone",
                 "03467517": "Guitar",
                 "03624134": "Knife",
                 "03636649": "Lamp",
                 "03642806": "Laptop",
                 "03790512": "Motorbike",
                 "03797390": "Mug",
                 "03948459": "Pistol",
                 "04099429": "Rocket",
                 "04225987": "Skateboard",
                 "04379243": "Table"}


def calc_mesh_area(mesh):
  t_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=False)
  mesh['area_faces'] = t_mesh.area_faces
  mesh['area_vertices'] = np.zeros((mesh['vertices'].shape[0]))
  for f_index, f in enumerate(mesh['faces']):
    for v in f:
      mesh['area_vertices'][v] += mesh['area_faces'][f_index] / f.size

def prepare_kdtree(point_cloud):
  vertices = point_cloud['vertices']
  point_cloud['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    point_cloud['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  point_cloud['kdtree_query'] = np.array(point_cloud['kdtree_query'])
  assert point_cloud['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(point_cloud['kdtree_query'].shape[1])


def prepare_new_kdtree_fpfh(point_cloud):
  vertices = point_cloud['vertices']
  vertices_fpfh = point_cloud['vertices_fpfh']
  point_cloud['kdtree_query'] = []
  tree_eucli = neighbors.KDTree(vertices, metric='euclidean')
  n_nbrs = max(min(11, vertices.shape[0] - 2), 0)
  for n in range(vertices.shape[0]):
    d, i = tree_eucli.query([vertices[n]], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i[0] if np.logical_and(inbr != n, inbr < vertices.shape[0])]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]

    n_nbrs_fpfh = max(min(3, len(vertices_fpfh[i_nbrs_cleared]) - 2), 0)
    nbrs = neighbors.NearestNeighbors(n_neighbors=n_nbrs_fpfh, algorithm='ball_tree', leaf_size=1)
    nbrs.fit(vertices_fpfh[i_nbrs_cleared])
    nbrs_dist, nbrs_ind = nbrs.kneighbors([vertices_fpfh[n]])
    point_cloud['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32)[nbrs_ind[0]])

    #tree_fpfh = neighbors.KDTree(vertices_fpfh[i_nbrs_cleared])
    #n_nbrs_fpfh = max(min(3, len(vertices_fpfh[i_nbrs_cleared]) - 2), 0)
    #if n_nbrs_fpfh != 0:
    #  d_fpfh, i_fpfh = tree_fpfh.query([vertices_fpfh[n]], n_nbrs_fpfh)
    #  if len(i_fpfh) > 0:
    #    if len(i_fpfh) > n_nbrs_fpfh - 1:
    #      i_fpfh = i_fpfh[:n_nbrs - 1]
    #    point_cloud['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32)[i_fpfh[0]])
    #else:
    #  point_cloud['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  point_cloud['kdtree_query'] = np.array(point_cloud['kdtree_query'])
  if point_cloud['kdtree_query'].shape[1] != n_nbrs_fpfh:
    assert point_cloud['kdtree_query'].shape[1] == n_nbrs_fpfh, 'Number of kdtree_query is wrong: ' + str(point_cloud['kdtree_query'].shape[1])


def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])


def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' and m['faces'].shape[0] == 0: #for point cloud
          prepare_kdtree(m)  # to remove
      #elif field == 'kdtree_query':
      #    prepare_edges_and_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)

  return m


def get_sig17_seg_bm_labels(mesh, file, seg_path):
  # Finding the best match file name .. :

  in_to_check = file.replace('obj', 'txt')
  in_to_check = in_to_check.replace('off', 'txt')
  in_to_check = in_to_check.replace('_fix_orientation', '')
  if in_to_check.find('MIT_animation') != -1 and in_to_check.split('\\')[-1].startswith('mesh_'):
    in_to_check = '\\'.join(in_to_check.split('\\')[:-2])
    in_to_check = in_to_check.replace("MIT_animation\\meshes_", "mit\\mit_")
    in_to_check += '.txt'
  elif in_to_check.find('scape') != -1:
    in_to_check = '\\'.join(in_to_check.split('\\')[:-1])
    in_to_check += '\\scape.txt'
  elif in_to_check.find('faust') != -1:
    in_to_check = '\\'.join(in_to_check.split('\\')[:-1])
    in_to_check += '\\faust.txt'

  seg_full_fn = []
  for fn in Path(seg_path).rglob('*.txt'):
    tmp = str(fn)
    tmp = tmp.replace('segs', 'meshes')
    tmp = tmp.replace('_full', '')
    tmp = tmp.replace('shrec_', '')
    tmp = tmp.replace('_corrected', '')
    if tmp == in_to_check:
      seg_full_fn.append(str(fn))
  if len(seg_full_fn) == 1:
    seg_full_fn = seg_full_fn[0]
  else:
    print('\nin_to_check', in_to_check)
    print('tmp', tmp)
    raise Exception('!!')
  face_labels = np.loadtxt(seg_full_fn)

  if FIX_BAD_ANNOTATION_HUMAN_15 and file.endswith('test\\shrec\\15.off'):
    face_center = []
    for f in mesh.faces:
      face_center.append(np.mean(mesh.vertices[f, :], axis=0))
    face_center = np.array(face_center)
    idxs = (face_labels == 6) * (face_center[:, 0] < 0) * (face_center[:, 1] < -0.4)
    face_labels[idxs] = 7
    np.savetxt(seg_full_fn + '.fixed.txt', face_labels.astype(np.int))

  return face_labels


def get_labels(dataset_name, mesh, file, fn2labels_map=None):
  v_labels_fuzzy = np.zeros((0,))
  if dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
    labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
    e_labels = np.loadtxt(labels_fn)
    v_labels = [[] for _ in range(mesh['vertices'].shape[0])]
    faces = mesh['faces']

    fuzzy_labels_fn = '/'.join(file.split('/')[:-2]) + '/sseg/' + file.split('/')[-1].split('.')[-2] + '.seseg'
    seseg_labels = np.loadtxt(fuzzy_labels_fn)
    v_labels_fuzzy = np.zeros((mesh['vertices'].shape[0], seseg_labels.shape[1]))

    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
      faces_edges = []
      for i in range(3):
        cur_edge = (face[i], face[(i + 1) % 3])
        faces_edges.append(cur_edge)
      for idx, edge in enumerate(faces_edges):
        edge = tuple(sorted(list(edge)))
        faces_edges[idx] = edge
        if edge not in edge2key:
          v_labels_fuzzy[edge[0]] += seseg_labels[edges_count]
          v_labels_fuzzy[edge[1]] += seseg_labels[edges_count]

          edge2key[edge] = edges_count
          edges.append(list(edge))
          v_labels[edge[0]].append(e_labels[edges_count])
          v_labels[edge[1]].append(e_labels[edges_count])
          edges_count += 1

    assert np.max(np.sum(v_labels_fuzzy != 0, axis=1)) <= 3, 'Number of non-zero labels must not acceeds 3!'

    vertex_labels = []
    for l in v_labels:
      l2add = np.argmax(np.bincount(l))
      vertex_labels.append(l2add)
    vertex_labels = np.array(vertex_labels)
    model_label = np.zeros((0,))

    return model_label, vertex_labels, v_labels_fuzzy
  else:
    tmp = file.split('/')[-1]
    model_name = '_'.join(tmp.split('_')[:-1])
    if dataset_name.lower().startswith('modelnet'):
      model_label = model_net_shape2label[model_name]
    elif dataset_name.lower().startswith('cubes'):
      model_label = cubes_shape2label[model_name]
    elif dataset_name.lower().startswith('shrec11'):
      model_name = file.split('/')[-3]
      if fn2labels_map is None:
        model_label = shrec11_shape2label[model_name]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    else:
      raise Exception('Cannot find labels for the dataset')
    vertex_labels = np.zeros((0,))
    return model_label, vertex_labels, v_labels_fuzzy


def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
  labels = labels_orig
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    str_to_add = '_simplified_to_' + str(target_n_faces)
    mesh = mesh.remove_unreferenced_vertices()
    if add_labels and labels_orig.size:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
  else:
    mesh = mesh_orig
    str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

  return mesh, labels, str_to_add

def fix_labels_by_dist(vertices, orig_vertices, labels_orig):
  labels = -np.ones((vertices.shape[0], ))

  for i, vertex in enumerate(vertices):
    d = np.linalg.norm(vertex - orig_vertices, axis=1)
    orig_idx = np.argmin(d)
    labels[i] = labels_orig[orig_idx]

  return labels

def load_mesh(model_fn, classification=True):
  # To load and clean up mesh - "remove vertices that share position"
  if classification:
    mesh_ = trimesh.load_mesh(model_fn, process=True)
    mesh_.remove_duplicate_faces()
  else:
    mesh_ = trimesh.load_mesh(model_fn, process=False)
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
  mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)

  return mesh

def create_tmp_dataset(model_fn, p_out, n_target_faces):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  mesh_orig = load_mesh(model_fn)
  mesh, labels, str_to_add = remesh(mesh_orig, n_target_faces)
  labels = np.zeros((np.asarray(mesh.vertices).shape[0],), dtype=np.int16)
  mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': 0, 'labels': labels})
  out_fn = p_out + '/tmp'
  add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, 'tmp')


def prepare_directory(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True):
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'vertices_fpfh']
  fileds_needed += ['labels_fuzzy']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  radius_normal = 0.01
  radius_feature = 0.1
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    mesh_orig = mesh
    pointcloud_dict = EasyDict({'vertices': np.asarray(mesh.vertices)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      label, labels_orig, v_labels_fuzzy = get_labels(dataset_name, pointcloud_dict, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      point_cloud_norm = normalize_point_cloud(mesh.vertices)
      point_cloud = EasyDict({'vertices': np.asarray(point_cloud_norm), 'faces': np.asarray([]), 'label': label, 'labels': labels})
      point_cloud['labels_fuzzy'] = v_labels_fuzzy
      vertices_fpfh = prepare_pointcloud(np.asarray(point_cloud['vertices']), radius_normal, radius_feature)
      point_cloud['vertices_fpfh'] = np.asarray(vertices_fpfh)
      out_fc_full = out_fn + str_to_add
      m = add_fields_and_dump_model(point_cloud, fileds_needed, out_fc_full, dataset_name)
    write_ply(point_cloud['vertices'], out_fc_full + '_off.ply')

def write_ply(points, filename, text=True):
  """ input: Nx3, write points to filename as PLY format. """
  points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
  vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
  PlyData([el], text=text).write(filename)

def pc_normalize(pc):
  l = pc.shape[0]
  centroid = np.mean(pc, axis=0)
  pc = pc - centroid
  m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
  pc = pc / m
  return pc

# ------------------------------------------------------- #

def prepare_modelnet40():
  n_target_faces = [2048]
  labels2use = model_net_labels
  for i, name in tqdm(enumerate(labels2use)):
    for part in ['test', 'train']:
      pin = 'datasets_raw/ModelNet40/' + name + '/' + part + '/'
      p_out = 'datasets_processed/modelnet40/'
      prepare_directory('modelnet40', pathname_expansion=pin + '*.off',
                        p_out=p_out, add_labels='modelnet', n_target_faces=n_target_faces,
                        fn_prefix=part + '_', verbose=False)

def prepare_modelnet40_normal_resampled():
  p = 'datasets_raw/modelnet40_normal_resampled/'
  p_out = 'datasets_processed-tmp/modelnet40_normal_resampled/'
  p_features = 'datasets_raw/modelnetFromPointNet'
  p_scale = 'datasets_raw/modelnet40_scales'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  catfile = os.path.join(p, 'modelnet40_shape_names.txt')
  cat = [line.rstrip() for line in open(catfile)]
  classes = dict(zip(cat, range(len(cat))))
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'model_features', 'model_scales', 'vertex_normals']
  npoints = 5000
  normalize = True
  for part in ['test', 'train']:
        print('part: ', part)
        count_files = 0
        path_models_file_per_part = p + 'modelnet40_' + part + '.txt'
        files_name = [line.rstrip() for line in open(path_models_file_per_part)]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in files_name]
        datapaths = [(files_name[i], shape_names[i], os.path.join(p, shape_names[i], files_name[i]) + '.txt') for i in range(len(files_name))]
        for f in datapaths:
          path_features_per_model = p_features + '/' + part + '/512_' + f[1] + '_' + f[0] + '.txt'
          point_set_features = np.loadtxt(path_features_per_model).astype(np.float32)
          point_set = np.loadtxt(f[2], delimiter=',').astype(np.float32)
          cls = classes[f[1]]

          #get scale
          path_scale_per_model = p_scale + '/scale_' + f[0] + '.txt'
          point_set_scale = np.loadtxt(path_scale_per_model).astype(np.float32)

          # Take the first npoints
          point_set = point_set[0:npoints, :]
          if normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

          file_name = os.path.basename(str(f))
          (prefix, sep, suffix) = file_name.rpartition('.')
          out_fn = p_out + part + '__' + f[1] + '__' + prefix
          if os.path.isfile(out_fn + '.npz'):
            continue

          point_cloud_dict = EasyDict({'vertices': np.asarray(point_set[:, 0:3]), 'faces': np.asarray([]),
                                       'edges': np.asarray([]), 'label': cls, 'labels': np.asarray([]),
                                       'model_features': np.asarray(point_set_features.reshape([1, -1])),
                                       'model_scales': point_set_scale, 'vertex_normals': np.asarray(point_set[:, 3:6])})
          add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)
          count_files += 1



def prepare_cubes(labels2use=cubes_labels,
                  path_in='datasets_raw/from_meshcnn/cubes/',
                  p_out='datasets_processed/cubes'):
  dataset_name = 'cubes'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test', 'train']:
      pin = path_in + name + '/' + part + '/'
      prepare_directory(dataset_name, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                     classification=False)


def PartNetPointNet_segmentation():
  dataset_name = 'ShapeNet_WithPointNet2Features'
  labels_fuzzy = False
  partnet_seg_path = 'datasets_raw/ShapeNet_WithPointNet2Features/'
  partnet_seg_label_path = 'datasets_raw/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
  p_out = 'datasets_processed-tmp/ShapeNet_WithPointNet2Features/'
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'category', 'vertices_pointnet']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  for part in ['test', 'val', 'train']:
    print('part: ', part)
    path_models_dir = partnet_seg_path + part
    fns = glob.glob(path_models_dir + "/*.txt")
    for f in fns:
      category_id = os.path.basename(f).split("_")
      data = np.loadtxt(f).astype(np.float32)
      path_orig_file = partnet_seg_label_path + category_id[0] + '/' + category_id[1]
      orig_data = np.loadtxt(path_orig_file).astype(np.float32)
      labels_per_point = np.asarray(orig_data[:, 6], dtype=int)
      unique_labels = (np.unique(labels_per_point))

      count = 1
      new_labels = []
      for j in labels_per_point:
        if j in unique_labels:
          new_labels.append(np.where(unique_labels == j)[0][0]+1)

      point_cloud_dict = EasyDict({'vertices': np.asarray(data[:,0:3]), 'faces': np.asarray([]),
                                   'edges': np.asarray([]), 'label': -1, 'labels': np.asarray(new_labels),
                                   'category': str(category_id[0]), 'vertices_pointnet': data[:,6:]})
      if part == 'val':
        part = 'train'
      out_fn = p_out + part + '__' + category_id[0] + '__' + category_id[1]
      if os.path.isfile(out_fn + '.npz'):
        continue
      add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)


def partnet_segmentation():
  dataset_name = 'partnet_ICCV17'
  labels_fuzzy = False
  partnet_seg_path = os.path.expanduser('~') + '\\mesh_walker\\datasets_raw\\partnet_ICCV17\\'
  p_out = os.path.expanduser('~') + '\\mesh_walker\\datasets_processed-tmp\\partnet_ICCV17\\'
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'category', 'vertices_fpfh']

  if labels_fuzzy:
    fileds_needed += ['labels_fuzzy']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  for part in ['test', 'val', 'train']:
    print('part: ', part)
    path_models_dir = partnet_seg_path + part + '_data'
    path_lables_dir = partnet_seg_path + part + '_label'
    all_catns = []
    onlyCats = [f for f in os.listdir(path_models_dir)]

    radius_normal = 0.01
    radius_feature = 0.1
    for cn in tqdm(onlyCats):
      onlyFiles = [f for f in os.listdir(path_models_dir + '\\' +cn)]
      for mn in tqdm(onlyFiles):
        if mn.endswith('.pts'):
          new_fn = name_only = mn.split('.')[-2]
          pnt_cloud_path = path_models_dir + '\\' + cn + '\\' + mn
          coordinates = pandas.read_csv(pnt_cloud_path, sep=" ", header=None, names=["x", "y", "z"])
          cloud = PyntCloud(coordinates)
          labels_path = path_lables_dir + '\\' + cn + '\\' + name_only + '.seg' #reduce the number of points
          lables = pandas.read_csv(labels_path, header=None, names=["l"])
          label = -1
          flat_labels_list = [item for sublist in lables.to_numpy() for item in sublist]
          vertices_fpfh = prepare_pointcloud(np.asarray(cloud.points), radius_normal, radius_feature)
          point_cloud_dict = EasyDict({'vertices': np.asarray(cloud.points), 'faces': np.asarray([]),
                                       'edges': np.asarray([]), 'label': label, 'labels': np.asarray(flat_labels_list),
                                       'category': str(cn), 'vertices_fpfh': vertices_fpfh})
          if part == 'val':
            part = 'train'
          out_fn = p_out + '\\' + part + '__' + cn + '__' + new_fn
          if os.path.isfile(out_fn + '.npz'):
            continue
          add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)


def prepare_pointcloud(vertices, radius_normal, radius_feature):
  #create point cloud
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(vertices)
  trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
  pcd.transform(trans_init) #verify this line

  source_fpfh = preprocess_point_cloud(pcd, radius_normal, radius_feature)

  #return pcd, source_down, source_fpfh
  return np.transpose(source_fpfh.data) #return only the features vector

def preprocess_point_cloud(pcd, radius_normal, radius_feature):
  #radius_normal = voxel_size * 2
  pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
  #radius_feature = voxel_size * 5
  print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
  pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd,
                                                   o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                                                                  max_nn=100))
  return pcd_fpfh

def normalize_point_cloud(points):
  centroid = np.mean(points, axis=0)
  points -= centroid
  furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
  points /= furthest_distance

  return points



def prepare_seg_from_meshcnn(dataset, subfolder=None):
  if dataset == 'human_body':
    dataset_name = 'human_seg_from_meshcnn'
    p_in2add = 'human_seg'
    p_out_sub = p_in2add
    p_ext = ''
  elif dataset == 'coseg':
    p_out_sub = dataset_name = 'coseg'
    p_in2add = dataset_name + '/' + subfolder
    p_ext = subfolder

  path_in = 'datasets_raw/from_meshcnn/' + p_in2add + '/'
  p_out = 'datasets_processed/' + p_out_sub + '_from_meshcnn/' + p_ext

  for part in ['test', 'train']:
    pin = path_in + '/' + part + '/'
    prepare_directory(dataset_name, pathname_expansion=pin + '*.obj',
                      p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                      classification=False)


# ------------------------------------------------------- #

def map_fns_to_label(path=None, filenames=None):
  lmap = {}
  if path is not None:
    iterate = glob.glob(path + '/*.npz')
  elif filenames is not None:
    iterate = filenames

  for fn in iterate:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    label = int(mesh_data['label'])
    if label not in lmap.keys():
      lmap[label] = []
    if path is None:
      lmap[label].append(fn)
    else:
      lmap[label].append(fn.split('/')[-1])
  return lmap


def change_train_test_split(path, n_train_examples, n_test_examples, split_name):
  np.random.seed()
  fns_lbls_map = map_fns_to_label(path)
  for label, fns_ in fns_lbls_map.items():
    fns = np.random.permutation(fns_)
    assert len(fns) == n_train_examples + n_test_examples
    train_path = path + '/' + split_name + '/train'
    if not os.path.isdir(train_path):
      os.makedirs(train_path)
    test_path = path + '/' + split_name + '/test'
    if not os.path.isdir(test_path):
      os.makedirs(test_path)
    for i, fn in enumerate(fns):
      out_fn = fn.replace('train_', '').replace('test_', '')
      if i < n_train_examples:
        shutil.copy(path + '/' + fn, train_path + '/' + out_fn)
      else:
        shutil.copy(path + '/' + fn, test_path + '/' + out_fn)


# ------------------------------------------------------- #


def prepare_one_dataset(dataset_name):
  dataset_name = dataset_name.lower()
  if dataset_name == 'modelnet40' or dataset_name == 'modelnet':
    prepare_modelnet40()

  if dataset_name == 'modelnet40_normal_resampled':
    prepare_modelnet40_normal_resampled()

  if dataset_name == 'shrec11':
    print('To do later')

  if dataset_name == 'cubes':
    prepare_cubes()

  # Semantic Segmentations
  if dataset_name == 'partnet':
    partnet_segmentation()

  if dataset_name == 'partnet_pn2':
    PartNetPointNet_segmentation()

  if dataset_name == 'human_seg':
    prepare_seg_from_meshcnn('human_body')

  if dataset_name == 'coseg':
    prepare_seg_from_meshcnn('coseg', 'coseg_aliens')
    prepare_seg_from_meshcnn('coseg', 'coseg_chairs')
    prepare_seg_from_meshcnn('coseg', 'coseg_vases')


if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(1)

  if len(sys.argv) != 2:
    print('Use: python dataset_prepare.py <dataset name>')
    print('For example: python dataset_prepare.py cubes')
    print('Another example: python dataset_prepare.py all')
  else:
    dataset_name = sys.argv[1]
    if dataset_name == 'all':
      for dataset_name in ['cubes', 'human_seg', 'coseg', 'modelnet40']:
        prepare_one_dataset(dataset_name)
    else:
      prepare_one_dataset(dataset_name)

