import pandas as pd
import os
import numpy as np
import random
import tensorlayerx as tlx
from scipy.spatial import distance_matrix
import scipy.sparse as sp
import tensorflow as tf

def index_to_mask(num_nodes, index):
    mask = np.zeros(num_nodes, dtype=np.int32)
    if isinstance(index, tf.Tensor):
        index = index.numpy()
    mask[index] = 1
    return mask

def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_tlx_tensor(sparse_mx):
    """
    将稀疏矩阵转换为 TensorLayerX 张量表示。
    
    参数:
    sparse_mx (scipy.sparse.coo_matrix): 一个以COO格式存储的稀疏矩阵。
    
    返回:
    dict: 包含索引、值和形状的稀疏张量表示。
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64).T
    values = sparse_mx.data
    shape = sparse_mx.shape

    # 将数据转换为 TensorLayerX 张量
    indices = tlx.convert_to_tensor(indices, dtype=tlx.int64)
    values = tlx.convert_to_tensor(values, dtype=tlx.float32)
    
    return {'indices': indices, 'values': values, 'shape': shape}
   

def feature_norm(features):
    min_values = tlx.reduce_min(features, axis=0)
    max_values = tlx.reduce_max(features, axis=0)
    return 2 * (features - min_values) / (max_values - min_values) - 1

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T, x.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map

def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="dataset/credit/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_tlx_tensor(adj_norm)

    features = tlx.convert_to_tensor(np.array(features.todense()), dtype=tlx.float32)
    labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = tlx.convert_to_tensor(sens, dtype=tlx.int64)

    train_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_train))
    val_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_val))
    test_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_test))

    from collections import Counter
    print('predict_attr:', Counter(idx_features_labels[predict_attr]))
    print('sens_attr:', Counter(idx_features_labels[sens_attr]))
    return adj_norm_sp, edges, features, labels, train_mask, val_mask, test_mask, sens, adj

def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/bail/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_tlx_tensor(adj_norm)

    features = tlx.convert_to_tensor(np.array(features.todense()), dtype=tlx.float32)
    labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = tlx.convert_to_tensor(sens, dtype=tlx.int64)
    train_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_train))
    val_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_val))
    test_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_test))

    from collections import Counter
    print('predict_attr:', Counter(idx_features_labels[predict_attr]))
    print('sens_attr:', Counter(idx_features_labels[sens_attr]))
    return adj_norm_sp, edges, features, labels, train_mask, val_mask, test_mask, sens, adj

def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/german/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_tlx_tensor(adj_norm)

    features = tlx.convert_to_tensor(np.array(features.todense()), dtype=tlx.float32)
    labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = tlx.convert_to_tensor(sens, dtype=tlx.int64)

    train_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_train))
    val_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_val))
    test_mask = index_to_mask(features.shape[0], tlx.convert_to_tensor(idx_test))

    from collections import Counter
    print('predict_attr:', Counter(idx_features_labels[predict_attr]))
    print('sens_attr:', Counter(idx_features_labels[sens_attr]))
    return adj_norm_sp, edges, features, labels, train_mask, val_mask, test_mask, sens, adj

def load_pokec(dataset, sens_attr="region", predict_attr="I_am_working_in_field", path="dataset/pokec/", label_number=3000, sens_number=500, seed=20, test_idx=True):
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_tlx_tensor(adj_norm)

    features = tlx.convert_to_tensor(np.array(features.todense()), dtype=tlx.float32)
    labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels > 0)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):],
                         label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values
    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = tlx.convert_to_tensor(sens, dtype=tlx.float32)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))

    random.shuffle(idx_sens_train)
    idx_sens_train = tlx.convert_to_tensor(idx_sens_train[:sens_number], dtype=tlx.int64)

    idx_train = tlx.convert_to_tensor(idx_train, dtype=tlx.int64)
    idx_val = tlx.convert_to_tensor(idx_val, dtype=tlx.int64)
    idx_test = tlx.convert_to_tensor(idx_test, dtype=tlx.int64)
    train_mask = index_to_mask(features.shape[0], idx_train)
    val_mask = index_to_mask(features.shape[0], idx_val)
    test_mask = index_to_mask(features.shape[0], idx_test)

    labels[labels > 1] = 1
    if sens_attr:
        sens[sens > 0] = 1

    from collections import Counter
    print('predict_attr:', Counter(idx_features_labels[predict_attr]))
    print('sens_attr:', Counter(idx_features_labels[sens_attr]))
    print('total dimension:', features.shape)

    return adj_norm_sp, edges, features, labels, train_mask, val_mask, test_mask, sens, adj

def get_dataset(dataname):
    if dataname == 'credit':
        load, label_num = load_credit, 6000
    elif dataname == 'bail':
        load, label_num = load_bail, 100
    elif dataname == 'german':
        load, label_num = load_german, 100
    elif dataname == 'pokec_z':
        dataname = 'region_job'
        load, label_num = load_pokec, 3000
    elif dataname == 'pokec_n':
        dataname = 'region_job_2'
        load, label_num = load_pokec, 3000

    adj_norm_sp, edges, features, labels, train_mask, val_mask, test_mask, sens, adj = load(dataset=dataname, label_number=label_num)

    if dataname == 'credit':
        sens_idx = 1
    elif dataname == 'bail' or dataname == 'german':
        sens_idx = 0
    elif dataname == 'region_job' or dataname == 'region_job_2':
        sens_idx = 3

    x_max, x_min = tlx.reduce_max(features, axis=0), tlx.reduce_min(features, axis=0)

    if dataname != 'german':
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    data = {
        'adj': adj,
        'x': features,
        'edge_index': edges,
        'adj_norm_sp': adj_norm_sp,
        'y': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'sens': sens
    }

    return data, sens_idx, x_min, x_max
