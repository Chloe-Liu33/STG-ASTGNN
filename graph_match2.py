import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F


def preprocess(type, npz_path):
    if type == 1:
        features = np.load(npz_path)['data'][:2016, :, 0]
    elif type == 2:
        features = np.load(npz_path)['data'][4032:, :, 0]
    mean = np.mean(features)
    std = np.std(features)
    features = (features - mean) / std
    features = features.T
    features_mean = np.mean(features, axis=0)

    # feature_level = []
    # for i, value in enumerate(features_mean):
    #     value_new = value/max
    #     # print(value_new)
    #     if value_new <=0.25:
    #         value_new =0
    #     elif value_new <= 0.5:
    #         value_new = 1
    #     elif value_new <=0.75:
    #         value_new =2
    #     else:
    #         value_new = 3
    #     feature_level.append(value_new)
    # return features_mean
    return features


def load_graph_from_files(type, npz_path, csv_path, id_path):
    # features = np.load(npz_path)['arr_0']
    features = preprocess(type, npz_path)
    print('preprocess graph')
    graph_structure = pd.read_csv(csv_path)
    G = nx.Graph()
    # 添加节点和对应的特征
    for index, feature in enumerate(features):
        # G.nodes[index]['feature'] = feature
        G.add_node(index, label=feature)  ## need 2 position but get 3
        # G.add_node(index, feature=feature.tolist())
    # 假设csv文件有两列，表示边的两个端点
    # if id_path isexist:
    if id_path is None:
        for _, row in graph_structure.iterrows():
            G.add_edge(row[0], row[1])
    else:
        with open(id_path, 'r') as f:
            a = f.read().strip('/n').split(',')  ##list
            id_dict = {int(i): idx for idx, i in enumerate(a)}
            # print(id_dict)
        for _, row in graph_structure.iterrows():
            G.add_edge(id_dict[row[0]], id_dict[row[1]])
            G.add_edge(id_dict[row[1]], id_dict[row[0]])

    return G
def load_graph_from_files_predict(adj_mx, npz_file, type):
    G = nx.Graph()
    # 添加节点和对应的特征
    l = np.load(npz_file)['data']
    if type == 0:
        features = np.load(npz_file)['data'][:l*3/5, :, 0]
    else:
        features = np.load(npz_file)['data'][l*4/5:, :, 0]

    mean = np.mean(features)
    std = np.std(features)
    features = (features - mean) / std
    features = features.T
    for index, feature in enumerate(features):
        # G.nodes[index]['feature'] = feature
        G.add_node(index, label=feature)  ## need 2 position but get 3
        # G.add_node(index, feature=feature.tolist())
    # 假设csv文件有两列，表示边的两个端点
    for i in range(100):
        for j in range(100):
            if adj_mx[i][j] !=0:
                G.add_edge(i, j)
                G.add_edge(j, i)
    return G


def GMDM(graph1, graph2):
    # 计算图1和图2的节点相似度矩阵
    sim_matrix = compute_similarity(graph1, graph2)

    # 将相似度矩阵转化为成本矩阵
    cost_matrix = 1 - sim_matrix

    # 使用linear_sum_assignment函数进行最小化成本的指派问题求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 构建匹配字典
    match_dict = {}
    for i, j in zip(row_ind, col_ind):
        match_dict[i] = j

    return match_dict

def GMDM_matrix(features1, features2):
    # 计算图1和图2的节点相似度矩阵
    sim_matrix = compute_similarity_matrix(features1, features2)

    # 将相似度矩阵转化为成本矩阵
    cost_matrix = 1 - sim_matrix
    # cost_matrix =  -torch.log(sim_matrix)
    # cost_matrix = cost_matrix.cpu()

    # 使用linear_sum_assignment函数进行最小化成本的指派问题求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())

    # 构建匹配字典
    match_dict = {}
    for i, j in zip(row_ind, col_ind):
        match_dict[i] = j

    return match_dict


# 计算节点相似度矩阵的示例函数，可根据实际需求进行定义和实现
def get_neighborhood(graph, node):
    neighbors = set(graph.neighbors(node))
    return neighbors
    # 创建空的节点相似度矩阵


def compute_similarity(graph_a, graph_b):
    num_nodes_a = len(graph_a.nodes)
    num_nodes_b = len(graph_b.nodes)
    similarity_matrix = np.zeros((num_nodes_a, num_nodes_b))
    # 遍历两个图中的节点，并计算节点之间的相似度
    for i, node_a in enumerate(graph_a.nodes):
        for j, node_b in enumerate(graph_b.nodes):
            # 获取节点A和B的邻居节点集合
            neighborhood_a = get_neighborhood(graph_a, node_a)
            neighborhood_b = get_neighborhood(graph_b, node_b)

            # 计算Jaccard相似度
            intersection = len(neighborhood_a.intersection(neighborhood_b))
            union = len(neighborhood_a.union(neighborhood_b))
            if union == 0:
                similarity = 0
            else:
                similarity = intersection / union

            similarity_matrix[i][j] = similarity
    return similarity_matrix


def compute_similarity_matrix(matrix1, matrix2):
    # 计算矩阵点积
    matrix1 = torch.mean(matrix1, dim = 2)
    matrix1 = torch.mean(matrix1, dim = 0).view(100,64)
    matrix2 = torch.mean(matrix2, dim = 2)
    matrix2 = torch.mean(matrix2, dim = 0).view(100,64)
    dot_product = torch.matmul(matrix1, torch.transpose(matrix2, 0, 1))

    # 计算矩阵范数
    norm1 = torch.norm(matrix1, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, dim=1, keepdim=True)

    # 计算相似度矩阵
    # similarity_matrix = dot_product / (norm1 * norm2.t())
    similarity_matrix = dot_product
    return similarity_matrix
def change_id(match, adj):
    adj_new = np.zeros((int(adj.shape[0]), int(adj.shape[1])),
                      dtype=np.float32)
    for row in range(adj.shape[0]):
        for col in range(adj.shape[1]):
            if adj[row, col] !=0:
                ## we need to change the position of the original graph and the target graph
                match = {int(i): idx for idx, i in enumerate(match)}
                adj_new[match[row], match[col]] = adj[row, col]
    return adj_new

# for i1 in [3]:
#     for i2 in [4, 7, 8]:
#         j1 = 0
#         j2 = 0
#         npz_path1 = 'PEMS0' + str(i1) + '/data_partitions_PEMS0' + str(i1) + '/part' + str(j1) + '/PEMS0' + str(
#             i1) + '_6048.npz'
#         npz_path2 = 'PEMS0' + str(i2) + '/data_partitions_PEMS0' + str(i2) + '/part' + str(j2) + '/PEMS0' + str(
#             i2) + '_6048.npz'
#         csv_path1 = 'PEMS0' + str(i1) + '/data_partitions_PEMS0' + str(i1) + '/part' + str(j1) + '/PEMS0' + str(
#             i1) + '.csv'
#         csv_path2 = 'PEMS0' + str(i2) + '/data_partitions_PEMS0' + str(i2) + '/part' + str(j2) + '/PEMS0' + str(
#             i2) + '.csv'
#         id_path1 = 'PEMS0' + str(i1) + '/data_partitions_PEMS0' + str(i1) + '/part' + str(j1) + '/sensor_ids_all.txt'
#         id_path2 = 'PEMS0' + str(i2) + '/data_partitions_PEMS0' + str(i2) + '/part' + str(j2) + '/sensor_ids_all.txt'
#         graph1 = load_graph_from_files(1, npz_path1, csv_path1, id_path1)
#         graph2 = load_graph_from_files(2, npz_path2, csv_path2, id_path2)
#         print("load two preprocessed graph sucessfully!")
#         if i2 == 4:
#             match_dict43 = GMDM(graph1, graph2)  ## this is the dict
#         elif i2 == 7:
#             match_dict47 = GMDM(graph1, graph2)  ## this is the dict
#         else:
#             match_dict48 = GMDM(graph1, graph2)  ## this is the dict


