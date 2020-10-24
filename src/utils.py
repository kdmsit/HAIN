import torch
import pickle
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def get_one_hot(labelset,classes):
    onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labelset = np.array(list(map(onehot.get, labelset)), dtype=np.int32)
    return labelset

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def data_preprocess(dataset):
    # File Paths
    split_path = '../splits/' + dataset + '_splits.pickle'
    if dataset != 'dblp':
        #For cora/citeseer/pubmed dataset
        edgelist_path='../data/'+dataset+'.edgelist'
        label_path='../data/'+dataset+'_label.csv'
        content_path='../data/'+dataset+'_content.csv'

        # Read Label File of the Graph and Generate One hot label
        labelset = np.genfromtxt(label_path, dtype=np.dtype(int))

        # Read Content File of the Graph
        contentset = pd.read_csv(content_path, header=None).values

        # Read Edgelist File for cited and citing paper ids
        f = open(edgelist_path, "r")
        citing_cited_paper_list = []
        citing_paper_list = set()
        edge_count = 0
        for line in f:
            edge_count = edge_count + 1
            cited_paper_id = int(line.split(' ')[0].strip())  # \t
            citing_paper_id = int(line.split(' ')[1].strip())
            citing_cited_paper_list.append([int(cited_paper_id), int(citing_paper_id)])
            citing_paper_list.add(citing_paper_id)
        f.close()
        '''
        Create Hyper-Edgelist from Graph Edge-list.Take all cited paper id for one corresponding paper in one hyper-edge.
        Also Create Edge Feature Matrix(or Line Node Feature Matrix) simultaneously.
        '''
        hyper_edge_list = []
        for citing_paper_id in citing_paper_list:
            hyper_edge = [citing_paper_id]
            for paper in citing_cited_paper_list:
                if (citing_paper_id == paper[1]):
                    hyper_edge.append(paper[0])
            hyper_edge_list.append(hyper_edge)
    else:
        # For dblp dataset
        edgelist_path = '../data/dblp_hypergraph.pickle'
        label_path = '../data/dblp_labels.pickle'
        content_path = '../data/dblp_content.pickle'
        with open(edgelist_path, 'rb') as handle:
            hyper_graph = pickle.load(handle)
        with open(label_path, 'rb') as handle:
            labelset = pickle.load(handle)
        with open(content_path, 'rb') as handle:
            contentset = pickle.load(handle).todense()
        hyper_edge_list = list(hyper_graph.values())

    classes = set(labelset)
    label = labelset
    labelset = get_one_hot(labelset, classes)


    # Generate Nodeset
    no_of_nodes = np.shape(labelset)[0]
    nodeset = [node for node in range(no_of_nodes)]

    with open(split_path, 'rb') as handle:
        splits = pickle.load(handle)
    train_node = splits["train_node"]
    train_y = splits["train_y"]
    val_node = splits["val_node"]
    val_y = splits["val_y"]
    test_node = splits["test_node"]
    test_y = splits["test_y"]
    splits = {"train_node": train_node, "train_y": train_y, "val_node": val_node, "val_y": val_y,
              "test_node": test_node, "test_y": test_y}


    # Build Incedence Matrix(H) of Hyper Graph
    no_of_hyper_edges=len(hyper_edge_list)
    hyper_incidence_matrix=np.zeros(shape=(no_of_nodes,no_of_hyper_edges),dtype=int)
    for i in range(len(hyper_edge_list)):
      for node in hyper_edge_list[i]:
        hyper_incidence_matrix[node][i]=1

    #Line Graph Feature File
    line_content=np.matmul(np.transpose(hyper_incidence_matrix),contentset)

    # Build Edge Degree Matrix(De) and Node Degree Matrix(Dn)
    edge_degree_matrix=np.diag(np.sum(hyper_incidence_matrix,axis=0))
    node_degree_matrix=np.diag(np.sum(hyper_incidence_matrix,axis=1))

    # Build Adjacency Matrix for Line Graph
    adj_line=np.matmul(np.matmul(np.matmul(np.linalg.inv(edge_degree_matrix),np.transpose(hyper_incidence_matrix)),np.linalg.inv(node_degree_matrix)),hyper_incidence_matrix)

    return line_content,contentset,hyper_incidence_matrix,adj_line,train_node,test_node,val_node,labelset,label,classes,splits
