from __future__ import division
from __future__ import print_function

import time
import torch
import pickle
import argparse
import datetime
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F

from utils import normalize, sparse_mx_to_torch_sparse_tensor,data_preprocess
from models import HAIN

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == "__main__":
    torch.manual_seed(5)
    np.random.seed(5)
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer',help='Dataset Name')
    args = parser.parse_args()
    dropout=0.0
    if args.dataset == "cora":
        learning_rate = 0.03
        epochs = 200
        nhid1 = 512
    elif args.dataset == "citeseer":
        learning_rate = 0.03
        epochs = 50
        nhid1 = 512
    elif args.dataset == "pubmed":
        learning_rate = 0.05
        epochs = 200
        nhid1 = 256
    elif args.dataset == "dblp":
        learning_rate = 0.03
        epochs = 100
        nhid1 = 256
    print("Dataset :",args.dataset)
    line_content, contentset, hyper_incidence_matrix, adj_line, train_node, test_node, val_node, labelset, label, classes,splits = data_preprocess(args.dataset)

    features = sp.csr_matrix(line_content, dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    hyper_incidence_tensor = torch.FloatTensor(hyper_incidence_matrix)
    adj_line = sp.csr_matrix(adj_line, dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj_line)
    idx_train = torch.LongTensor(train_node)
    idx_test = torch.LongTensor(test_node)
    idx_val = torch.LongTensor(val_node)
    labels = torch.LongTensor(np.where(labelset)[1])
    model = HAIN(nfeat=contentset.shape[1],
                nhid1=nhid1,
                nclass=len(classes),
                dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    # Model Training

    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output= model(features, adj, hyper_incidence_matrix, hyper_incidence_tensor)
        loss_train = F.nll_loss(output[train_node], labels[train_node])
        acc_train = accuracy(output[train_node], labels[train_node])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              "time=", "{:.5f}".format(time.time() - t))
    # Test
    model.eval()
    output = model(features, adj, hyper_incidence_matrix, hyper_incidence_tensor)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test]) * 100
    acc = round(acc_test.item(), 2)
    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))
