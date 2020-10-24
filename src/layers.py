import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class HypergraphAttentionIsomorphism(Module):
    def __init__(self, in_features, out_features):
        super(HypergraphAttentionIsomorphism, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.attn = Parameter(torch.FloatTensor(in_features, 1))
        self.alpha = Parameter(torch.rand(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.attn.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, incidence_matrix):
        # with Attention
        support1=F.softmax(torch.mm(input, self.attn),dim=0)
        support2=torch.diag(support1.view(support1.size()[0]))
        support3=torch.spmm(adj,torch.mm(support2,input))
        support4=torch.add(support3,torch.mul(self.alpha,input))
        support = torch.mm(support4, self.weight)
        output=torch.mm(incidence_matrix,support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'