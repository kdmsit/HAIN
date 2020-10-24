import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import HypergraphAttentionIsomorphism


class HAIN(nn.Module):
    def __init__(self, nfeat, nhid1,nclass, dropout):
        super(HAIN, self).__init__()
        self.gc1 = HypergraphAttentionIsomorphism(nfeat, nhid1)
        self.gc3 = HypergraphAttentionIsomorphism(nhid1, nclass)
        self.dropout = dropout

    def forward(self, x, adj,incidence_matrix,incidence_tensor):
        x = F.relu(self.gc1(x, adj,incidence_tensor))
        a=x.detach().numpy()
        x=np.matmul(np.transpose(incidence_matrix),a)
        x = self.gc3(torch.FloatTensor(x), adj,incidence_tensor)
        return F.log_softmax(x, dim=1)
