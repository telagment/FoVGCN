import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import Parameter
import math
from torchvision import models
from datasets.nud_gl_multicases import preprocessing


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            init.constant_(self.bias.data, 0.1)

    def forward(self, input, adj):
        # print('input_support', input.shape, adj.shape, self.weight.shape)
        support = torch.matmul(input, self.weight)
        # print('adj', adj.shape, adj.dtype)
        # print('adj  , support ', adj.shape, support.shape)

        output = torch.matmul(adj, support)
        # print('output', output.shape)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()

        self.gc1 = GraphConvolution(1440, 720)
        self.bn1 = nn.BatchNorm1d(360, eps=1e-05, momentum=0.1, affine=True)
        
        self.gc2 = GraphConvolution(720, 360)
        self.bn2 = nn.BatchNorm1d(360, eps=1e-05, momentum=0.1, affine=True)
        
        self.gc3 = GraphConvolution(360, 180)
        self.bn3 = nn.BatchNorm1d(360, eps=1e-05, momentum=0.1, affine=True)
        
        self.gc4 = GraphConvolution(180, 90)
        self.bn4 = nn.BatchNorm1d(360, eps=1e-05, momentum=0.1, affine=True)
        
        self.gc5 = GraphConvolution(90, 45)
        self.bn5 = nn.BatchNorm1d(360, eps=1e-05, momentum=0.1, affine=True)
        
        self.gc6 = GraphConvolution(45, 1)
        self.relu = nn.Softplus()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

    def norm_adj(self, matrix):
        D = torch.diag_embed(matrix.sum(2))
        D = D ** 0.5
        D = D.inverse()
        normal = D.bmm(matrix).bmm(D) #batch matrix mutiplication
        # print(' Caculating D !')
        return normal.detach()

    def forward(self, feature, A):
        adj = self.norm_adj(A)
        gc1 = self.gc1(feature, adj)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)

        gc2 = self.gc2(gc1, adj)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)

        gc3 = self.gc3(gc2, adj)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)

        gc4 = self.gc4(gc3, adj)
        gc4 = self.bn4(gc4)
        gc4 = self.relu(gc4)

        gc5 = self.gc5(gc4, adj)
        gc5 = self.bn5(gc5)
        gc5 = self.relu(gc5)

        gc6 = self.gc6(gc5, adj)
        gc6 = self.relu(gc6)
        
        return gc6

class FovGCN(nn.Module):
    def __init__(self):
        super(FovGCN, self).__init__()

        self.GCN = GCNNet()
        self.fc = nn.Linear(360, 1)

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def loss_build(self, x_hat, x):
        distortion = F.mse_loss(x_hat, x, size_average=True, reduction='mean')
        return distortion

    def rmse(self, x_hat, x):
        distortion = F.mse_loss(x_hat, x, size_average=True, reduction='mean')
        loss = torch.sqrt(distortion)
        return loss

    def forward(self, x, label, A, requires_loss):
        batch_size = x.size(0)
        # print('A_before', A.shape)

        all_feature = x
        feature = preprocessing(all_feature)
        # print('feature', feature.shape)

        gc6 = self.GCN(feature, A)
        fc_in = gc6.view(gc6.size()[0], -1)
        score = torch.mean(fc_in, dim=1).unsqueeze(1)

        if requires_loss:
            return score, label, self.loss_build(score, label), self.rmse(score, label)
        else:
            return score, label