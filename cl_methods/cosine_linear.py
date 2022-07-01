# From Official LUCIR Code
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from cl_methods.cl_utils import stable_cosine_distance

def _reduce_proxies(similarities, num_proxy):
    # shape (batch_size, n_classes * proxy_per_class)
    n_classes = similarities.shape[1] / num_proxy
    assert n_classes.is_integer(), (similarities.shape[1], num_proxy)
    n_classes = int(n_classes)
    bs = similarities.shape[0]

    simi_per_class = similarities.view(bs, n_classes, num_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)
    return (simi_per_class * attentions).sum(-1)

class CosineLinear(Module):
    def __init__(self, in_features, out_features, num_proxy=1, sigma_learnable=True, sigma=1.0,
            eta_learnable=True, eta=1.0, version='cc', nca_margin=0.6, is_train=True):
        super(CosineLinear, self).__init__()
        self.version = version
        self.num_proxy = num_proxy
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.num_proxy * out_features, in_features))

        self.sigma_learnable = sigma_learnable
        self.sigma = sigma
        if self.sigma_learnable:
            self.sigma = Parameter(torch.ones(1))

        self.eta_learnable = eta_learnable
        self.eta = eta

        if self.eta_learnable:
            self.eta = Parameter(torch.ones(1))
        self.nca_margin = nca_margin
        self.reset_parameters()
        self.is_train = is_train

    def reset_parameters(self):
        if self.version == 'lsc':
            nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        if isinstance(self.sigma, Parameter):
            self.sigma.data.fill_(1) #for initializaiton of sigma
        if self.eta and isinstance(self.eta, Parameter):
            self.eta.data.fill_(1)

    def forward(self, input, sigma_from_wrapper=None):
        if self.version == 'cc':
            out = F.linear(F.normalize(input, p=2,dim=1), \
                    F.normalize(self.weight, p=2, dim=1))
            out = self.sigma * out
        elif self.version == 'lsc':
            if sigma_from_wrapper is None:
                features = self.sigma * F.normalize(input,p=2,dim=1)
                weights = self.sigma * F.normalize(self.weight,p=2,dim=1)
                out = - stable_cosine_distance(features, weights)
                out = _reduce_proxies(out, self.num_proxy)
                if self.is_train:
                    out = self.eta * (out - self.nca_margin)
            else:
                features = sigma_from_wrapper * F.normalize(input,p=2,dim=1)
                weights = sigma_from_wrapper * F.normalize(self.weight,p=2,dim=1)
                out = - stable_cosine_distance(features, weights)
                out = _reduce_proxies(out, self.num_proxy)
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, sigma={}, eta={}'.format(
            self.in_features, self.num_proxy*self.out_features,
            self.sigma.data if self.sigma_learnable else self.sigma,
            self.eta.data if self.eta_learnable else self.eta
        )

class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, num_proxy=1, sigma_learnable=True, sigma=1.0,
            eta_learnable=False, eta=1.0, version='cc', nca_margin=0.6, is_train=True):
        super(SplitCosineLinear, self).__init__()
        self.version = version
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.num_proxy = num_proxy
        self.fc1 = CosineLinear(in_features, out_features1, self.num_proxy,
                False, 1.0, False, 1.0, version=self.version)
        self.fc2 = CosineLinear(in_features, out_features2, self.num_proxy,
                False, 1.0, False, 1.0, version=self.version)
        self.sigma_learnable = sigma_learnable
        self.sigma = sigma
        self.eta_learnable = eta_learnable
        self.eta = eta
        if self.sigma_learnable:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)

        if self.eta_learnable:
            self.eta = Parameter(torch.Tensor(1))

        self.nca_margin = nca_margin
        self.is_train = is_train

    def forward(self, x):
        if self.version == 'cc':
            out1 = self.fc1(x)
            out2 = self.fc2(x)
            out = torch.cat((out1, out2), dim=1) #concatenate along the channel
            if self.sigma is not None:
                out = self.sigma * out
        elif self.version == 'lsc': # for pod (nca_loss)...
            out1 = self.fc1(x, self.sigma)
            out2 = self.fc2(x, self.sigma)
            out = torch.cat((out1,out2),dim=1)
            if self.is_train:
                out = self.eta * (out - self.nca_margin)

        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, sigma={}, eta={}'.format(
            self.in_features, self.num_proxy * self.out_features,
            self.sigma.data if self.sigma_learnable else self.sigma,
            self.eta.data if self.eta_learnable else self.eta
        )

