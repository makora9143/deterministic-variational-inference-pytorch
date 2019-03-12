import torch
import torch.nn as nn

from bayes_layers import LinearCertainActivations, LinearReLU
from variables import GaussianVar

class MLP(nn.Module):
    def __init__(self, x_dim, y_dim, prior_type, hidden_size=None):
        super(MLP, self).__init__()

        self.sizes = [x_dim]
        if hidden_size is not None:
            self.sizes += hidden_size
        self.sizes += [y_dim]
        self.prior_type = prior_type
        self.make_layers(True)


    def make_layers(self, bias=True):
        layers = [LinearCertainActivations(self.sizes[0], self.sizes[1], self.prior_type, bias)]
        for in_dim, out_dim in zip(self.sizes[1:-1], self.sizes[2:]):
            layers.append(LinearReLU(in_dim, out_dim, self.prior_type, bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

    def surprise(self):
        all_surprise = 0
        for layer in self.layers:
            all_surprise += layer.surprise()
        return all_surprise


class AdaptedMLP(object):
    def __init__(self, mlp, adapter):
        self.mlp = mlp
        self.__dict__.update(mlp.__dict__)
        self.make_adapters(adapter)

    def make_adapters(self, adapter):
        self.adapter = {}
        for ad in ['in', 'out']:
            self.adapter[ad] = {
                'scale': torch.tensor(adapter[ad]['scale']),
                'shift': torch.tensor(adapter[ad]['shift'])
            }

    def __call__(self, input):
        x_ad = self.adapter['in']['scale'] * input + self.adapter['in']['shift']
        self.pre_adapt = self.mlp(x_ad)
        mean = self.adapter['out']['scale'] * self.pre_adapt.mean + self.adapter['out']['shift']
        cov = self.adapter['out']['scale'].reshape(-1, 1) * self.adapter['out']['scale'].reshape(1, -1) * self.pre_adapt.var
        return GaussianVar(mean, cov)

    def __repr__(self):
        return "AdaptedMLP(\n" + self.mlp.__repr__() + ")"

    def surprise(self):
        return self.mlp.surprise()

    def parameters(self):
        return self.mlp.parameters()
