import torch
import torch.nn as nn
import bayes_layers as bnn
import gaussian_variables as gv


class MLP(nn.Module):
    def __init__(self, input_size, output_size, prior_type, hidden_dims=None, nonlinear='relu'):
        super(MLP, self).__init__()
        self.sizes = [input_size]
        if hidden_dims is not None:
            self.sizes += hidden_dims
        self.sizes += [output_size]
        self.prior_type = prior_type
        self.layer_factory = self.get_layer_factory(nonlinear)
        self.make()

    def get_layer_factory(self, nonlinearity):
        nonlinearity = nonlinearity.strip().lower()
        if nonlinearity == 'relu':
            return bnn.LinearReLU
        elif nonlinearity == 'heaviside':
            return bnn.LinearHeaviside
        else:
            raise NotImplementedError("Not Implemented.")

    def make(self):
        layers = [bnn.LinearCertainActivations(self.sizes[0], self.sizes[1], self.prior_type)]

        for input_dim, output_dim in zip(self.sizes[1:-1], self.sizes[2:]):
            layers.append(self.layer_factory(input_dim, output_dim, self.prior_type))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def surprise(self):
        total = 0
        for layer in self.layers:
            total += layer.surprise()
        return total


class PointMLP(MLP):
    def forward(self, x):
        self.h = [x]
        for l in self.layers:
            self.h.append()


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

    def __call__(self, x):
        x_ad = self.adapter['in']['scale'] * x + self.adapter['in']['shift']
        self.pre_adapt = self.mlp(x_ad)
        mean = self.adapter['out']['scale'] * self.pre_adapt.mean + self.adapter['out']['shift']
        cov = self.adapter['out']['scale'].reshape(-1, 1) * self.adapter['out']['scale'].reshape(1, -1) * self.pre_adapt.var
        return gv.GaussianVar(mean, cov)
