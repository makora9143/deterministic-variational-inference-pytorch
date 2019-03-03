import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist

import distributions as bdist
from kl import kl_normal_invgamma


def gaussian_init(mean, sigma, shape):
    param = nn.Parameter(torch.Tensor(*shape))
    param.data.normal_(mean, sigma)
    return param


def constant_init(value, shape):
    param = nn.Parameter(torch.Tensor(*shape))
    param.data.fill_(value)
    return param


def laplace_init(mean, sigma, shape):
    return torch.tensor(np.random.laplace(mean, sigma / np.sqrt(2.0), size=shape)).float()


class GaussianVar(object):
    def __init__(self, mean, var, shape=None):
        self.mean = mean
        self.var = var
        self.shape = mean.shape if shape is None else shape


class Parameter(nn.Module):
    def __init__(self, variational_mean, variational_var, prior, variational_distribution):
        super(Parameter, self).__init__()
        self.variational_mean = variational_mean
        self.variational_var = variational_var
        self.prior = prior
        self.variational_distribution = variational_distribution

    def __repr__(self):
        args_string = 'Variational Distribution: {}(loc={}, scale={})\n\tPrior: {}'.format(
            self.variational_distribution.__name__,
            self.variational_mean if self.variational_mean.dim() == 0 else self.variational_mean.size(),
            self.variational_var if self.variational_var.dim() == 0 else self.variational_var.size(),
            self.prior
        )
        return self.__class__.__name__ + '=>\n\t' + args_string

    def surprise(self):
        q = self.variational_distribution(self.variational_mean, torch.sqrt(self.variational_var))
        p = self.prior
        return tdist.kl_divergence(q, p)


def make_weight_matrix(shape, prior_type):
    # scalar
    s2 = get_variance_scale(prior_type[1].strip().lower(), shape)
    # torch tensor
    log_sigma = constant_init(math.log(math.sqrt(s2)), shape)
    sigma = torch.exp(log_sigma)

    if prior_type[0].strip().lower() == 'gaussian':
        init_function = gaussian_init
        # prior_generator = bdist.DiagonalNormal
        prior_generator = tdist.Normal
    elif prior_type[0].strip().lower() == 'laplace':
        init_function = laplace_init
        # prior_generator = bdist.DiagonalLaplace
        prior_generator = tdist.Laplace
    elif prior_type[0].strip().lower() == 'empirical':
        # mainly here
        a = 4.4798
        alpha = torch.tensor(a).float()
        beta = torch.tensor((1 + a)* s2).float() 

        mean = gaussian_init(0.0, math.sqrt(s2), shape)
        prior = bdist.InverseGamma(alpha, beta)
        # return Parameter(mean, sigma * sigma, prior, bdist.DiagonalNormal)
        return Parameter(mean, sigma * sigma, prior, tdist.Normal)
    else:
        raise NotImplementedError('prior type {} not recognized'.format(prior_type[0]))

    mean = init_function(0.0, math.sqrt(s2), shape)

    prior_loc = torch.zeros(shape).float()
    prior_scale = torch.ones(shape).float() * math.sqrt(s2)
    prior = prior_generator(prior_loc, prior_scale)

    return Parameter(mean, sigma * sigma, prior, tdist.Normal)


def make_bias_vector(shape, prior_type):
    fudge_factor = 10.0
    s2 = get_variance_scale(prior_type[2].strip().lower(), shape)
    log_sigma = constant_init(math.log(math.sqrt(s2 / fudge_factor)), (shape[-2],))
    sigma = torch.exp(log_sigma)

    if prior_type[0].strip().lower() == 'gaussian':
        prior_generator = tdist.Normal
    elif prior_type[0].strip().lower() == 'laplace':
        prior_generator = tdist.Laplace
    elif prior_type[0].strip().lower() == 'empirical':
        a = 4.4798
        alpha = torch.tensor(a)
        beta = torch.tensor((a + 1.0) * s2)

        mean = constant_init(0.0, (shape[-2],))
        prior = bdist.InverseGamma(alpha, beta)

        return Parameter(mean, sigma * sigma, prior, tdist.Normal)
    else:
        raise NotImplementedError('bias prior type {} not recognized'.format(prior_type[0]))

    mean = constant_init(0.0, (shape[-2],))

    prior_loc = torch.zeros((shape[-2],))
    prior_scale = torch.ones((shape[-2],)) * math.sqrt(s2)
    prior = prior_generator(prior_loc, prior_scale)

    return Parameter(mean, sigma * sigma, prior, tdist.Normal)


def get_variance_scale(initialization_type, shape):
    if initialization_type == 'standard':
        prior_var = 1.0
    elif initialization_type == 'wide':
        prior_var = 100.0
    elif initialization_type == 'narrow':
        prior_var = 0.01
    elif initialization_type == 'glorot':
        prior_var = (2.0 / (shape[-1] + shape[-2]))
    elif initialization_type == 'xavier':
        prior_var = 1.0 / shape[-1]
    elif initialization_type == 'he':
        prior_var = 2.0 / shape[-1]
    elif initialization_type == 'wider_he':
        prior_var = 5.0 / shape[-1]
    else:
        raise NotImplementedError('prior type {} not recognized'.format(initialization_type))
    return prior_var

