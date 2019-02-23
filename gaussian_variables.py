import math

import numpy as np
import torch
import torch.nn as nn

import bayes_utils as bu


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


class DiagonalGaussianVar(nn.Module):
    def __init__(self, mean, var, shape=None):
        super(DiagonalGaussianVar, self).__init__()
        self.mean = mean
        self.var = var
        self.shape = mean.shape if shape is None else shape
        # super(DiagonalGaussianVar, self).register_parameter('mean', self.mean)
        # super(DiagonalGaussianVar, self).register_parameter('var', self.var)

    def sample(self, n_sample=None):
        no_sample_dim = False
        if n_sample is None:
            no_sample_dim = True
            n_sample = 1
        s = torch.randn() * torch.sqrt(self.var) + self.mean
        if no_sample_dim:
            return s[0, ...]
        else:
            return s


class DiagonalLaplaceVar(object):
    def __init__(self, mean, var, shape):
        self.mean = mean
        self.var = var
        self.b = torch.sqrt(var / 2)
        self.shape = mean.shape if shape is None else shape


class InverseGammaVar(object):
    def __init__(self, alpha, beta, shape=None):
        self.mean = beta / (alpha - 1.0)
        self.var = self.mean * self.mean / (alpha - 2.0)
        self.alpha = alpha
        self.beta = beta
        self.shape = self.mean.shape if shape is None else shape


class Parameter(nn.Module):
    def __init__(self, value, prior, variables=None):
        super(Parameter, self).__init__()
        self.value = value
        self.prior = prior
        self.variables = variables


def make_weight_matrix(shape, prior_type):
    # scalar
    s2 = get_variance_scale(prior_type[1].strip().lower(), shape)
    # torch tensor
    log_sigma = constant_init(math.log(math.sqrt(s2)), shape)
    sigma = torch.exp(log_sigma)

    if prior_type[0].strip().lower() == 'gaussian':
        init_function = gaussian_init
        prior_generator = DiagonalGaussianVar
    elif prior_type[0].strip().lower() == 'laplace':
        init_function = laplace_init
        prior_generator = DiagonalLaplaceVar
    elif prior_type[0].strip().lower() == 'empirical':
        # mainly here
        a = 4.4798
        alpha = torch.tensor(a).float()
        beta = torch.tensor(1 + a).float() * s2

        mean = gaussian_init(0.0, math.sqrt(s2), shape)
        value = DiagonalGaussianVar(mean, sigma * sigma, shape)
        prior = InverseGammaVar(alpha, beta)
        return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})
    else:
        raise NotImplementedError('prior type {} not recognized'.format(prior_type[0]))

    # mean = torch.tensor(init_function(0.0, math.sqrt(s2), shape))
    mean = init_function(0.0, math.sqrt(s2), shape)
    value = DiagonalGaussianVar(mean, sigma * sigma, shape)

    # ここもnn.Parameterにしないといけないはず
    prior_mean = torch.zeros(shape).float()
    prior_var = torch.ones(shape).float() * s2
    prior = prior_generator(prior_mean, prior_var, shape)

    return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})


def make_bias_vector(shape, prior_type):
    fudge_factor = 10.0
    s2 = get_variance_scale(prior_type[2].strip().lower(), shape)
    log_sigma = constant_init(math.log(math.sqrt(s2 / fudge_factor)), (shape[-2],))
    sigma = torch.exp(log_sigma)

    if prior_type[0].strip().lower() == 'gaussian':
        prior_generator = DiagonalGaussianVar
    elif prior_type[0].strip().lower() == 'laplace':
        prior_generator = DiagonalLaplaceVar
    elif prior_type[0].strip().lower() == 'empirical':
        a = 4.4798
        alpha = torch.tensor(a)
        beta = torch.tensor((a + 1.0) * s2)

        mean = constant_init(0.0, (shape[-2],))
        value = DiagonalGaussianVar(mean, sigma * sigma, (shape[-2],))
        prior = InverseGammaVar(alpha, beta)

        return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})
    else:
        raise NotImplementedError('bias prior type {} not recognized'.format(prior_type[0]))

    mean = torch.zeros((shape[-1],)).float()
    mean = constant_init(0.0, (shape[-1],))
    value = DiagonalGaussianVar(mean, sigma * sigma, (shape[-1],))

    prior_mean = torch.zeros((shape[-1],))
    prior_var = torch.ones((shape[-1],)) * s2
    prior = prior_generator(prior_mean, prior_var, (shape[-1],))

    return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})


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


def KL(p, q):
    if isinstance(p, DiagonalGaussianVar):
        if isinstance(q, DiagonalGaussianVar):
            safe_qvar = q.var + bu.EPSILON
            entropy_term = 0.5 * (1 + bu.log2pi + torch.log(p.var))
            cross_entropy_term = 0.5 * (bu.log2pi + torch.log(safe_qvar) + (p.var + (p.mean - q.mean)**2) / safe_qvar)
            return torch.sum(cross_entropy_term - entropy_term)
        elif isinstance(q, DiagonalLaplaceVar):
            sigma = torch.sqrt(p.var)
            mu_ovr_sigma = p.mean / sigma
            tmp = 2 * bu.standard_gaussian(mu_ovr_sigma) + mu_ovr_sigma * torch.erf(mu_ovr_sigma * bu.one_ovr_sqrt2)
            tmp *= sigma / q.b
            tmp += 0.5 * torch.log(2 * q.b * q.b / (math.pi * p.var)) - 0.5
            return torch.sum(tmp)
        elif isinstance(q, InverseGammaVar):
            return EBKL(p, q)
    print('unsupported KL')


def EBKL(p, q, hypers=None, global_step=1.0E99):
    if isinstance(p, DiagonalGaussianVar):
        if isinstance(q, InverseGammaVar):
            m = torch.prod(p.mean.shape).float()
            S = torch.sum(p.var + p.mean * p.mean)
            m_plus_2alpha_plus_2 = m + 2.0 * q.alpha + 2.0
            S_plus_2beta = S + 2.0 * q.beta

            tmp = m * torch.log(S_plus_2beta / m_plus_2alpha_plus_2)
            tmp += S * (m_plus_2alpha_plus_2 / S_plus_2beta)
            tmp += -(m + torch.sum(torch.log(p.var)))
            return 0.5 * tmp
    print('unsupported KL')
