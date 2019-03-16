import math
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as tdist

from .invgamma import InverseGamma
from .kl import kl_normal_invgamma, kl_normal_laplace

class GaussianVar(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.shape = mean.shape


class Parameter(nn.Module):
    def __init__(self, prior, approximation, q_loc, log_q_scale):
        super(Parameter, self).__init__()

        self.prior = prior
        self.approximation = approximation

        self.q_loc = q_loc
        self.log_q_scale = log_q_scale

    def q(self):
        return self.approximation(loc=self.q_loc, scale=torch.exp(self.log_q_scale))

    def __repr__(self):
        args_string = 'Prior: {}\n  Variational: {}'.format(
            self.prior,
            self.q()
        )
        return self.__class__.__name__ + '(\n  ' + args_string + '\n)'

    def surprise(self):
        q = self.q()
        p = self.prior
        return tdist.kl_divergence(q, p)

    def sample(self, n_sample=None, average=False):
        if n_sample is None:
            n_sample = 1
        samples = self.q().rsample((n_sample,))
        return samples


def get_variance_scale(initialization_type, shape):
    if initialization_type == "standard":
        prior_var = 1.0
    elif initialization_type == "wide":
        prior_var = 100.0
    elif initialization_type == "narrow":
        prior_var = 0.01
    elif initialization_type == "glorot":
        prior_var = (2.0 / (shape[-1] + shape[-2]))
    elif initialization_type == "xavier":
        prior_var = 1.0/shape[-1]
    elif initialization_type == "he":
        prior_var = 2.0/shape[-1]
    elif initialization_type == "wider_he":
        prior_var = 5.0/shape[-1]
    else:
        raise NotImplementedError('prior type "%s" not recognized' % initialization_type)
    return prior_var


def gaussian_init(loc, scale, shape):
    return loc + scale * torch.randn(*shape)


def laplace_init(loc, scale, shape):
    return torch.from_numpy(np.random.laplace(loc, scale/np.sqrt(2.0), size=shape).astype(np.float32))


def make_weight_matrix(shape, prior_type, variance):
    """
    Args:
        shape (list, required): The shape of weight matrix. It should be `(out_features, in_features)`.
        prior_type (list, required): Prior Type. It should be `[prior, weight_scale, bias_scale]`
        `["gaussian", "wider_he", "wider_he"]`.
    """
    variance = get_variance_scale(variance.strip().lower(), shape)
    stddev = torch.sqrt(torch.ones(shape) * variance).float()
    log_stddev = nn.Parameter(torch.log(stddev))
    stddev = torch.exp(log_stddev)

    prior = prior_type.strip().lower()

    if prior == 'empirical':
        a = 4.4798
        alpha = a
        beta = (1 + a) * variance

        prior = InverseGamma(alpha, beta)

        mean = nn.Parameter(torch.Tensor(*shape))
        nn.init.normal_(mean, 0.0, math.sqrt(variance))
        return Parameter(prior, tdist.Normal, mean, log_stddev)

    elif prior == 'gaussian' or prior == 'normal':
        init_function = gaussian_init
        prior_generator = tdist.Normal
    elif prior == 'laplace':
        init_function = laplace_init
        prior_generator = tdist.Laplace
    else:
        raise NotImplementedError('prior type "{}" not recognized'.format(prior))

    mean = nn.Parameter(init_function(0.0, math.sqrt(variance), shape))

    prior_loc = torch.zeros(*shape)
    prior_scale = torch.ones(*shape) * math.sqrt(variance)
    prior = prior_generator(prior_loc, prior_scale)

    return Parameter(prior, tdist.Normal, mean, log_stddev)


def make_bias_vector(shape, prior_type, variance):
    fudge_factor = 10.0
    variance = get_variance_scale(variance.strip().lower(), shape)
    stddev = torch.sqrt(torch.ones(shape[-2],) * variance / fudge_factor)
    log_stddev = nn.Parameter(torch.log(stddev))
    stddev = torch.exp(log_stddev)

    prior = prior_type.strip().lower()

    if prior == 'empirical':
        a = 4.4798
        alpha = a
        beta = (1 + a) * variance

        prior = InverseGamma(alpha, beta)

        mean = nn.Parameter(torch.zeros(shape[-2],))
        return Parameter(prior, tdist.Normal, mean, log_stddev)
    else:
        raise NotImplementedError('prior type "{}" not recognized'.format(prior))
