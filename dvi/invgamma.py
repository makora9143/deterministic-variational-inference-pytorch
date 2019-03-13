from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import MultivariateNormal, Laplace
from torch.distributions.utils import broadcast_all


def _standard_gamma(concentration):
    return torch._standard_gamma(concentration)


class InverseGamma(ExponentialFamily):
    arg_constraints = {'concentration': constraints.positive,
                       'scale': constraints.positive,
                       'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.scale / (self.concentration - 1)

    @property
    def variance(self):
        return torch.pow(self.scale, 2) / torch.pow(self.concentration - 1, 2) / (self.constraints - 2)

    def __init__(self, concentration, scale=None, rate=None, validate_args=None):
        if rate is not None:
            scale = rate

        self.concentration, self.scale = broadcast_all(concentration, scale)
        if isinstance(concentration, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super(InverseGamma, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(InverseGamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = 1 / _standard_gamma(self.concentration.expand(shape)) * self.scale.expand(shape)
        value.detach().clamp_(min=torch.finfo(value.dtype).tiny)
        return value

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (self.concentration * torch.log(self.scale) -
                (self.concentration + 1) * torch.log(value) -
                self.scale / value - torch.lgamma(self.concentration))

    def entropy(self):
        return (self.concentration + torch.log(self.scale) + torch.lgamma(self.concentration) -
                (1 + self.concentration) * torch.digamma(self.concentration))

    @property
    def _natural_params(self):
        return (-self.concentration - 1, -self.scale)

    def _log_normalizer(self, x, y):
        return torch.lgamma(- x - 1) + (- x - 1) * torch.log(-y.reciprocal())
