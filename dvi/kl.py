import math
import torch
import torch.distributions as tdist
from torch.distributions.kl import register_kl

from .invgamma import InverseGamma

one_ovr_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
one_ovr_sqrt2 = 1.0 / math.sqrt(2.)


def _standard_gaussian(x):
    return one_ovr_sqrt2pi * torch.exp(- x * x / 2.0)


@register_kl(tdist.Normal, InverseGamma)
def kl_normal_invgamma(p, q):
    # p: loc/scale, q: concentration, scale
    m = torch.numel(p.loc)
    S = p.scale.pow(2) + p.loc.pow(2)
    m_plus_2alpha_plus_2 = m + 2.0 * q.concentration + 2.0
    S_plus_2beta = S + 2.0 * q.scale / m

    term1 = torch.log(torch.sum(S_plus_2beta) / m_plus_2alpha_plus_2)
    term2 = S * (m_plus_2alpha_plus_2 / torch.sum(S_plus_2beta))
    term3 = -(1 + torch.log(p.scale.pow(2)))
    return 0.5 * (term1 + term2 + term3)


@register_kl(tdist.Normal, tdist.Laplace)
def kl_normal_laplace(p, q):
    sigma = p.scale
    mu_ovr_sigma = p.loc / sigma
    tmp = 2 * _standard_gaussian(mu_ovr_sigma) + mu_ovr_sigma * torch.erf(mu_ovr_sigma * one_ovr_sqrt2)
    tmp *= sigma / q.scale
    tmp += 0.5 * torch.log(2 * q.scale * q.scale / (math.pi * p.scale)) - 0.5
    return tmp
