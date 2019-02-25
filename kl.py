import torch
import torch.distributions as tdist
from torch.distributions.kl import register_kl
from distributions import InverseGamma


@register_kl(tdist.Normal, InverseGamma)
def kl_normal_invgamma(p, q):
    # p: loc/scale, q: concentration, scale
    m = torch.numel(p.loc)
    S = torch.sum(p.scale.pow(2) + p.loc.pow(2))
    m_plus_2alpha_plus_2 = m + 2.0 * q.concentration + 2.0
    S_plus_2beta = S + 2.0 * q.scale

    tmp = m * torch.log(S_plus_2beta / m_plus_2alpha_plus_2)
    tmp += S * (m_plus_2alpha_plus_2 / S_plus_2beta)
    tmp += -(m + torch.sum(torch.log(p.scale.pow(2))))
    return 0.5 * tmp
