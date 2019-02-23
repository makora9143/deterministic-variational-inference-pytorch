import math
import torch


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


twopi = torch.tensor(2.0 * math.pi).float()
one_ovr_sqrt2 = torch.tensor(1.0 / math.sqrt(2)).float()
one_ovr_sqrt2pi = torch.tensor(1.0 / math.sqrt(2) * math.pi).float()
log2pi = torch.tensor(math.log(2.0 * math.pi))


EPSILON = 1e-6
HALF_EPSILON = EPSILON / 2.0


def standard_gaussian(x):
    return one_ovr_sqrt2pi * torch.exp(-x * x / 2.0)


def gaussian_cdf(x):
    return 0.5 * (1.0 + torch.erf(x * one_ovr_sqrt2))


def softrelu(x):
    return standard_gaussian(x) + x * gaussian_cdf(x)


def delta(rho, mu1, mu2):
    return gaussian_cdf(mu1) * gaussian_cdf(mu2) + g(rho, mu1, mu2)


def g(rho, mu1, mu2):
    one_plus_sqrt_one_minus_rho_sqr = 1.0 + torch.sqrt(1.0 - rho * rho)
    a = torch.asin(rho) - rho / one_plus_sqrt_one_minus_rho_sqr
    safe_a = torch.abs(a) + HALF_EPSILON
    safe_rho = torch.abs(rho) + EPSILON

    A = a / twopi
    sxx = safe_a * one_plus_sqrt_one_minus_rho_sqr / safe_rho
    one_ovr_sxy = (torch.asin(rho) - rho) / (safe_a * safe_rho)

    return A * torch.exp(-(mu1 * mu1 + mu2 * mu2) / (2.0 * sxx) + one_ovr_sxy * mu1 * mu2)


def matrix_set_diag(x, y):
    if len(y.size()) > 1:
        batch_size, dim = y.size()
        mask = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        return mask * y.unsqueeze(-1) + (1. - mask) * x
    else:
        mask = torch.diag(torch.ones_like(y))
        return mask * torch.diag(y) + (1. - mask) * x

