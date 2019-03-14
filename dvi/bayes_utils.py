import math
import torch

EPSILON = 1e-6
HALF_EPSILON = EPSILON / 2.0

def matrix_set_diag(input, diagonal, dim1=0, dim2=1):
    org_diag = torch.diag_embed(torch.diagonal(input, dim1=dim1, dim2=dim2),
                                dim1=dim1, dim2=dim2)
    new_diag = torch.diag_embed(diagonal, dim1=dim1, dim2=dim2)
    return input - org_diag + new_diag

def gaussian_cdf(x):
    return 0.5 * (1.0 + torch.erf(x * 1 / math.sqrt(2.0)))

def g(rho, mu1, mu2):
    one_plus_sqrt_one_minus_rho_sqr = 1.0 + torch.sqrt(1.0 - rho * rho)
    a = torch.asin(rho) - rho / one_plus_sqrt_one_minus_rho_sqr
    safe_a = torch.abs(a) + HALF_EPSILON
    safe_rho = torch.abs(rho) + EPSILON

    A = a / (2.0 * math.pi)
    sxx = safe_a * one_plus_sqrt_one_minus_rho_sqr / safe_rho
    one_ovr_sxy = (torch.asin(rho) - rho) / (safe_a * safe_rho)

    return A * torch.exp(-(mu1 * mu1 + mu2 * mu2) / (2.0 * sxx) + one_ovr_sxy * mu1 * mu2)


def delta(rho, mu1, mu2):
    return gaussian_cdf(mu1) * gaussian_cdf(mu2) + g(rho, mu1, mu2)


def standard_gaussian(x):
    return 1.0 / math.sqrt(2.0 * math.pi) * torch.exp(- x * x / 2.0)

def softrelu(x):
    return standard_gaussian(x) + x * gaussian_cdf(x)

def anneal(epoch, warmup=14000, anneal=1000):
    return 1.0 * max(min((epoch - warmup) / anneal, 1.0), 0.0)

