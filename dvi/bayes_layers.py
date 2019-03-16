import torch
import torch.nn as nn

import dvi.bayes_utils as bu
from .variables import make_weight_matrix, make_bias_vector, GaussianVar


class VariationalLinear(nn.Module):
    def __init__(self, input_features, output_features,
                 prior_type='empirical',
                 variance='wider_he', bias=True):
        super(VariationalLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = make_weight_matrix((output_features, input_features), prior_type, variance)
        if bias:
            self.bias = make_bias_vector((output_features, input_features), prior_type, variance)
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        x_mean = input.mean
        y_mean = x_mean.mm(self.weight.q_loc.t())
        if self.bias:
            y_mean += self.bias.q_loc.unsqueeze(0).expand_as(y_mean)
        x_cov = input.var
        y_cov = self.forward_covariance(x_mean, x_cov)
        return GaussianVar(y_mean, y_cov)

    def surprise(self):
        kl = torch.sum(self.weight.surprise())
        if self.bias:
            kl += torch.sum(self.bias.surprise())
        return kl

    def forward_covariance(self, x_mean, x_cov):
        output_dim, input_dim = self.weight.q_loc.shape

        x_var_diag = torch.diagonal(x_cov, dim1=-2, dim2=-1)
        xx_mean = x_var_diag + x_mean * x_mean

        term1_diag = xx_mean.mm(torch.pow(torch.exp(self.weight.log_q_scale), 2).t())

        flat_xCov = x_cov.reshape(-1, input_dim)
        xCov_W = flat_xCov.mm(self.weight.q_loc.t())
        xCov_W = xCov_W.reshape(-1, input_dim, output_dim)
        xCov_W = xCov_W.transpose(1, 2)
        xCov_W = xCov_W.reshape(-1, input_dim)
        W_xCov_W = xCov_W.mm(self.weight.q_loc.t())
        W_xCov_W = W_xCov_W.reshape(-1, output_dim, output_dim)

        term2 = W_xCov_W
        term2_diag = torch.diagonal(term2, dim1=-2, dim2=-1)

        term3_diag = torch.pow(torch.exp(self.bias.log_q_scale), 2).unsqueeze(0).expand_as(term2_diag)

        result_diag = term1_diag + term2_diag + term3_diag
        return bu.matrix_set_diag(term2, result_diag, dim1=-2, dim2=-1)

    def forward_mcmc(self, input, n_samples=None, average=False):
        if n_samples is None:
            n_samples = 1

        repeated_x = input.unsqueeze(0).repeat(n_samples, 1, 1)

        sampled_w = self.weight.sample(n_samples, average)

        h = torch.matmul(repeated_x, sampled_w.transpose(1, 2))

        if self.bias:
            sampled_b = self.bias.sample(n_samples, average)
            h += sampled_b.unsqueeze(1).expand_as(h)
        return h


class VariationalLinearCertainActivations(VariationalLinear):
    def forward(self, input):
        x_mean = input
        xx = x_mean * x_mean
        y_mean = x_mean.mm(self.weight.q_loc.t())
        if self.bias:
            y_mean += self.bias.q_loc.unsqueeze(0).expand_as(y_mean)

        y_cov = xx.mm(torch.pow(torch.exp(self.weight.log_q_scale), 2).t())
        if self.bias:
            y_cov += torch.pow(torch.exp(self.bias.log_q_scale), 2).unsqueeze(0).expand_as(y_cov)
        y_cov = torch.diag_embed(y_cov)
        return GaussianVar(y_mean, y_cov)


class VariationalLinearReLU(VariationalLinear):
    def forward(self, input):
        x_var_diag = torch.diagonal(input.var, dim1=-2, dim2=-1)
        sqrt_x_var_diag = torch.sqrt(x_var_diag)
        mu = input.mean / (sqrt_x_var_diag + bu.EPSILON)

        def relu_covariance(x):
            mu1 = mu.unsqueeze(2)
            mu2 = mu1.transpose(1, 2)

            s11s22 = x_var_diag.unsqueeze(2) * x_var_diag.unsqueeze(1)
            rho = x.var / torch.sqrt(s11s22)
            rho = rho.clamp(-1 / (1 + bu.EPSILON), 1 / (1 + bu.EPSILON))

            return x.var * bu.delta(rho, mu1, mu2)

        z_mean = sqrt_x_var_diag * bu.softrelu(mu)
        y_mean = z_mean.mm(self.weight.q_loc.t())
        if self.bias:
            y_mean += self.bias.q_loc.unsqueeze(0).expand_as(y_mean)
        z_cov = relu_covariance(input)
        y_cov = self.forward_covariance(z_mean, z_cov)
        return GaussianVar(y_mean, y_cov)

    def forward_mcmc(self, input, n_samples=None, average=False):
        if n_samples is None:
            n_samples = 1

        sampled_w = self.weight.sample(n_samples, average)

        h = torch.matmul(input, sampled_w.transpose(1, 2))

        if self.bias:
            sampled_b = self.bias.sample(n_samples, average)
            h += sampled_b.unsqueeze(1).expand_as(h)
        return h
