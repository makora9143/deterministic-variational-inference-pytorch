import torch
import torch.nn as nn

import bayes_utils as bu
from variables import make_weight_matrix, make_bias_vector, GaussianVar


class Linear(nn.Module):
    def __init__(self, input_features, output_features, prior_type, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = make_weight_matrix((output_features, input_features), prior_type)
        if bias:
            self.bias = make_bias_vector((output_features, input_features), prior_type)
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        x_mean = input.mean
        y_mean = x_mean.mm(self.weight.q_params[0].t())
        if self.bias:
            y_mean += self.bias.q_params[0].unsqueeze(0).expand_as(y_mean)
        x_cov = input.var
        y_cov = self.forward_covariance(x_mean, x_cov)
        return GaussianVar(y_mean, y_cov)

    def surprise(self):
        kl = torch.sum(self.weight.surprise())
        if self.bias:
            kl += torch.sum(self.bias.surprise())
        return kl

    def forward_covariance(self, x_mean, x_cov):
        output_dim, input_dim = self.weight.q_params[0].shape

        x_var_diag = torch.diagonal(x_cov, dim1=-2, dim2=-1)
        xx_mean = x_var_diag + x_mean * x_mean

        term1_diag = xx_mean.mm(self.weight.q_params[1].t())

        flat_xCov = x_cov.reshape(-1, input_dim)
        xCov_W = flat_xCov.mm(self.weight.q_params[0].t())
        xCov_W = xCov_W.reshape(-1, input_dim, output_dim)
        xCov_W = xCov_W.transpose(1, 2)
        xCov_W = xCov_W.reshape(-1, input_dim)
        W_xCov_W = xCov_W.mm(self.weight.q_params[0].t())
        W_xCov_W = W_xCov_W.reshape(-1, output_dim, output_dim)

        term2 = W_xCov_W
        term2_diag = torch.diagonal(term2, dim1=-2, dim2=-1)

        term3_diag = self.bias.q_params[1].unsqueeze(0).expand_as(term2_diag)

        result_diag = term1_diag + term2_diag + term3_diag
        return bu.matrix_set_diag(term2, result_diag, dim1=-2, dim2=-1)


class LinearCertainActivations(Linear):
    def __init__(self, input_features, output_features, prior_type, bias=True):
        super(LinearCertainActivations, self).__init__(input_features, output_features, prior_type, bias)

    def forward(self, input):
        x_mean = input
        xx = x_mean * x_mean
        y_mean = x_mean.mm(self.weight.q_params[0].t())
        if self.bias:
            y_mean += self.bias.q_params[0].unsqueeze(0).expand_as(y_mean)

        y_cov = xx.mm(self.weight.q_params[1].t())
        if self.bias:
            y_cov += self.bias.q_params[1].unsqueeze(0).expand_as(y_cov)
        y_cov = torch.diag_embed(y_cov)
        return GaussianVar(y_mean, y_cov)


class LinearReLU(Linear):
    def __init__(self, input_features, output_features, prior_type, bias=True):
        super(LinearReLU, self).__init__(input_features, output_features, prior_type, bias)

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
        y_mean = z_mean.mm(self.weight.q_params[0].t())
        if self.bias:
            y_mean += self.bias.q_params[1].unsqueeze(0).expand_as(y_mean)
        z_cov = relu_covariance(input)
        y_cov = self.forward_covariance(z_mean, z_cov)
        return GaussianVar(y_mean, y_cov)

