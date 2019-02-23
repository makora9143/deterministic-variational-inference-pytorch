import torch
import torch.nn as nn

import gaussian_variables as gv
import bayes_utils as bu

EPSILON = 1e-6


class Linear(nn.Module):
    def __init__(self, input_features, output_features, prior_type, bias=True):
        """Compute y = x^T W + b
        """
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.prior_type = prior_type

        self.weight = gv.make_weight_matrix((output_features, input_features), prior_type)
        if bias:
            self.bias = gv.make_bias_vector((output_features, input_features), prior_type)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        x_mean = input.mean
        y_mean = x_mean.mm(self.weight.value.mean.t())
        if self.bias is not None:
            y_mean += self.bias.value.mean.unsqueeze(0).expand_as(y_mean)
        x_cov = input.var
        y_cov = linear_covariance(x_mean, x_cov, self.weight, self.bias)
        return gv.GaussianVar(y_mean, y_cov)

    def extra_repr(self):
        return 'prior={}, in_features={}, out_features={}, bias={}'.format(
            self.prior_type[0], self.input_features, self.output_features, self.bias is not None
        )

    def surprise(self):



def linear_covariance(x_mean, x_cov, A, b):
    x_var_diag = torch.diagonal(x_cov, dim1=1, dim2=2)
    xx_mean = x_var_diag + x_mean * x_mean

    term1_diag = xx_mean.mm(A.var.t())

    flat_xCov = x_cov.reshape(-1, A.shape[1])  # [B * x, x]
    xCov_A = flat_xCov.mm(A.mean.t())  # [B * x, y]
    xCov_A = xCov_A.reshape(-1, A.shape[1], A.shape[0])  # [B, x, y]
    xCov_A = torch.transpose(xCov_A, 1, 2)  # [B, y, x]
    xCov_A = xCov_A.reshape(-1, A.shape[1])  # [B*y, x]
    A_xCov_A = xCov_A.mm(A.mean.t())  # [B*y, y]
    A_xCov_A = A_xCov_A.reshape(-1, A.shape[0], A.shape[0])  # [B, y, y]

    term2 = A_xCov_A
    term2_diag = torch.diagonal(term2, dim1=1, dim2=2)

    term3_diag = b.var.unsqueeze(0).expand_as(term2_diag)

    result_diag = term1_diag + term2_diag + term3_diag
    return bu.matrix_set_diag(term2, result_diag)


class LinearCertainActivations(Linear):
    def forward(self, input):
        x_mean = input
        xx = x_mean * x_mean
        y_mean = x_mean.mm(self.weight.value.mean.t())
        diag_cov = xx.mm(self.weight.value.var.t())
        if self.bias is not None:
            y_mean += self.bias.value.mean.unsqueeze(0).expand_as(y_mean)
            diag_cov += self.bias.value.var.unsqueeze(0).expand_as(diag_cov)
        y_cov = bu.matrix_diag(diag_cov)
        return gv.GaussianVar(y_mean, y_cov)


class LinearReLU(Linear):
    def forward(self, input):
        x_var_diag = torch.diagonal(input.var, dim1=1, dim2=2)
        sqrt_x_var_diag = torch.sqrt(x_var_diag)
        mu = input.mean / (sqrt_x_var_diag + EPSILON)

        def relu_covariance(x):
            mu1 = mu.unsqueeze(2)
            mu2 = torch.transpose(mu1, 1, 2)

            s11s22 = x_var_diag.unsqueeze(2) * x_var_diag.unsqueeze(1)
            rho = input.var / torch.sqrt(s11s22)
            rho.clamp_(-1 / (1 + EPSILON), 1 / (1 + EPSILON))

            return input.var * bu.delta(rho, mu1, mu2)

        z_mean = sqrt_x_var_diag * bu.softrelu(mu)
        y_mean = z_mean.mm(self.weight.value.mean.t())
        if self.bias is not None:
            y_mean += self.bias.value.mean
        z_cov = relu_covariance(input)
        y_cov = linear_covariance(z_mean, z_cov, self.weight.value, self.bias.value)
        return gv.GaussianVar(y_mean, y_cov)


class LinearHeaviside(Linear):
    def forward(self, input):
        pass


class RegressionLoss(nn.Module):
    def forward(self, pred, target):

