import torch
import torch.nn as nn
import numpy as np
# from ReluTransformer import ReluTransformer

def compute_bounds(weights_lower, weights_upper, bias_lower, bias_upper, lower_bounds, upper_bounds):
    # weights should have dim: 1 x output_flat x input_flat
    # bias should have dim: 1 x output_flat x 1

    weights_lower = weights_lower.squeeze(0)
    weights_upper = weights_upper.squeeze(0)
    pivot_matrix = torch.zeros_like(weights_lower)

    lower_weights_positive = torch.max(weights_lower, pivot_matrix)
    lower_weights_negative = torch.min(weights_lower, pivot_matrix)

    upper_weights_positive = torch.max(weights_upper, pivot_matrix)
    upper_weights_negative = torch.min(weights_upper, pivot_matrix)

    lower_bound_new = (lower_weights_positive @ lower_bounds.reshape(-1,1) +
                      lower_weights_negative @ upper_bounds.reshape(-1,1)).T +\
                        bias_lower.reshape(1, -1)
    upper_bound_new = (upper_weights_positive @ upper_bounds.reshape(-1,1) +
                      upper_weights_negative @ lower_bounds.reshape(-1,1)).T +\
                        bias_upper.reshape(1, -1)

    return lower_bound_new, upper_bound_new


class OverlapSmooth(nn.Module):
    def __init__(self):
        super(OverlapSmooth, self).__init__()
        self.loss_func = torch.nn.SmoothL1Loss()

    def forward(self, out_lower_bounds, transform_list):
        relu = nn.ReLU()
        relu_out_lower_bounds = relu(-out_lower_bounds)
        target = torch.zeros_like(out_lower_bounds)
        penalty = torch.tensor([0.])
        for transform in transform_list:
            if hasattr(transform, "alpha_lower"):
                alpha_data = transform.alpha.data
                alpha_lower_bound = transform.alpha_lower
                alpha_upper_bound = transform.alpha_upper
                penalty += torch.where((alpha_data < alpha_lower_bound) | (alpha_data > alpha_upper_bound),
                                       torch.abs(alpha_data - alpha_lower_bound) + torch.abs(alpha_data - alpha_upper_bound),
                                       torch.tensor([0.])).sum()
        loss = self.loss_func(relu_out_lower_bounds, target)
        lambda_penalty = 1e-2
        return loss + lambda_penalty*penalty

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.loss_func = torch.nn.MSELoss()

    def forward(self, out_lower_bounds, transform_list):
        relu = nn.ReLU()
        relu_out_lower_bounds = relu(-out_lower_bounds)
        target = torch.zeros_like(out_lower_bounds)
        penalty = torch.tensor([0.])
        for transform in transform_list:
            if hasattr(transform, "alpha_lower"):

                alpha_data = transform.alpha.data
                alpha_lower_bound = transform.alpha_lower
                alpha_upper_bound = transform.alpha_upper
                penalty += torch.where((alpha_data < alpha_lower_bound) | (alpha_data > alpha_upper_bound),
                                       torch.abs(alpha_data - alpha_lower_bound) + torch.abs(alpha_data - alpha_upper_bound),
                                       torch.tensor([0.])).sum()
        loss = self.loss_func(relu_out_lower_bounds, target)
        lambda_penalty = 1e-2
        return loss + lambda_penalty*penalty

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.loss_func = torch.nn.L1Loss()

    def forward(self, out_lower_bounds, transform_list):
        relu = nn.ReLU()
        relu_out_lower_bounds = relu(-out_lower_bounds)
        target = torch.zeros_like(out_lower_bounds)
        penalty = torch.tensor([0.])
        for transform in transform_list:
            if hasattr(transform, "alpha_lower"):
                alpha_data = transform.alpha.data
                alpha_lower_bound = transform.alpha_lower
                alpha_upper_bound = transform.alpha_upper
                penalty += torch.where((alpha_data < alpha_lower_bound) | (alpha_data > alpha_upper_bound),
                                       torch.abs(alpha_data - alpha_lower_bound) + torch.abs(alpha_data - alpha_upper_bound),
                                       torch.tensor([0.])).sum()
        loss = self.loss_func(relu_out_lower_bounds, target)
        lambda_penalty = 1e-2
        return loss + lambda_penalty*penalty

class OverlapHuber(nn.Module):
    def __init__(self, delta=5.0):
        super(OverlapHuber, self).__init__()
        self.loss_func = torch.nn.HuberLoss(delta=delta)

    def forward(self, out_lower_bounds, transform_list):
        relu = nn.ReLU()
        relu_out_lower_bounds = relu(-out_lower_bounds)
        target = torch.zeros_like(out_lower_bounds)
        penalty = torch.tensor([0.])
        for transform in transform_list:
            if hasattr(transform, "alpha_lower"):
                alpha_data = transform.alpha.data
                alpha_lower_bound = transform.alpha_lower
                alpha_upper_bound = transform.alpha_upper
                penalty += torch.where((alpha_data < alpha_lower_bound) | (alpha_data > alpha_upper_bound),
                                       torch.abs(alpha_data - alpha_lower_bound) + torch.abs(alpha_data - alpha_upper_bound),
                                       torch.tensor([0.])).sum()
        loss = self.loss_func(relu_out_lower_bounds, target)
        lambda_penalty = 1e-2
        return loss + lambda_penalty*penalty

def overlap_error(out_lower_bounds):
    relu = nn.ReLU()
    relu_out_lower_bounds = relu(-out_lower_bounds)
    loss = torch.sum(relu_out_lower_bounds)/relu_out_lower_bounds.shape[0]

    return loss

class LossFunction(torch.nn.Module):
    def __init__(self, loss_fcn_label, **kwargs):
        super(LossFunction, self).__init__()
        self.loss_fcn_label = loss_fcn_label
        if loss_fcn_label == "overlap_huber":
            self.loss_func = OverlapHuber(**kwargs)
            self.loss_fcn_label = loss_fcn_label + "_delta_" + str(kwargs["delta"])
        elif loss_fcn_label == "overlap_smooth":
            self.loss_func = OverlapSmooth()
        elif loss_fcn_label == "mse":
            self.loss_func = MSE()
        elif loss_fcn_label == "mae":
            self.loss_func = MAE()

    def forward(self, out_lower_bounds, transform_list):
        return self.loss_func(out_lower_bounds, transform_list)