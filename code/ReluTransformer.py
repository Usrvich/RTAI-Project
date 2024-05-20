import torch.nn as nn
import torch
import numpy as np
from general_functions import compute_bounds

class ReluTransformer(nn.Module):
    def __init__(self, prev_layer, cur_layer, prev_transformer):
        super(ReluTransformer, self).__init__()
        self.prev_layer = prev_layer
        self.cur_layer = cur_layer

        self.weights_lower = None
        self.weights_upper = None

        self.bias_lower = None
        self.bias_upper = None

        self.prev_transformer = prev_transformer

        self.shape_in = None
        self.shape_out = None

        self.negative_slope = None

        self.alpha_initialized = False
        self.alpha = nn.Parameter(requires_grad=True)

        if isinstance(self.cur_layer, torch.nn.LeakyReLU):
            self.negative_slope = torch.tensor([self.cur_layer.negative_slope])
        else:
            self.negative_slope = torch.tensor([0.])

        if self.negative_slope == 0:
            self.slope_upper = None
            self.slope_lower = self.alpha
        if self.negative_slope < 1 and self.negative_slope > 0:
            self.slope_upper = None
            self.slope_lower = self.alpha
        if self.negative_slope == 1:
            self.slope_upper = None
            self.slope_lower = None#self.alpha
        if self.negative_slope > 1:
            self.slope_upper = self.alpha
            self.slope_lower = None

        self.clamp = Clamp()


        self.lower_bound = None
        self.upper_bound = None














    def forward(self, lower_bound_prev, upper_bound_prev):

        self.shape_in = self.prev_transformer.shape_out
        self.shape_out = self.shape_in


        if self.negative_slope == 0:
            self.alpha_lower = torch.tensor([0.])
            self.alpha_upper = torch.tensor([1.])
        elif self.negative_slope < 1 and self.negative_slope > 0:
            self.alpha_lower = self.negative_slope
            self.alpha_upper = torch.tensor([1.])
        elif self.negative_slope == 1:
            self.alpha_lower = torch.tensor([1.])
            self.alpha_upper = torch.tensor([1.])
        elif self.negative_slope > 1:
            self.alpha_lower = torch.tensor([1.])
            self.alpha_upper = self.negative_slope

        if not self.alpha_initialized:
            self.alpha.data = torch.ones_like(lower_bound_prev) * self.alpha_lower
            self.alpha_initialized = True



        self.alpha.data = self.clamp.apply(self.alpha.data, self.alpha_lower, self.alpha_upper)
        # alpha_normalized = (self.alpha - alpha_lower) / (alpha_upper - alpha_lower)

        # analyze the 4 cases of negative_slope
        # case 1: negative_slope = 0
        if self.negative_slope == 0:
            self.slope_upper = upper_bound_prev / (upper_bound_prev - lower_bound_prev).detach()
            self.slope_lower = self.alpha
        # case 2: negative_slope < 1 and negative_slope > 0
        if self.negative_slope < 1 and self.negative_slope > 0:
            self.slope_upper = (upper_bound_prev - self.negative_slope * lower_bound_prev)/ (upper_bound_prev - lower_bound_prev).detach()
            self.slope_lower = self.alpha
        # case 3: negative_slope = 1
        if self.negative_slope == 1:
            self.slope_upper = torch.ones_like(upper_bound_prev)
            self.slope_lower = torch.ones_like(upper_bound_prev)
        # case 4: negative_slope > 1
        if self.negative_slope > 1:
            self.slope_lower = (upper_bound_prev - self.negative_slope * lower_bound_prev) / (upper_bound_prev - lower_bound_prev).detach()
            self.slope_upper = self.alpha


        # compute the weights and bias
        self.weights_lower = torch.ones_like(lower_bound_prev)
        self.weights_upper = torch.ones_like(upper_bound_prev)

        self.weights_lower =(lower_bound_prev <  0) * (upper_bound_prev <  0) * self.negative_slope * self.weights_lower +\
                            (lower_bound_prev >= 0) * (upper_bound_prev >= 0) * self.weights_lower +\
                            (lower_bound_prev <  0) * (upper_bound_prev >= 0) * self.slope_lower * self.weights_lower

        self.weights_upper =(lower_bound_prev <  0) * (upper_bound_prev <  0) * self.negative_slope * self.weights_upper +\
                            (lower_bound_prev >= 0) * (upper_bound_prev >= 0) * self.weights_upper +\
                            (lower_bound_prev <  0) * (upper_bound_prev >= 0) * self.slope_upper * self.weights_upper

        assert torch.any(upper_bound_prev >= lower_bound_prev), "upper_bound_prev is smaller than lower_bound_prev"

        self.weights_lower = self.weights_lower.reshape(-1).diag().unsqueeze(0) # dim: 1 x output_flat x input_flat
        self.weights_upper = self.weights_upper.reshape(-1).diag().unsqueeze(0) # dim: 1 x output_flat x input_flat

        self.bias_lower = torch.zeros_like(lower_bound_prev) # dim: 1 x output_flat x 1
        self.bias_upper = torch.zeros_like(upper_bound_prev) # dim: 1 x output_flat x 1
        if self.negative_slope == 0:
            self.bias_upper = (lower_bound_prev < 0) * (upper_bound_prev < 0) * self.bias_upper +\
                                (lower_bound_prev >= 0) * (upper_bound_prev >= 0) * self.bias_upper +\
                                (lower_bound_prev < 0) * (upper_bound_prev > 0) * (-self.slope_upper * lower_bound_prev)
            self.bias_lower = self.bias_lower.reshape(1, -1, 1) # dim: 1 x output_flat x 1
            self.bias_upper = self.bias_upper.reshape(1, -1, 1) # dim: 1 x output_flat x 1
        elif self.negative_slope < 1 and self.negative_slope > 0:
            self.bias_upper = (lower_bound_prev < 0) * (upper_bound_prev < 0) * self.bias_upper +\
                                (lower_bound_prev >= 0) * (upper_bound_prev >= 0) * self.bias_upper +\
                                (lower_bound_prev < 0) * (upper_bound_prev > 0) * (self.negative_slope*lower_bound_prev-self.slope_upper * lower_bound_prev)
            self.bias_lower = self.bias_lower.reshape(1, -1, 1) # dim: 1 x output_flat x 1
            self.bias_upper = self.bias_upper.reshape(1, -1, 1) # dim: 1 x output_flat x 1
        elif self.negative_slope > 1:
            self.bias_lower = (lower_bound_prev < 0) * (upper_bound_prev < 0) * self.bias_lower +\
                                (lower_bound_prev >= 0) * (upper_bound_prev >= 0) * self.bias_lower +\
                                (lower_bound_prev < 0) * (upper_bound_prev > 0) * (upper_bound_prev - self.slope_lower * upper_bound_prev)
            self.bias_lower = self.bias_lower.reshape(1, -1, 1) # dim: 1 x output_flat x 1
            self.bias_upper = self.bias_upper.reshape(1, -1, 1) # dim: 1 x output_flat x 1

        # # compute the new lower/upper bounds
        lower_bound, upper_bound = compute_bounds(self.weights_lower, self.weights_upper, self.bias_lower, self.bias_upper, lower_bound_prev, upper_bound_prev)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        return lower_bound, upper_bound

class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min=0, max=1):
        return input.clamp(min=min, max=max) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
