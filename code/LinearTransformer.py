import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from general_functions import compute_bounds

class LinearTransformer(nn.Module):
    def __init__(self, prev_layer, cur_layer, prev_transformer=None):
        super(LinearTransformer, self).__init__()

        self.weights_lower = cur_layer.weight.unsqueeze(0).detach()
        self.weights_upper = cur_layer.weight.unsqueeze(0).detach()
        #
        self.bias_lower = cur_layer.bias.reshape(1,-1,1).detach()
        self.bias_upper = cur_layer.bias.reshape(1,-1,1).detach()

        # self.weights_lower = cur_layer.weight.unsqueeze(0)
        # self.weights_upper = cur_layer.weight.unsqueeze(0)

        # self.bias_lower = cur_layer.bias.reshape(1,-1,1)
        # self.bias_upper = cur_layer.bias.reshape(1,-1,1)

        self.shape_in = self.weights_lower.shape[2]
        self.shape_out = self.bias_upper.shape[1]

        self.prev_layer = prev_layer
        self.cur_layer = cur_layer

        self.prev_transformer = prev_transformer

        self.lower_bound = None
        self.upper_bound = None

    def forward(self, lower_bound_prev, upper_bound_prev):

        bias = self.cur_layer.bias.reshape(1,-1,1) # dim: 1 x output_flat x 1
        weights = self.cur_layer.weight.unsqueeze(0) # dim: 1 x output_flat x input_flat
        # print("weights shape: ", self.cur_layer.weight.shape)
        # print("bias shape: ", self.cur_layer.bias.shape)

        lower_bound, upper_bound = compute_bounds(weights, weights, bias, bias, lower_bound_prev, upper_bound_prev)

        self.weights_lower = weights
        self.weights_upper = weights
        self.bias_lower = bias
        self.bias_upper = bias

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # print("lower bound grad in linear: ", self.lower_bound.grad)
        # print("upper bound grad in linear: ", self.upper_bound.grad)


        return lower_bound, upper_bound
