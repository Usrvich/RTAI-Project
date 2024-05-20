import torch.nn as nn
import torch
import numpy as np
from general_functions import compute_bounds

class OutputTransformer(nn.Module):
    def __init__(self, prev_layer, target_label, prev_transformer=None):
        super(OutputTransformer, self).__init__()
        self.prev_layer = prev_layer
        self.weights_lower = None
        self.weights_upper = None

        self.bias_lower = None
        self.bias_upper = None

        self.shape_in = prev_layer.bias.shape
        self.shape_out = self.shape_in

        self.target_label = target_label

        self.prev_transformer = prev_transformer

        self.lower_bound = None
        self.upper_bound = None


    # lower bound for x_target - x_other
    def forward(self, lower_bound_prev, upper_bound_prev):
        self.shape_in = self.prev_transformer.shape_out
        self.shape_out = self.shape_in



        self.weights_lower = torch.zeros((self.shape_out, self.shape_in))
        self.weights_upper = torch.zeros((self.shape_out, self.shape_in))

        self.bias_lower = torch.zeros((self.shape_out,1))
        self.bias_upper = torch.zeros((self.shape_out,1))

        self.weights_lower[self.target_label,:] = 1
        self.weights_upper[self.target_label,:] = 1
        for i in range(self.shape_out):
            if i != self.target_label:
                self.weights_upper[i,i] = -1
                self.weights_lower[i,i] = -1
            else:
                self.weights_upper[i,i] = 0
                self.weights_lower[i,i] = 0

        self.weights_lower = self.weights_lower.T
        self.weights_upper = self.weights_upper.T

        self.weights_upper = self.weights_upper.reshape(1, self.shape_out, self.shape_in)
        self.weights_lower = self.weights_lower.reshape(1, self.shape_out, self.shape_in)

        self.bias_lower = self.bias_lower.unsqueeze(0)
        self.bias_upper = self.bias_upper.unsqueeze(0)
        lower_bound_prev = lower_bound_prev.T
        upper_bound_prev = upper_bound_prev.T

        lower_bound, upper_bound = compute_bounds(self.weights_lower, self.weights_upper, self.bias_lower, self.bias_upper, lower_bound_prev, upper_bound_prev)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        return lower_bound, upper_bound
