import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from general_functions import compute_bounds

class InputTransformer(nn.Module):
    def __init__(self, first_constarint):
        super(InputTransformer, self).__init__()
        self.weights_lower = first_constarint["weights_lower"]
        self.weights_upper = first_constarint["weights_upper"]
        self.bias_lower = first_constarint["bias_lower"]
        self.bias_upper = first_constarint["bias_upper"]
        self.lower_bound = first_constarint["bounds_lower"]
        self.upper_bound = first_constarint["bounds_upper"]
        self.shape_in = first_constarint["shape_in"]
        self.shape_out = first_constarint["shape_out"]

    def forward(self, lower_bound_prev, upper_bound_prev):
        return lower_bound_prev, upper_bound_prev